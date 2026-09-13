# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import math
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import robonex_walking.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class DiagnosticVecEnvWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, clip_actions=None):
        super().__init__(env, clip_actions=clip_actions)
        term = self.unwrapped.action_manager.get_term("joint_pos")
        self.diagnostic_joint_names = tuple(term._joint_names)
        self.diagnostic_scales = torch.as_tensor(
            term._scale, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, self.num_actions)[0].abs().clone()
        if len(self.diagnostic_joint_names) != self.num_actions:
            raise ValueError("Policy diagnostics require one joint-position action per joint")
        self.diagnostic_counts = torch.zeros(self.num_actions + 2, device=self.device)
        # The target fence is asymmetric per joint, so runner-clip counts alone
        # cannot see a joint whose target saturates far below the runner clip.
        self.diagnostic_offsets = torch.as_tensor(
            term._offset, device=self.device, dtype=torch.float32
        ).expand(self.num_envs, self.num_actions)[0].clone()
        target_clip = getattr(term, "_clip", None)
        self.diagnostic_target_clip = (
            target_clip[0].detach().clone().to(self.device) if target_clip is not None else None
        )
        self.diagnostic_target_counts = torch.zeros(self.num_actions + 1, device=self.device)
        # split the fence side: an eversion tail loads the low fence of one crank and the
        # high fence of its mirror, which an unsigned fraction cannot distinguish
        self.diagnostic_target_lo = torch.zeros(self.num_actions, device=self.device)
        self.diagnostic_target_hi = torch.zeros(self.num_actions, device=self.device)
        self.diagnostic_samples = 0
        scene = getattr(self.unwrapped, "scene", None)
        sensors = getattr(scene, "sensors", {}) if scene is not None else {}
        self.walk_metrics = None
        if "contact_forces" in sensors:
            from robonex_walking.tasks.manager_based.robonex_walking.mdp.walk_metrics import (
                WalkMetrics,
            )
            from robonex_walking.tasks.manager_based.robonex_walking.robot_contract import (
                FOOT_ORIGIN_REST_HEIGHT,
            )

            self.walk_metrics = WalkMetrics(
                self.unwrapped, FOOT_ORIGIN_REST_HEIGHT, self.unwrapped.step_dt
            )

    def step(self, actions):
        with torch.no_grad():
            self.unwrapped.raw_policy_action = actions.detach().clone()
            if self.walk_metrics is not None:
                self.walk_metrics.record_action(actions.detach())
            reached = torch.zeros_like(actions, dtype=torch.bool)
            if self.clip_actions is not None:
                reached = actions.abs() >= self.clip_actions
            self.diagnostic_counts[:-2] += reached.sum(dim=0)
            self.diagnostic_counts[-2] += reached.any(dim=1).sum()
            self.diagnostic_counts[-1] += (~torch.isfinite(actions)).sum()
            if self.diagnostic_target_clip is not None:
                target = (
                    torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
                    * self.diagnostic_scales
                    + self.diagnostic_offsets
                )
                below = target <= self.diagnostic_target_clip[:, 0]
                above = target >= self.diagnostic_target_clip[:, 1]
                fenced = below | above
                self.diagnostic_target_counts[:-1] += fenced.sum(dim=0)
                self.diagnostic_target_counts[-1] += fenced.any(dim=1).sum()
                self.diagnostic_target_lo += below.sum(dim=0)
                self.diagnostic_target_hi += above.sum(dim=0)
            self.diagnostic_samples += actions.shape[0]
        result = super().step(actions)
        if self.walk_metrics is not None:
            self.walk_metrics.update(self.unwrapped)
        return result

    def take_action_diagnostics(self):
        samples = max(1, self.diagnostic_samples)
        counts = self.diagnostic_counts.detach().cpu().tolist()
        self.diagnostic_counts.zero_()
        self.diagnostic_samples = 0
        values = {
            f"Policy/runner_clip_fraction/{name}": counts[index] / samples
            for index, name in enumerate(self.diagnostic_joint_names)
        }
        values["Policy/runner_clip_any_fraction"] = counts[-2] / samples
        values["Policy/runner_clip_element_fraction"] = sum(counts[:-2]) / (
            samples * len(self.diagnostic_joint_names)
        )
        values["Policy/raw_action_nonfinite_fraction"] = counts[-1] / (
            samples * len(self.diagnostic_joint_names)
        )
        if self.diagnostic_target_clip is not None:
            target_counts = self.diagnostic_target_counts.detach().cpu().tolist()
            self.diagnostic_target_counts.zero_()
            lo_counts = self.diagnostic_target_lo.detach().cpu().tolist()
            hi_counts = self.diagnostic_target_hi.detach().cpu().tolist()
            self.diagnostic_target_lo.zero_()
            self.diagnostic_target_hi.zero_()
            for index, name in enumerate(self.diagnostic_joint_names):
                values[f"Policy/target_clip_fraction/{name}"] = target_counts[index] / samples
                values[f"Policy/target_clip_lo_fraction/{name}"] = lo_counts[index] / samples
                values[f"Policy/target_clip_hi_fraction/{name}"] = hi_counts[index] / samples
            values["Policy/target_clip_any_fraction"] = target_counts[-1] / samples
            values["Policy/target_clip_element_fraction"] = sum(target_counts[:-1]) / (
                samples * len(self.diagnostic_joint_names)
            )
        if self.walk_metrics is not None:
            values.update(self.walk_metrics.take_log())
        return values


class DiagnosticOnPolicyRunner(OnPolicyRunner):
    def log(self, locs, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        policy = self.alg.policy
        with torch.no_grad():
            if policy.noise_std_type == "log":
                std = policy.log_std.detach().exp()
            else:
                std = policy.std.detach()
            std = std.cpu().reshape(-1)
            scales = self.env.diagnostic_scales.detach().cpu()
            names = self.env.diagnostic_joint_names
            if std.numel() != len(names):
                raise ValueError("Policy diagnostics require state-independent per-joint std")
            physical_std = std * scales * (180.0 / math.pi)
            values = self.env.take_action_diagnostics()
            for index, name in enumerate(names):
                values[f"Policy/std/{name}"] = std[index].item()
                values[f"Policy/preclip_target_std_deg/{name}"] = physical_std[index].item()
            valid = bool(torch.isfinite(std).all() and (std > 0).all())
            values["Policy/std_valid"] = float(valid)
            values["Policy/std_max"] = std.max().item()
            values["Policy/std_spread"] = (std.max() / std.min()).item() if valid else math.inf
            values["Policy/preclip_target_std_deg_max"] = physical_std.max().item()
            values["Policy/std_gate_pass"] = float(
                valid and std.max() < 3.0 and std.max() / std.min() < 10.0
                and physical_std.max() < 10.0
            )
        for name, value in values.items():
            self.writer.add_scalar(name, value, locs["it"])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    wrapper = DiagnosticVecEnvWrapper if agent_cfg.class_name == "OnPolicyRunner" else RslRlVecEnvWrapper
    env = wrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = DiagnosticOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
