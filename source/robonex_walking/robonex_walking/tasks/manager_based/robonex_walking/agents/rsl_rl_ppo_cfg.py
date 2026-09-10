# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from robonex_common.limits import RUNNER_ACTION_CLIP

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "robonex_walking"
    clip_actions = RUNNER_ACTION_CLIP

    policy = RslRlPpoActorCriticCfg(
        init_noise_std = 0.5,
        noise_std_type = "log",
        actor_obs_normalization = True,
        critic_obs_normalization = True,
        actor_hidden_dims = [256, 128, 128],
        critic_hidden_dims = [256, 128, 128],
        activation = "elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef = 1.0,
        use_clipped_value_loss = True,
        clip_param = 0.2,
        entropy_coef = 0.005,
        num_learning_epochs = 5,
        num_mini_batches = 4,
        learning_rate = 1.0e-3,
        schedule = "adaptive",
        gamma = 0.99,
        lam = 0.95,
        desired_kl = 0.01,
        max_grad_norm = 1.0,
    )


@configclass
class ClosedLoopPPORunnerCfg(PPORunnerCfg):

    experiment_name = "robonex_walking_closed_loop"
