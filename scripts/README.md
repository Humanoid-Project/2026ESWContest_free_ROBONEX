# Script parameter inventory

Current source inventory (2026-09-05). This page documents accepted arguments; it is not a claim that training has been run or validated. `None` means no CLI override, not necessarily the final task value. `--task` must identify a registered task. All parsers also provide `-h` / `--help`.

## `export_policy_manifest.py`

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `policy` | `Required` | Exported ONNX policy path.  |
| `--output` | `None` | Manifest destination; defaults to policy_manifest.json beside the policy.  |
| `--description-root` | `None` | Description checkout override; otherwise resolver/environment/sibling lookup.  |
| `--common-root` | `None` | Local common checkout used for Git provenance; still needed despite tag-pip installation.  |
| `--description-model` | `mujoco/robot/scene.xml` | Model path relative to the description checkout.  |

Schema 2 fingerprints the policy, MuJoCo model bundle, common runtime source, and training source.

## `list_envs.py`

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `--keyword` | `None` | Keyword to filter environments.  |

## `random_agent.py`

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `--disable_fabric` | `False` | Disable fabric and use USD I/O operations.  |
| `--num_envs` | `None` | Number of environments to simulate.  |
| `--task` | `None` | Name of the task.  |

## `rsl_rl/cli_args.py`

Shared by `rsl_rl/train.py` and `rsl_rl/play.py`; not a standalone executable.

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `--experiment_name` | `None` | Name of the experiment folder where logs will be stored. WARNING: currently parsed but not applied by update_rsl_rl_cfg; the task configuration determines the experiment name.  |
| `--run_name` | `None` | Run name suffix to the log directory.  |
| `--resume` | `False` | Whether to resume from a checkpoint.  |
| `--load_run` | `None` | Name of the run folder to resume from.  |
| `--checkpoint` | `None` | Checkpoint file to resume from.  |
| `--logger` | `None` | Logger module to use. Choices: `{'wandb', 'tensorboard', 'neptune'}`. |
| `--log_project_name` | `None` | Name of the logging project when using wandb or neptune.  |

## `rsl_rl/play.py`

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `--video` | `False` | Record videos during training.  |
| `--video_length` | `200` | Length of the recorded video (in steps).  |
| `--disable_fabric` | `False` | Disable fabric and use USD I/O operations.  |
| `--num_envs` | `None` | Number of environments to simulate.  |
| `--task` | `None` | Name of the task.  |
| `--agent` | `rsl_rl_cfg_entry_point` | Name of the RL agent configuration entry point.  |
| `--seed` | `None` | Seed used for the environment  |
| `--use_pretrained_checkpoint` | `False` | Use the pre-trained checkpoint from Nucleus.  |
| `--real-time` | `False` | Run in real-time, if possible.  |

## `rsl_rl/train.py`

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `--video` | `False` | Record videos during training.  |
| `--video_length` | `200` | Length of the recorded video (in steps).  |
| `--video_interval` | `2000` | Interval between video recordings (in steps).  |
| `--num_envs` | `None` | Number of environments to simulate.  |
| `--task` | `None` | Name of the task.  |
| `--agent` | `rsl_rl_cfg_entry_point` | Name of the RL agent configuration entry point.  |
| `--seed` | `None` | Seed used for the environment  |
| `--max_iterations` | `None` | RL Policy training iterations.  |
| `--distributed` | `False` | Run training with multiple GPUs or nodes.  |
| `--export_io_descriptors` | `False` | Export IO descriptors.  |
| `--ray-proc-id` / `-rid` | `None` | Automatically configured by Ray integration, otherwise None.  |

## `zero_agent.py`

| Parameter | Parser default | Meaning / accepted values |
| --- | --- | --- |
| `--disable_fabric` | `False` | Disable fabric and use USD I/O operations.  |
| `--num_envs` | `None` | Number of environments to simulate.  |
| `--task` | `None` | Name of the task.  |

## Inherited Isaac Lab AppLauncher arguments

The following apply to `zero_agent.py`, `random_agent.py`, `rsl_rl/train.py`, and `rsl_rl/play.py`, not to the manifest exporter. These are the locally installed Isaac Lab definitions; they can change independently of this repository. Environment variables and launcher resolution can change effective values.

| Parameter | Parser default | Meaning |
| --- | --- | --- |
| `--headless` | `False` | Disable display |
| `--livestream` | `-1` | Defer to environment; explicit values: `0`, `1`, `2` |
| `--enable_cameras` | `False` | Enable camera dependencies; video mode also enables this |
| `--xr` | `False` | Enable XR |
| `--device` | `cuda:0` | Simulation device: `cpu`, `cuda`, `cuda:N` |
| `--cpu` | `False` | Deprecated; rejected. Use `--device cpu` |
| `--verbose` / `--info` | `False` | Kit logging levels |
| `--experience` | empty string | Auto-selected Kit experience, or explicit path |
| `--rendering_mode` | `None` | `performance`, `balanced`, `quality` |
| `--kit_args` | empty string | Additional Kit arguments as one string |
| `--anim_recording_enabled` | `False` | Record USD animation |
| `--anim_recording_start_time` | `0` | Animation start, seconds |
| `--anim_recording_stop_time` | `10` | Animation stop, seconds |

## Configuration values are separate from CLI flags

`train.py` and `play.py` also accept Hydra configuration overrides through their remaining arguments. The full configuration namespace is defined by the selected environment/agent, not by a fixed argparse list. Review `source/*/*/tasks/manager_based/*/*_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py`, and `robot_contract.py` for rewards, episode duration, environment count, physics timestep, decimation, gains, observation noise, action normalization, and PPO settings. `ROBONEX_DESCRIPTION_ROOT` overrides the description checkout. `ROBONEX_COMMON_ROOT` is still used by the manifest exporter's checkout-provenance path.

`zero_agent.py` sends a zero **normalized action**, not a mechanical-zero joint target: the current action offset is the midpoint of each clipped joint range. Neither agent script is a hardware controller.
