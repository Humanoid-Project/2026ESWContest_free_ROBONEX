import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "robonex_walking"))

MODULE = (
    Path(__file__).resolve().parents[1]
    / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/mdp/symmetry.py"
)

JOINT_ORDER = [
    "l_hip_yaw_joint",
    "r_hip_yaw_joint",
    "l_hip_pitch_joint",
    "r_hip_pitch_joint",
    "l_hip_roll_joint",
    "r_hip_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_lower_joint",
    "l_ankle_upper_joint",
    "r_ankle_lower_joint",
    "r_ankle_upper_joint",
]
OFFSETS = [0.0, 0.0, 0.1, -0.1, 0.0, 0.0, -0.3857, 0.3857, -0.2057, 0.2057, 0.2057, -0.2057]
SCALES = [0.25, 0.25, 0.25, 0.25, 0.1526, 0.1526, 0.25, 0.25, 0.1317, 0.1317, 0.1317, 0.1317]
TERMS = [
    ("joint_pos_rel", 12),
    ("joint_vel_rel", 12),
    ("imu_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("gait_phase", 2),
    ("actions", 12),
]
HISTORY = 5
EXPECTED_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 10, 11, 8, 9]


def _load_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("robonex_symmetry", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ActionTerm:
    def __init__(self):
        self._joint_names = list(JOINT_ORDER)
        self._offset = torch.tensor(OFFSETS)
        self._scale = torch.tensor(SCALES)


class _ActionManager:
    def __init__(self):
        self._term = _ActionTerm()

    def get_term(self, name):
        return self._term


class _SceneEntityCfg:
    def __init__(self, name="robot", joint_ids=None):
        self.name = name
        self.joint_ids = list(range(len(JOINT_ORDER))) if joint_ids is None else joint_ids


class _TermCfg:
    history_length = HISTORY

    def __init__(self, params=None):
        self.params = {} if params is None else params


class _Articulation:
    joint_names = list(JOINT_ORDER)


class _ObsManager:
    def __init__(self):
        self.active_terms = {
            "policy": [n for n, _ in TERMS],
            "critic": ["base_lin_vel"],
        }
        self.group_obs_term_dim = {
            "policy": [(d * HISTORY,) for _, d in TERMS],
            "critic": [(3 * HISTORY,)],
        }
        self.group_obs_dim = {
            "policy": (sum(d for _, d in TERMS) * HISTORY,),
            "critic": (3 * HISTORY,),
        }
        joint_cfg = _TermCfg({"asset_cfg": _SceneEntityCfg()})
        self._group_obs_term_cfgs = {
            "policy": [joint_cfg, joint_cfg, _TermCfg(), _TermCfg(), _TermCfg(), _TermCfg(), _TermCfg()],
            "critic": [_TermCfg()],
        }


class _Env:
    device = "cpu"

    def __init__(self):
        self.action_manager = _ActionManager()
        self.observation_manager = _ObsManager()
        self.scene = {"robot": _Articulation()}

    @property
    def unwrapped(self):
        return self


@pytest.fixture()
def module_and_env():
    module = _load_module()
    module._CACHE.clear()
    return module, _Env()


def test_permutation_matches_the_mujoco_validated_map(module_and_env):
    module, env = module_and_env
    assert module._joint_permutation(env).tolist() == EXPECTED_PERM


def test_asymmetric_default_pose_is_rejected(module_and_env):
    module, env = module_and_env
    env.action_manager._term._offset = torch.tensor([0.5] + OFFSETS[1:])
    with pytest.raises(RuntimeError, match="mirror-antisymmetric"):
        module._joint_permutation(env)


def test_actions_mirror_is_an_involution(module_and_env):
    module, env = module_and_env
    actions = torch.randn(7, 12)
    _, once = module.compute_symmetric_states(env, actions=actions)
    assert once.shape == (14, 12)
    torch.testing.assert_close(once[:7], actions)
    _, twice = module.compute_symmetric_states(env, actions=once[7:])
    torch.testing.assert_close(twice[7:], actions)


def test_observation_mirror_is_an_involution(module_and_env):
    module, env = module_and_env
    width = env.observation_manager.group_obs_dim["policy"][0]
    obs = torch.randn(7, width)
    layout = module._layout(env)
    blocks, history = layout["groups"]["policy"]
    once = module._mirror_group(obs, blocks, history, layout["perm"])
    twice = module._mirror_group(once, blocks, history, layout["perm"])
    torch.testing.assert_close(twice, obs)
    assert not torch.allclose(once, obs)


def test_each_term_gets_its_own_rule(module_and_env):
    module, env = module_and_env
    layout = module._layout(env)
    blocks, history = layout["groups"]["policy"]
    width = env.observation_manager.group_obs_dim["policy"][0]
    obs = torch.ones(1, width)
    out = module._mirror_group(obs, blocks, history, layout["perm"])[0]

    starts = {}
    cursor = 0
    for name, dim in TERMS:
        starts[name] = cursor
        cursor += dim * HISTORY

    assert torch.all(out[starts["joint_pos_rel"] : starts["joint_pos_rel"] + 60] == -1.0)
    grav = out[starts["projected_gravity"] : starts["projected_gravity"] + 15].view(5, 3)
    torch.testing.assert_close(grav[0], torch.tensor([1.0, -1.0, 1.0]))
    cmd = out[starts["velocity_commands"] : starts["velocity_commands"] + 15].view(5, 3)
    torch.testing.assert_close(cmd[0], torch.tensor([1.0, -1.0, -1.0]))
    gait = out[starts["gait_phase"] : starts["gait_phase"] + 10].view(5, 2)
    torch.testing.assert_close(gait[0], torch.tensor([-1.0, -1.0]))
    imu = out[starts["imu_ang_vel"] : starts["imu_ang_vel"] + 15].view(5, 3)
    torch.testing.assert_close(imu[0], torch.tensor([-1.0, 1.0, -1.0]))


def test_observation_action_joint_order_drift_is_rejected(module_and_env):
    module, env = module_and_env
    reordered = list(range(len(JOINT_ORDER)))
    reordered[0], reordered[2] = reordered[2], reordered[0]
    env.observation_manager._group_obs_term_cfgs["policy"][0].params["asset_cfg"].joint_ids = reordered
    with pytest.raises(RuntimeError, match="wrong observation slots"):
        module._layout(env)


def test_unknown_observation_term_is_rejected(module_and_env):
    module, env = module_and_env
    env.observation_manager.active_terms["policy"] = ["height_scan"]
    env.observation_manager.group_obs_term_dim["policy"] = [(10 * HISTORY,)]
    env.observation_manager.group_obs_dim["policy"] = (10 * HISTORY,)
    env.observation_manager._group_obs_term_cfgs["policy"] = [_TermCfg()]
    with pytest.raises(RuntimeError, match="no mirror rule"):
        module._layout(env)


def test_joint_permutation_applies_per_history_frame(module_and_env):
    module, env = module_and_env
    layout = module._layout(env)
    blocks, history = layout["groups"]["policy"]
    width = env.observation_manager.group_obs_dim["policy"][0]
    obs = torch.zeros(1, width)
    obs[0, 0] = 1.0
    obs[0, 12 + 4] = 2.0
    out = module._mirror_group(obs, blocks, history, layout["perm"])[0]
    assert out[1] == -1.0
    assert out[12 + 5] == -2.0


def test_tensordict_round_trip(module_and_env):
    module, env = module_and_env
    TensorDict = pytest.importorskip("tensordict").TensorDict
    batch = 5
    obs = TensorDict(
        {"policy": torch.randn(batch, 235), "critic": torch.randn(batch, 15)},
        batch_size=[batch],
    )
    actions = torch.randn(batch, 12)

    obs_aug, actions_aug = module.compute_symmetric_states(env, obs=obs, actions=actions)
    assert obs_aug.batch_size[0] == 2 * batch
    assert actions_aug.shape == (2 * batch, 12)
    torch.testing.assert_close(obs_aug["policy"][:batch], obs["policy"])
    torch.testing.assert_close(obs_aug["critic"][:batch], obs["critic"])
    torch.testing.assert_close(actions_aug[:batch], actions)
    assert not torch.allclose(obs_aug["policy"][batch:], obs["policy"])

    mirrored = TensorDict(
        {"policy": obs_aug["policy"][batch:].clone(), "critic": obs_aug["critic"][batch:].clone()},
        batch_size=[batch],
    )
    twice, actions_twice = module.compute_symmetric_states(
        env, obs=mirrored, actions=actions_aug[batch:].clone()
    )
    torch.testing.assert_close(twice["policy"][batch:], obs["policy"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(twice["critic"][batch:], obs["critic"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actions_twice[batch:], actions, atol=1e-6, rtol=1e-6)


def test_one_sided_calls(module_and_env):
    module, env = module_and_env
    TensorDict = pytest.importorskip("tensordict").TensorDict
    obs = TensorDict({"policy": torch.randn(3, 235), "critic": torch.randn(3, 15)}, batch_size=[3])
    obs_aug, actions_aug = module.compute_symmetric_states(env, obs=obs, actions=None)
    assert obs_aug.batch_size[0] == 6 and actions_aug is None
    obs_aug, actions_aug = module.compute_symmetric_states(env, obs=None, actions=torch.randn(3, 12))
    assert obs_aug is None and actions_aug.shape == (6, 12)
