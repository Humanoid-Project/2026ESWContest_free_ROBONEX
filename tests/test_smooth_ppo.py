import ast
import copy
import importlib.util
import inspect
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
TensorDict = pytest.importorskip("tensordict").TensorDict

from robonex_common.runtime import OBSERVATION_HISTORY_LENGTH, OBSERVATION_SIZE, OBSERVATION_TERM_SIZES

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source/robonex_walking/robonex_walking"
TASK = PKG / "tasks/manager_based/robonex_walking"
SMOOTH = PKG / "algorithms/smooth_ppo.py"
SYMMETRY = TASK / "mdp/symmetry.py"
ENV_CFG = TASK / "robonex_walking_v2_env_cfg.py"
AGENT_CFG = TASK / "agents/rsl_rl_ppo_cfg.py"
CONTRACT = TASK / "robot_contract_v2.py"

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
FREQS = (12.5, 18.75, 25.0)
DT = 0.02


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMOOTH_PPO = _load(SMOOTH, "robonex_smooth_ppo")
SYM = _load(SYMMETRY, "robonex_symmetry_for_smooth_ppo")
CONTRACT_V2 = _load(CONTRACT, "robonex_contract_v2_for_smooth_ppo")


def _policy_terms():
    tree = ast.parse(ENV_CFG.read_text())
    obs_cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    policy_cfg = next(n for n in obs_cfg.body if isinstance(n, ast.ClassDef) and n.name == "PolicyCfg")
    names = []
    for node in policy_cfg.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "ObsTerm"
        ):
            names.append(node.targets[0].id)
    return names


def _real_terms():
    names = _policy_terms()
    contract = [("actions" if n == "last_action" else n, size) for n, size in OBSERVATION_TERM_SIZES]
    assert names == [n for n, _ in contract]
    return contract


TERMS = _real_terms()
HISTORY = OBSERVATION_HISTORY_LENGTH


class _ActionTerm:
    def __init__(self):
        self._joint_names = list(JOINT_ORDER)
        self._offset = torch.tensor([CONTRACT_V2.ACTION_OFFSETS[n] for n in JOINT_ORDER])
        self._scale = torch.tensor([CONTRACT_V2.ACTION_SCALES[n] for n in JOINT_ORDER]).expand(4, -1)


class _ActionManager:
    def __init__(self):
        self._term = _ActionTerm()
        self.total_action_dim = len(JOINT_ORDER)

    def get_term(self, name):
        assert name == "joint_pos"
        return self._term


class _SceneEntityCfg:
    name = "robot"
    joint_ids = list(range(len(JOINT_ORDER)))


class _TermCfg:
    history_length = HISTORY
    flatten_history_dim = True

    def __init__(self, params=None):
        self.params = {} if params is None else params


class _Articulation:
    joint_names = list(JOINT_ORDER)


class _ObsManager:
    def __init__(self):
        self.active_terms = {"policy": [n for n, _ in TERMS], "critic": ["base_lin_vel"]}
        self.group_obs_term_dim = {"policy": [(d * HISTORY,) for _, d in TERMS], "critic": [(3 * HISTORY,)]}
        self.group_obs_dim = {"policy": (sum(d for _, d in TERMS) * HISTORY,), "critic": (3 * HISTORY,)}
        joint_cfg = _TermCfg({"asset_cfg": _SceneEntityCfg()})
        self._group_obs_term_cfgs = {
            "policy": [joint_cfg if n in ("joint_pos_rel", "joint_vel_rel") else _TermCfg() for n, _ in TERMS],
            "critic": [_TermCfg()],
        }


class _Env:
    device = "cpu"
    step_dt = DT

    def __init__(self):
        self.action_manager = _ActionManager()
        self.observation_manager = _ObsManager()
        self.scene = {"robot": _Articulation()}

    @property
    def unwrapped(self):
        return self


OBS_GROUPS = {"policy": ["policy"], "critic": ["policy", "critic"]}


def _obs(batch, generator=None):
    return TensorDict(
        {
            "policy": torch.randn(batch, OBSERVATION_SIZE, generator=generator),
            "critic": torch.randn(batch, 3 * HISTORY, generator=generator),
        },
        batch_size=[batch],
    )


def _policy(seed=0, hidden=(32, 32)):
    from rsl_rl.modules import ActorCritic

    torch.manual_seed(seed)
    return ActorCritic(
        _obs(4),
        OBS_GROUPS,
        len(JOINT_ORDER),
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=list(hidden),
        critic_hidden_dims=list(hidden),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="log",
    )


def _symmetry_cfg(env):
    SYM._CACHE.clear()
    return {
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "data_augmentation_func": SYM.compute_symmetric_states,
        "mirror_loss_coeff": 0.0,
        "_env": env,
    }


def _alg(cls, policy, env, **extra):
    return cls(
        policy,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1.0e-3,
        entropy_coef=0.008,
        device="cpu",
        symmetry_cfg=_symmetry_cfg(env),
        **extra,
    )


def _rollout_and_update(alg, seed, num_envs=16, steps=6):
    torch.manual_seed(seed)
    data = torch.Generator().manual_seed(seed + 1)
    obs = _obs(num_envs, data)
    alg.init_storage("rl", num_envs, steps, obs, [len(JOINT_ORDER)])
    alg.policy.train()
    for _ in range(steps):
        alg.act(obs)
        obs = _obs(num_envs, data)
        rewards = torch.randn(num_envs, generator=data)
        dones = (torch.rand(num_envs, generator=data) < 0.1).float()
        alg.process_env_step(obs, rewards, dones, {})
    alg.compute_returns(obs)
    return alg.update()


def _mirror_obs(x):
    layout = SYM._layout(_Env())
    blocks, history = layout["groups"]["policy"]
    return SYM._mirror_group(x, blocks, history, layout["perm"])


def _mirror_actions(a):
    perm = SYM._layout(_Env())["perm"]
    return -a[:, perm]


def _analytic(weight, lo, scales):
    total = 0.0
    for axis in range(3):
        for freq in FREQS:
            h = torch.zeros(weight.shape[0], dtype=torch.complex128)
            for k in range(HISTORY):
                phase = torch.tensor(-2j * math.pi * freq * DT * (HISTORY - 1 - k), dtype=torch.complex128).exp()
                h = h + weight[:, lo + 3 * k + axis].to(torch.complex128) * phase
            total += ((scales.double() * h).abs() ** 2).sum().item()
    return total


def test_installed_rsl_rl_matches_the_copied_update():
    assert SMOOTH_PPO.installed_rsl_rl_version() == SMOOTH_PPO.RSL_RL_VERSION


def test_version_drift_is_rejected(monkeypatch):
    monkeypatch.setattr(SMOOTH_PPO, "installed_rsl_rl_version", lambda: "3.2.0")
    with pytest.raises(RuntimeError, match="re-diff"):
        _alg(SMOOTH_PPO.SmoothPPO, _policy(), _Env())


def test_class_is_resolvable_by_the_runner():
    import rsl_rl.runners.on_policy_runner as runner

    assert eval("SmoothPPO", vars(runner)) is SMOOTH_PPO.SmoothPPO
    assert eval("PPO", vars(runner)) is runner.PPO


def test_gyro_slice_matches_the_real_ver2_layout():
    lo, hi, history, width = SMOOTH_PPO.resolve_obs_term_slice(_Env().observation_manager, ["policy"], "imu_ang_vel")
    assert (lo, hi, history, width) == (120, 135, 5, 235)
    assert width == OBSERVATION_SIZE


def test_gyro_slice_accounts_for_group_offset_and_rejects_bad_layouts():
    manager = _Env().observation_manager
    lo, hi, _, width = SMOOTH_PPO.resolve_obs_term_slice(manager, ["critic", "policy"], "imu_ang_vel")
    assert (lo, hi, width) == (135, 150, 250)
    manager._group_obs_term_cfgs["policy"][2].flatten_history_dim = False
    with pytest.raises(RuntimeError, match="unflattened"):
        SMOOTH_PPO.resolve_obs_term_slice(manager, ["policy"], "imu_ang_vel")
    with pytest.raises(RuntimeError, match="not in the actor"):
        SMOOTH_PPO.resolve_obs_term_slice(_Env().observation_manager, ["critic"], "imu_ang_vel")


def test_history_frames_are_oldest_first():
    import numpy as np
    from robonex_common.runtime import ObservationHistory

    buffer_path = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py")
    if not buffer_path.exists():
        pytest.skip("Isaac Lab source not found")
    buffer = _load(buffer_path, "isaaclab_circular_buffer_for_smooth_ppo").CircularBuffer(HISTORY, 1, "cpu")
    for step in range(HISTORY + 2):
        buffer.append(torch.full((1, 3), float(step)))
    assert buffer.buffer[0, :, 0].tolist() == [float(s) for s in range(2, HISTORY + 2)]

    history = ObservationHistory()
    for step in range(HISTORY + 2):
        frame = [np.zeros(size) for _, size in OBSERVATION_TERM_SIZES]
        frame[2] = np.full(3, float(step))
        history.append(np.concatenate(frame))
    gyro = history.observation()[120:135].reshape(HISTORY, 3)
    assert gyro[:, 0].tolist() == [float(s) for s in range(2, HISTORY + 2)]


def test_action_scales_resolve_in_action_order_and_reject_mismatch():
    manager = _Env().action_manager
    live = SMOOTH_PPO.resolve_action_scales(manager, "joint_pos")
    by_name = SMOOTH_PPO.resolve_action_scales(manager, "joint_pos", dict(CONTRACT_V2.ACTION_SCALES))
    torch.testing.assert_close(live, by_name)
    assert live.tolist() == pytest.approx(
        [0.25, 0.25, 0.25, 0.25, 0.148886, 0.148886, 0.1648, 0.1648, 0.219621, 0.160604, 0.219621, 0.160604]
    )
    wrong = dict(CONTRACT_V2.ACTION_SCALES)
    wrong["l_knee_pitch_joint"] = 0.3
    with pytest.raises(RuntimeError, match="differ from the live"):
        SMOOTH_PPO.resolve_action_scales(manager, "joint_pos", wrong)


def test_directions_cover_cos_and_sin_without_the_zero_nyquist_sine():
    dirs = SMOOTH_PPO.hf_directions(FREQS, DT, HISTORY, 3)
    assert dirs.shape == (15, 15)
    nyquist_cos = dirs[-3:].view(3, HISTORY, 3)
    torch.testing.assert_close(nyquist_cos[0, :, 0], torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]))
    assert torch.all(nyquist_cos[0, :, 1:] == 0)


def test_zero_coefficients_match_stock_ppo_exactly():
    from rsl_rl.algorithms import PPO

    base = _policy(seed=3)
    stock = _alg(PPO, copy.deepcopy(base), _Env())
    smooth = _alg(SMOOTH_PPO.SmoothPPO, copy.deepcopy(base), _Env(), hf_gyro_coef=0.0, lcp_coef=0.0)
    stock_losses = _rollout_and_update(stock, seed=11)
    smooth_losses = _rollout_and_update(smooth, seed=11)
    assert stock_losses == smooth_losses
    for (name, a), (_, b) in zip(stock.policy.state_dict().items(), smooth.policy.state_dict().items()):
        assert torch.equal(a, b), name
    assert stock.learning_rate == smooth.learning_rate


def test_penalties_change_the_update_and_are_logged():
    base = _policy(seed=3)
    plain = _alg(SMOOTH_PPO.SmoothPPO, copy.deepcopy(base), _Env())
    penalised = _alg(SMOOTH_PPO.SmoothPPO, copy.deepcopy(base), _Env(), hf_gyro_coef=0.05, lcp_coef=0.002, hf_samples=8)
    _rollout_and_update(plain, seed=11)
    losses = _rollout_and_update(penalised, seed=11)
    assert losses["hf_gyro"] > 0 and math.isfinite(losses["hf_gyro"])
    assert losses["lcp"] > 0 and math.isfinite(losses["lcp"])
    actor = [(a, b) for (n, a), (_, b) in zip(plain.policy.state_dict().items(), penalised.policy.state_dict().items())
             if n.startswith("actor.")]
    assert any(not torch.equal(a, b) for a, b in actor)


def test_hf_gyro_needs_the_environment():
    from rsl_rl.algorithms import PPO

    with pytest.raises(ValueError, match="symmetry_cfg"):
        SMOOTH_PPO.SmoothPPO(_policy(), hf_gyro_coef=0.05, device="cpu")
    assert isinstance(SMOOTH_PPO.SmoothPPO(_policy(), device="cpu"), PPO)


def test_step_dt_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="step_dt"):
        _alg(SMOOTH_PPO.SmoothPPO, _policy(), _Env(), hf_gyro_coef=0.05, step_dt=0.01)


def _linear_alg(weight, normalize=False):
    policy = _policy()
    policy.actor = torch.nn.Sequential(torch.nn.Linear(OBSERVATION_SIZE, len(JOINT_ORDER)))
    with torch.no_grad():
        policy.actor[0].weight.copy_(weight)
        policy.actor[0].bias.normal_()
        if normalize:
            policy.actor_obs_normalizer._mean.normal_()
            policy.actor_obs_normalizer._std.uniform_(0.2, 2.0)
    return _alg(SMOOTH_PPO.SmoothPPO, policy, _Env(), hf_gyro_coef=0.05)


@pytest.mark.parametrize("normalize", [False, True])
def test_linear_policy_matches_the_analytic_value(normalize):
    torch.manual_seed(5)
    weight = torch.randn(len(JOINT_ORDER), OBSERVATION_SIZE) * 0.1
    alg = _linear_alg(weight, normalize)
    x = torch.randn(64, OBSERVATION_SIZE)
    got = alg.hf_gyro_penalty_on(x).item()
    norm = alg.policy.actor_obs_normalizer
    raw_weight = weight / (norm._std + norm.eps)
    want = _analytic(raw_weight, 120, alg.hf_scales)
    assert got == pytest.approx(want, rel=1e-4)


def _per_axis_form():
    dirs = SMOOTH_PPO.hf_directions(FREQS, DT, HISTORY, 1).double()
    return dirs.T @ dirs


def test_per_axis_form_is_full_rank_so_dc_gain_has_a_small_nonzero_floor():
    form = _per_axis_form()
    eig = torch.linalg.eigvalsh(form)
    assert eig.min().item() > 0.02
    ones = torch.ones(HISTORY, dtype=torch.float64)
    floor = 1.0 / (ones @ torch.linalg.solve(form, ones)).item()
    assert floor == pytest.approx(0.0066, abs=2e-4)
    assert ((ones / HISTORY) @ form @ (ones / HISTORY)).item() == pytest.approx(0.0869, abs=2e-4)
    assert form[-1, -1].item() == pytest.approx(3.0)


def _gyro_weight(taps):
    weight = torch.zeros(len(JOINT_ORDER), OBSERVATION_SIZE)
    for k, tap in enumerate(taps):
        weight[:, 120 + 3 * k : 123 + 3 * k] = float(tap)
    return weight


def test_dc_gain_through_a_smooth_window_costs_little_and_alternating_gain_costs_a_lot():
    ones = torch.ones(HISTORY, dtype=torch.float64)
    window = torch.linalg.solve(_per_axis_form(), ones)
    window = (window / window.sum()).tolist()
    assert window == pytest.approx([0.122, 0.240, 0.276, 0.240, 0.122], abs=1e-3)
    x = torch.randn(32, OBSERVATION_SIZE)
    dc_alg = _linear_alg(_gyro_weight(window))
    norm = dc_alg.policy.actor_obs_normalizer
    raw = _gyro_weight(window) / (norm._std + norm.eps)
    dc_gain = raw[:, 120:135].view(-1, HISTORY, 3).sum(1).double()
    dc_gain_sq = (dc_alg.hf_scales.double()[:, None] * dc_gain).square().sum()
    dc = dc_alg.hf_gyro_penalty_on(x).item()
    assert dc == pytest.approx(0.0066 * dc_gain_sq.item(), rel=0.03)
    alternating = [(-1.0) ** k * 0.2 for k in range(HISTORY)]
    hf_alg = _linear_alg(_gyro_weight(alternating))
    hf = hf_alg.hf_gyro_penalty_on(x).item()
    assert hf > 100.0 * dc
    raw_alt = _gyro_weight(alternating) / (norm._std + norm.eps)
    assert hf == pytest.approx(_analytic(raw_alt, 120, hf_alg.hf_scales), rel=1e-4)


def test_penalty_ignores_non_gyro_inputs():
    weight = torch.randn(len(JOINT_ORDER), OBSERVATION_SIZE)
    weight[:, 120:135] = 0.0
    assert _linear_alg(weight).hf_gyro_penalty_on(torch.randn(16, OBSERVATION_SIZE)).item() < 1e-8


def test_penalty_is_mirror_invariant():
    alg = _alg(SMOOTH_PPO.SmoothPPO, _policy(seed=7, hidden=(64, 64)), _Env(), hf_gyro_coef=0.05)
    x = torch.randn(32, OBSERVATION_SIZE)
    mirrored_x = _mirror_obs(x)
    torch.testing.assert_close(_mirror_obs(mirrored_x), x)

    def mirrored_policy(obs):
        return _mirror_actions(alg.actor_mean(_mirror_obs(obs)))

    lo, hi = alg.hf_slice
    on_mirrored_state = SMOOTH_PPO.hf_gain_penalty(alg.actor_mean, mirrored_x, lo, hi, alg.hf_dirs, alg.hf_eps,
                                                  alg.hf_scales)
    of_mirrored_policy = SMOOTH_PPO.hf_gain_penalty(mirrored_policy, x, lo, hi, alg.hf_dirs, alg.hf_eps,
                                                   alg.hf_scales)
    torch.testing.assert_close(on_mirrored_state, of_mirrored_policy, rtol=1e-4, atol=1e-7)

    def symmetric_policy(obs):
        return 0.5 * (alg.actor_mean(obs) + mirrored_policy(obs))

    per_sample = [
        SMOOTH_PPO.hf_gain_penalty(symmetric_policy, x[i : i + 1], lo, hi, alg.hf_dirs, alg.hf_eps, alg.hf_scales)
        for i in range(4)
    ]
    per_mirror = [
        SMOOTH_PPO.hf_gain_penalty(symmetric_policy, mirrored_x[i : i + 1], lo, hi, alg.hf_dirs, alg.hf_eps,
                                   alg.hf_scales)
        for i in range(4)
    ]
    torch.testing.assert_close(torch.stack(per_sample), torch.stack(per_mirror), rtol=1e-4, atol=1e-7)


def test_penalty_samples_originals_and_their_mirrors():
    alg = _alg(SMOOTH_PPO.SmoothPPO, _policy(seed=7), _Env(), hf_gyro_coef=0.05, hf_samples=8)
    batch = 10
    obs = _obs(batch)
    aug, _ = SYM.compute_symmetric_states(_Env(), obs=obs)
    seen = []
    alg.hf_gyro_penalty_on = lambda rows: seen.append(rows) or rows.sum()
    alg._hf_gyro_penalty(aug, batch, 2)
    rows = seen[0]
    assert rows.shape[0] == 8
    torch.testing.assert_close(_mirror_obs(rows[:4]), rows[4:])


def test_hf_gradient_reduces_hf_gain():
    torch.manual_seed(2)
    weight = torch.randn(len(JOINT_ORDER), OBSERVATION_SIZE) * 0.05
    alg = _linear_alg(weight)
    x = torch.randn(64, OBSERVATION_SIZE)
    before = alg.hf_gyro_penalty_on(x).item()
    opt = torch.optim.SGD(alg.policy.actor.parameters(), lr=0.5)
    for _ in range(50):
        opt.zero_grad()
        alg.hf_gyro_penalty_on(x).backward()
        opt.step()
    assert alg.hf_gyro_penalty_on(x).item() < 0.05 * before
    torch.testing.assert_close(alg.policy.actor[0].weight[:, :120].detach(), weight[:, :120], rtol=0.0, atol=1e-5)
    torch.testing.assert_close(alg.policy.actor[0].weight[:, 135:].detach(), weight[:, 135:], rtol=0.0, atol=1e-5)


def _cfg_fields(path, class_name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]


def test_cfg_fields_are_accepted_by_the_algorithm():
    fields = _cfg_fields(AGENT_CFG, "SmoothPpoAlgorithmCfg")
    assert fields[0] == "class_name"
    params = inspect.signature(SMOOTH_PPO.SmoothPPO.__init__).parameters
    for name in fields[1:]:
        assert name in params, name
    isaac_rl_cfg = Path("/home/polygon/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py")
    if not isaac_rl_cfg.exists():
        pytest.skip("Isaac Lab source not found")
    stock = inspect.signature(SMOOTH_PPO.PPO.__init__).parameters
    for name in _cfg_fields(isaac_rl_cfg, "RslRlPpoAlgorithmCfg"):
        assert name == "class_name" or name in stock, name


def test_ver2_agent_cfg_uses_smooth_ppo_with_the_w92_default():
    tree = ast.parse(AGENT_CFG.read_text())
    runner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PPORunnerCfg")
    algorithm = next(n for n in runner.body if isinstance(n, ast.Assign) and n.targets[0].id == "algorithm")
    assert algorithm.value.func.id == "SmoothPpoAlgorithmCfg"
    kwargs = {k.arg: ast.literal_eval(k.value) for k in algorithm.value.keywords if isinstance(k.value, ast.Constant)}
    assert kwargs["hf_gyro_coef"] == 0.0
    assert kwargs["hf_closed_loop"] is True
    assert kwargs["lcp_coef"] == 0.0


def _closed_alg(policy, **extra):
    return _alg(SMOOTH_PPO.SmoothPPO, policy, _Env(), hf_gyro_coef=0.05, hf_closed_loop=True, **extra)


def _analytic_closed(weight, scales, damping=0.0):
    total = 0.0
    eye = torch.eye(weight.shape[0], dtype=torch.complex128)
    for freq in FREQS:
        b = torch.zeros(weight.shape[0], 3, dtype=torch.complex128)
        a = torch.zeros(weight.shape[0], weight.shape[0], dtype=torch.complex128)
        for k in range(HISTORY):
            omega = 2.0 * math.pi * freq * DT
            b = b + weight[:, 120 + 3 * k : 123 + 3 * k].to(torch.complex128) * complex(
                math.cos(omega * (HISTORY - 1 - k)), -math.sin(omega * (HISTORY - 1 - k)))
            a = a + weight[:, 175 + 12 * k : 187 + 12 * k].to(torch.complex128) * complex(
                math.cos(omega * (HISTORY - k)), -math.sin(omega * (HISTORY - k)))
        m = eye - a
        if damping > 0.0:
            t = torch.linalg.solve(m.mH @ m + damping**2 * eye, m.mH @ b)
        else:
            t = torch.linalg.inv(m) @ b
        total += (scales.double()[:, None] * t).abs().square().sum().item()
    return total


def _linear_policy(weight, normalize=False):
    policy = _policy()
    policy.actor = torch.nn.Sequential(torch.nn.Linear(OBSERVATION_SIZE, len(JOINT_ORDER)))
    with torch.no_grad():
        policy.actor[0].weight.copy_(weight)
        policy.actor[0].bias.normal_()
        if normalize:
            policy.actor_obs_normalizer._mean.normal_()
            policy.actor_obs_normalizer._std.uniform_(0.2, 2.0)
    return policy


def test_closed_loop_resolves_the_last_action_history():
    alg = _closed_alg(_policy())
    assert alg.hf_slice == (120, 135)
    assert alg.hf_action_slice == (175, 235)
    assert alg.hf_history == HISTORY
    assert SMOOTH_PPO.hf_phasors((25.0,), DT, HISTORY, 1)[0].real.tolist() == pytest.approx([-1, 1, -1, 1, -1])
    assert SMOOTH_PPO.hf_phasors((25.0,), DT, HISTORY, 0)[0].real.tolist() == pytest.approx([1, -1, 1, -1, 1])


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("damping", [0.0, 0.05])
def test_closed_loop_of_a_linear_policy_matches_the_analytic_transfer(normalize, damping):
    torch.manual_seed(6)
    weight = torch.randn(len(JOINT_ORDER), OBSERVATION_SIZE) * 0.1
    weight[:, 175:235] *= 0.5
    alg = _closed_alg(_linear_policy(weight, normalize), hf_damping=damping)
    got = alg.hf_gyro_penalty_on(torch.randn(8, OBSERVATION_SIZE)).item()
    norm = alg.policy.actor_obs_normalizer
    raw = weight / (norm._std + norm.eps)
    want = _analytic_closed(raw.double(), alg.hf_scales, damping)
    assert got == pytest.approx(want, rel=1e-4)
    assert got != pytest.approx(_analytic(raw, 120, alg.hf_scales), rel=1e-2)


def test_closed_loop_equals_the_open_form_without_action_feedback():
    torch.manual_seed(8)
    weight = torch.randn(len(JOINT_ORDER), OBSERVATION_SIZE) * 0.1
    weight[:, 175:235] = 0.0
    x = torch.randn(16, OBSERVATION_SIZE)
    closed = _closed_alg(_linear_policy(weight), hf_damping=0.0).hf_gyro_penalty_on(x).item()
    opened = _linear_alg(weight).hf_gyro_penalty_on(x).item()
    assert closed == pytest.approx(opened, rel=1e-4)

    alg = _closed_alg(_policy(seed=4, hidden=(64, 64)), hf_damping=0.0)
    with torch.no_grad():
        alg.policy.actor[0].weight[:, 175:235] = 0.0
    open_alg = _alg(SMOOTH_PPO.SmoothPPO, alg.policy, _Env(), hf_gyro_coef=0.05)
    x = torch.randn(16, OBSERVATION_SIZE)
    assert alg.hf_gyro_penalty_on(x).item() == pytest.approx(open_alg.hf_gyro_penalty_on(x).item(), rel=2e-3)


def test_closed_loop_gradient_reaches_the_actor_parameters():
    alg = _closed_alg(_policy(seed=9, hidden=(32, 32)))
    x = torch.randn(8, OBSERVATION_SIZE)
    alg.policy.zero_grad()
    alg.hf_gyro_penalty_on(x).backward()
    last_bias = f"actor.{len(alg.policy.actor) - 1}.bias"
    grads = [(n, p.grad) for n, p in alg.policy.named_parameters() if n.startswith("actor.") and n != last_bias]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for _, g in grads)
    assert dict(alg.policy.named_parameters())[last_bias].grad is None
    assert all(p.grad is None or p.grad.abs().sum() == 0 for n, p in alg.policy.named_parameters()
               if n.startswith("critic."))

    param = alg.policy.actor[2].weight
    direction = torch.randn_like(param)
    analytic = (param.grad * direction).sum().item()
    step = 1e-3
    with torch.no_grad():
        param += step * direction
        plus = alg.hf_gyro_penalty_on(x).item()
        param -= 2 * step * direction
        minus = alg.hf_gyro_penalty_on(x).item()
        param += step * direction
    assert analytic == pytest.approx((plus - minus) / (2 * step), rel=2e-2)


def test_closed_loop_penalty_is_mirror_invariant():
    alg = _closed_alg(_policy(seed=7, hidden=(64, 64)), hf_damping=0.0)
    layout = SYM._layout(_Env())
    blocks, history = layout["groups"]["policy"]
    perm = layout["perm"]

    def mirror_obs(obs):
        return SYM._mirror_group(obs, blocks, history, perm)

    x = torch.randn(16, OBSERVATION_SIZE)
    mirrored_x = mirror_obs(x)
    torch.testing.assert_close(mirrored_x, _mirror_obs(x))

    def mirrored_policy(obs):
        return -alg.actor_mean(mirror_obs(obs))[..., perm]

    def penalty(fn, obs):
        return SMOOTH_PPO.hf_closed_loop_penalty(fn, obs, alg.hf_slice, alg.hf_action_slice, alg.hf_history,
                                                 alg.hf_freqs, alg.step_dt, alg.hf_scales, alg.hf_damping)

    torch.testing.assert_close(penalty(alg.actor_mean, mirrored_x), penalty(mirrored_policy, x), rtol=1e-4, atol=1e-7)

    def symmetric_policy(obs):
        return 0.5 * (alg.actor_mean(obs) + mirrored_policy(obs))

    per_sample = torch.stack([penalty(symmetric_policy, x[i : i + 1]) for i in range(4)])
    per_mirror = torch.stack([penalty(symmetric_policy, mirrored_x[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(per_sample, per_mirror, rtol=1e-4, atol=1e-7)


def test_closed_loop_damping_bounds_a_singular_loop():
    weight = torch.zeros(len(JOINT_ORDER), OBSERVATION_SIZE)
    weight[:, 120] = 0.1
    for k in range(HISTORY):
        weight[:, 175 + 12 * k : 187 + 12 * k] = torch.eye(len(JOINT_ORDER)) * ((-1.0) ** (HISTORY - k)) / HISTORY
    alg = _closed_alg(_linear_policy(weight), hf_damping=0.05)
    norm = alg.policy.actor_obs_normalizer
    raw = weight / (norm._std + norm.eps)
    got = alg.hf_gyro_penalty_on(torch.randn(4, OBSERVATION_SIZE)).item()
    assert math.isfinite(got)
    assert got == pytest.approx(_analytic_closed(raw.double(), alg.hf_scales, 0.05), rel=1e-3)
    b_gain = _analytic(raw, 120, alg.hf_scales)
    assert got <= b_gain / (4 * 0.05**2) + 1e-9


def test_closed_loop_update_logs_and_off_switch_is_bit_identical():
    base = _policy(seed=3)
    default = _alg(SMOOTH_PPO.SmoothPPO, copy.deepcopy(base), _Env(), hf_gyro_coef=0.05, hf_samples=8)
    explicit = _alg(SMOOTH_PPO.SmoothPPO, copy.deepcopy(base), _Env(), hf_gyro_coef=0.05, hf_samples=8,
                    hf_closed_loop=False, hf_damping=0.3, last_action_term="missing")
    closed = _alg(SMOOTH_PPO.SmoothPPO, copy.deepcopy(base), _Env(), hf_gyro_coef=0.05, hf_samples=8,
                  hf_closed_loop=True)
    losses_default = _rollout_and_update(default, seed=11)
    assert _rollout_and_update(explicit, seed=11) == losses_default
    for (name, a), (_, b) in zip(default.policy.state_dict().items(), explicit.policy.state_dict().items()):
        assert torch.equal(a, b), name
    losses_closed = _rollout_and_update(closed, seed=11)
    assert losses_closed["hf_gyro"] > 0 and math.isfinite(losses_closed["hf_gyro"])
    assert losses_closed["hf_gyro"] != losses_default["hf_gyro"]


def test_closed_loop_rejects_a_bad_action_history():
    env = _Env()
    env.observation_manager.active_terms["policy"][-1] = "last_act"
    with pytest.raises(RuntimeError, match="not in the actor"):
        _alg(SMOOTH_PPO.SmoothPPO, _policy(), env, hf_gyro_coef=0.05, hf_closed_loop=True)
    with pytest.raises(RuntimeError, match="values per frame"):
        _alg(SMOOTH_PPO.SmoothPPO, _policy(), _Env(), hf_gyro_coef=0.05, hf_closed_loop=True,
             last_action_term="projected_gravity")
    with pytest.raises(ValueError, match="non-negative"):
        _alg(SMOOTH_PPO.SmoothPPO, _policy(), _Env(), hf_gyro_coef=0.05, hf_closed_loop=True, hf_damping=-0.1)
