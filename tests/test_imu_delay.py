import ast
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
OBS = ROOT / "mdp/observations.py"
ENV_CFG = ROOT / "robonex_walking_v2_env_cfg.py"
NAMES = {"DelayedImu", "DelayedImuCfg", "delayed_imu_ang_vel", "delayed_imu_projected_gravity"}


class FakeImu:
    def __init__(self, cfg, num_envs, source):
        self.cfg = cfg
        self._num_envs = num_envs
        self._device = "cpu"
        self._source = source
        self._is_outdated = torch.ones(num_envs, dtype=torch.bool)
        self._data = types.SimpleNamespace(ang_vel_b=torch.zeros(num_envs, 3), projected_gravity_b=torch.zeros(num_envs, 3))
        self._data.projected_gravity_b[:, 2] = -1.0

    def _initialize_impl(self):
        pass

    def update(self, dt, force_recompute=False):
        self._is_outdated[:] = True
        if force_recompute or self.cfg.history_length > 0:
            self._update_outdated_buffers()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._is_outdated[ids] = True
        self._data.ang_vel_b[ids] = 0.0
        self._data.projected_gravity_b[ids] = torch.tensor([0.0, 0.0, -1.0])

    def _update_outdated_buffers(self):
        ids = self._is_outdated.nonzero().squeeze(-1)
        if len(ids) > 0:
            self._update_buffers_impl(ids)
            self._is_outdated[ids] = False

    def _update_buffers_impl(self, env_ids):
        ang_vel, gravity = self._source()
        self._data.ang_vel_b[env_ids] = ang_vel[env_ids]
        self._data.projected_gravity_b[env_ids] = gravity[env_ids]


class FakeImuCfg:
    pass


def load():
    nodes = [n for n in ast.parse(OBS.read_text()).body if getattr(n, "name", None) in NAMES]
    ns = {"torch": torch, "Imu": FakeImu, "ImuCfg": FakeImuCfg, "configclass": lambda cls: cls, "ManagerBasedRLEnv": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "observations", "exec"), ns)
    return ns


class Signal:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.t = 0
        self.history = []

    def set(self, t):
        self.t = t
        self.history.append(self.current())

    def current(self):
        env = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(1)
        ang_vel = torch.stack([torch.full((self.num_envs,), float(self.t))] * 3, dim=1) + env * 100.0 + torch.tensor([[0.0, 0.25, 0.5]])
        gravity = -ang_vel * 0.001 - 1.0
        return ang_vel, gravity

    def __call__(self):
        return self.current()

    def at(self, index):
        return self.history[index]


def make(ns, num_envs, min_delay, max_delay, signal):
    cfg = types.SimpleNamespace(min_delay_steps=min_delay, max_delay_steps=max_delay, history_length=1)
    imu = ns["DelayedImu"](cfg, num_envs, signal)
    imu._initialize_impl()
    return imu


class ImuDelayTest(unittest.TestCase):
    def setUp(self):
        self.ns = load()

    def test_delay_of_k_substeps_is_exact_and_per_env(self):
        signal = Signal(4)
        imu = make(self.ns, 4, 0, 6, signal)
        delays = torch.tensor([0, 1, 3, 6])
        imu._delay = delays.clone()
        for step in range(3 * 8):
            signal.set(step)
            imu.update(0.0025)
            if (step + 1) % 8 == 0:
                ang_vel, gravity = imu.delayed()
                for env, k in enumerate(delays.tolist()):
                    expected_w, expected_g = signal.at(max(step - k, 0))
                    self.assertTrue(torch.equal(ang_vel[env], expected_w[env]), (step, env, k))
                    self.assertTrue(torch.equal(gravity[env], expected_g[env]), (step, env, k))
        self.assertTrue(torch.equal(imu._delay, delays))

    def test_history_before_k_samples_is_the_first_fill(self):
        signal = Signal(2)
        imu = make(self.ns, 2, 0, 4, signal)
        imu._delay = torch.tensor([4, 2])
        signal.set(10)
        imu.update(0.0025)
        signal.set(11)
        imu.update(0.0025)
        ang_vel, _ = imu.delayed()
        self.assertTrue(torch.equal(ang_vel, signal.at(0)[0]))

    def test_reset_refills_only_the_reset_env_and_redraws_its_delay(self):
        signal = Signal(3)
        imu = make(self.ns, 3, 2, 2, signal)
        for step in range(8):
            signal.set(step)
            imu.update(0.0025)
        before, _ = imu.delayed()
        self.assertTrue(torch.equal(before, signal.at(5)[0]))
        signal.set(50)
        imu.reset(torch.tensor([1]))
        after, gravity = imu.delayed()
        self.assertTrue(torch.equal(after[[0, 2]], signal.at(5)[0][[0, 2]]))
        self.assertTrue(torch.equal(after[1], signal.current()[0][1]))
        self.assertTrue(torch.equal(gravity[1], signal.current()[1][1]))
        self.assertEqual(imu._delay.tolist(), [2, 2, 2])
        self.assertFalse(imu._fresh.any())
        signal.set(51)
        imu.update(0.0025)
        ang_vel, _ = imu.delayed()
        self.assertTrue(torch.equal(ang_vel[[0, 2]], signal.at(6)[0][[0, 2]]))
        self.assertTrue(torch.equal(ang_vel[1], signal.at(8)[0][1]))
        for step in (52, 53):
            signal.set(step)
            imu.update(0.0025)
        ang_vel, _ = imu.delayed()
        self.assertTrue(torch.equal(ang_vel[1], signal.at(-3)[0][1]))
        imu.reset()
        self.assertTrue(imu._fresh.all())

    def test_draw_respects_the_range_and_reset_redraws(self):
        signal = Signal(64)
        imu = make(self.ns, 64, 1, 6, signal)
        torch.manual_seed(0)
        draws = torch.cat([imu._draw(64) for _ in range(20)])
        self.assertEqual(sorted(set(draws.tolist())), [1, 2, 3, 4, 5, 6])
        self.assertTrue(((imu._delay >= 1) & (imu._delay <= 6)).all())
        imu._delay[:] = 0
        imu.reset(torch.arange(32))
        self.assertTrue(((imu._delay[:32] >= 1) & (imu._delay[:32] <= 6)).all())
        self.assertTrue((imu._delay[32:] == 0).all())
        fixed = make(self.ns, 8, 3, 3, Signal(8))
        self.assertEqual(fixed._delay.tolist(), [3] * 8)
        with self.assertRaises(ValueError):
            make(self.ns, 2, 4, 3, Signal(2))

    def test_zero_delay_is_bit_identical_to_the_sensor_data(self):
        torch.manual_seed(1)
        num_envs = 5
        state = {"w": torch.randn(num_envs, 3), "g": torch.randn(num_envs, 3)}
        imu = make(self.ns, num_envs, 0, 0, lambda: (state["w"], state["g"]))
        env = types.SimpleNamespace(scene={"imu": imu})
        asset_cfg = types.SimpleNamespace(name="imu")
        for step in range(40):
            state["w"] = torch.randn(num_envs, 3)
            state["g"] = torch.randn(num_envs, 3)
            imu.update(0.0025)
            if step % 8 == 7:
                expected_w, expected_g = state["w"], state["g"]
                if step == 23:
                    imu.reset(torch.tensor([0, 3]))
                    state["w"] = torch.randn(num_envs, 3)
                    state["g"] = torch.randn(num_envs, 3)
                    expected_w, expected_g = expected_w.clone(), expected_g.clone()
                    expected_w[[0, 3]] = state["w"][[0, 3]]
                    expected_g[[0, 3]] = state["g"][[0, 3]]
                self.assertTrue(torch.equal(self.ns["delayed_imu_ang_vel"](env, asset_cfg), imu._data.ang_vel_b))
                self.assertTrue(torch.equal(self.ns["delayed_imu_projected_gravity"](env, asset_cfg), imu._data.projected_gravity_b))
                self.assertTrue(torch.equal(imu._data.ang_vel_b, expected_w))
                self.assertTrue(torch.equal(imu._data.projected_gravity_b, expected_g))

    def test_cfg_wiring_is_off_by_default_and_term_names_are_unchanged(self):
        tree = ast.parse(ENV_CFG.read_text())
        consts = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
        self.assertEqual(ast.literal_eval(consts["IMU_DELAY_STEPS"]), (0, 0))
        cfg_default = next(n for n in ast.parse(OBS.read_text()).body if getattr(n, "name", None) == "DelayedImuCfg")
        defaults = {n.target.id: ast.literal_eval(n.value) for n in cfg_default.body if isinstance(n, ast.AnnAssign) and n.target.id.endswith("_steps")}
        self.assertEqual(defaults, {"min_delay_steps": 0, "max_delay_steps": 0})
        scene = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoboNexWalkingSceneCfg")
        imu = next(n for n in scene.body if isinstance(n, ast.Assign) and n.targets[0].id == "imu")
        self.assertEqual(imu.value.func.attr, "DelayedImuCfg")
        kw = {k.arg: k.value for k in imu.value.keywords}
        self.assertEqual(ast.literal_eval(kw["history_length"]), 1)
        self.assertEqual(ast.literal_eval(kw["update_period"]), 0.0)
        self.assertEqual((kw["min_delay_steps"].value.id, ast.literal_eval(kw["min_delay_steps"].slice)), ("IMU_DELAY_STEPS", 0))
        self.assertEqual((kw["max_delay_steps"].value.id, ast.literal_eval(kw["max_delay_steps"].slice)), ("IMU_DELAY_STEPS", 1))
        obs = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
        policy = next(n for n in obs.body if isinstance(n, ast.ClassDef) and n.name == "PolicyCfg")
        terms = [(n.targets[0].id, {k.arg: k.value for k in n.value.keywords}) for n in policy.body if isinstance(n, ast.Assign)]
        self.assertEqual([name for name, _ in terms],
                         ["joint_pos_rel", "joint_vel_rel", "imu_ang_vel", "projected_gravity", "velocity_commands", "gait_phase", "actions"])
        funcs = {name: kw["func"].attr for name, kw in terms}
        self.assertEqual(funcs["imu_ang_vel"], "delayed_imu_ang_vel")
        self.assertEqual(funcs["projected_gravity"], "delayed_imu_projected_gravity")
        for name in ("imu_ang_vel", "projected_gravity"):
            params = dict(terms)[name]["params"]
            asset = dict(zip([ast.literal_eval(k) for k in params.keys], params.values))["asset_cfg"]
            self.assertEqual(ast.literal_eval(asset.args[0]), "imu")


if __name__ == "__main__":
    unittest.main()
