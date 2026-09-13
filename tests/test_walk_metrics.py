import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[1] / "source" / "robonex_walking"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

MODULE = (
    SOURCE
    / "robonex_walking/tasks/manager_based/robonex_walking/mdp/walk_metrics.py"
)


def load_walk_metrics():
    import types

    namespace = types.ModuleType("walk_metrics_under_test")
    namespace.__dict__["__file__"] = str(MODULE)
    source = MODULE.read_text()
    source = source.replace("from isaaclab.assets import Articulation", "Articulation = object")
    source = source.replace("from isaaclab.sensors import ContactSensor", "ContactSensor = object")
    exec(compile(source, str(MODULE), "exec"), namespace.__dict__)
    return namespace.WalkMetrics


WalkMetrics = load_walk_metrics()
DT = 0.02
REST = 0.06545


class FakeEnv:
    def __init__(self, num_envs=1):
        self.num_envs = num_envs
        self.device = "cpu"
        self.common_step_counter = 0
        self._forces = torch.zeros(num_envs, 1, 2, 3)
        self._pos = torch.zeros(num_envs, 2, 3)
        self._vel = torch.zeros(num_envs, 2, 2)
        finder = lambda names, preserve_order=False: ([0, 1], list(names))
        sensor = SimpleNamespace(
            data=SimpleNamespace(net_forces_w_history=self._forces), find_bodies=finder
        )
        robot = SimpleNamespace(
            data=SimpleNamespace(body_pos_w=self._pos, body_lin_vel_w=self._vel),
            find_bodies=finder,
        )
        self.scene = SimpleNamespace(
            sensors={"contact_forces": sensor},
            env_origins=torch.zeros(num_envs, 3),
        )
        self.scene.__getitem__ = None
        self._robot = robot
        self.action_manager = SimpleNamespace(
            get_term=lambda name: SimpleNamespace(_joint_names=["j0", "j1"])
        )
        self.termination_manager = SimpleNamespace(dones=torch.zeros(num_envs, dtype=torch.bool))

    def __getitem__(self, key):
        return self._robot


class SceneProxy(dict):
    def __init__(self, robot, sensors, origins):
        super().__init__()
        self._robot = robot
        self.sensors = sensors
        self.env_origins = origins

    def __getitem__(self, key):
        return self._robot


def make_env(num_envs=1):
    env = FakeEnv(num_envs)
    env.scene = SceneProxy(env._robot, env.scene.sensors, env.scene.env_origins)
    return env


def drive(metrics, env, contacts, heights=None):
    """contacts: list of per-step (num_envs, 2) boolean-ish tensors."""
    for index, contact in enumerate(contacts):
        c = torch.as_tensor(contact, dtype=torch.float32).reshape(env.num_envs, 2)
        env._forces.zero_()
        env._forces[:, 0, :, 2] = c * 10.0
        if heights is not None:
            env._pos[:, :, 2] = torch.as_tensor(
                heights[index], dtype=torch.float32
            ).reshape(env.num_envs, 2) + REST
        env.common_step_counter += 1
        metrics.update(env)


class WalkMetricsTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.metrics = WalkMetrics(self.env, REST, DT)

    def test_perfect_alternation_has_no_same_foot_touchdown(self):
        # L down, R down, L down, R down with clean single->double transitions
        seq = [[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [1, 1], [1, 0], [1, 0], [1, 1], [0, 1]]
        drive(self.metrics, self.env, seq)
        log = self.metrics.take_log()
        self.assertEqual(log["Gait/same_foot_td_frac"], 0.0)

    def test_same_foot_repeats_are_counted(self):
        # only the left foot ever lifts and lands again
        seq = [[1, 1], [0, 1], [1, 1], [0, 1], [1, 1], [0, 1], [1, 1]]
        drive(self.metrics, self.env, seq)
        log = self.metrics.take_log()
        self.assertEqual(log["Gait/same_foot_td_frac"], 1.0)

    def test_step_duration_matches_synthetic_timing(self):
        # L touchdown at step 1, R touchdown 5 steps later, L touchdown 5 steps after that
        seq = [[0, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 1]]
        seq += [[0, 1], [0, 1], [0, 1], [0, 1], [1, 1]]
        drive(self.metrics, self.env, seq)
        log = self.metrics.take_log()
        self.assertAlmostEqual(log["Gait/step_duration_s"], 5 * DT, places=6)

    def test_phase_fractions(self):
        seq = [[1, 1], [1, 1], [1, 0], [0, 0]]
        drive(self.metrics, self.env, seq)
        log = self.metrics.take_log()
        self.assertAlmostEqual(log["Gait/double_stance_frac"], 0.5, places=6)
        self.assertAlmostEqual(log["Gait/single_stance_frac"], 0.25, places=6)
        self.assertAlmostEqual(log["Gait/flight_frac"], 0.25, places=6)

    def test_duty_ratio(self):
        seq = [[1, 0], [1, 0], [1, 1], [0, 1]]
        drive(self.metrics, self.env, seq)
        log = self.metrics.take_log()
        self.assertAlmostEqual(log["Gait/duty_l"], 0.75, places=6)
        self.assertAlmostEqual(log["Gait/duty_r"], 0.5, places=6)

    def test_swing_peak_height(self):
        # right foot lifts to 0.08 m above rest, then lands
        seq = [[1, 1], [1, 0], [1, 0], [1, 0], [1, 1]]
        h = [[0.0, 0.0], [0.0, 0.03], [0.0, 0.08], [0.0, 0.05], [0.0, 0.0]]
        drive(self.metrics, self.env, seq, h)
        log = self.metrics.take_log()
        self.assertAlmostEqual(log["Gait/swing_peak_m"], 0.08, places=5)

    def test_reset_clears_touchdown_history(self):
        drive(self.metrics, self.env, [[1, 0], [1, 1]])
        self.env.termination_manager.dones = torch.ones(1, dtype=torch.bool)
        drive(self.metrics, self.env, [[0, 1]])
        self.env.termination_manager.dones = torch.zeros(1, dtype=torch.bool)
        drive(self.metrics, self.env, [[1, 1]])
        log = self.metrics.take_log()
        self.assertEqual(log["Gait/same_foot_td_frac"], 0.0)

    def test_action_statistics(self):
        self.metrics.record_action(torch.tensor([[2.0, -6.0]]))
        self.metrics.record_action(torch.tensor([[4.0, -2.0]]))
        log = self.metrics.take_log()
        self.assertAlmostEqual(log["ActionMean/j0"], 3.0, places=6)
        self.assertAlmostEqual(log["ActionMean/j1"], -4.0, places=6)
        self.assertAlmostEqual(log["ActionAbsMax/j1"], 6.0, places=6)

    def test_nonfinite_actions_do_not_poison_statistics(self):
        self.metrics.record_action(torch.tensor([[float("nan"), float("inf")]]))
        log = self.metrics.take_log()
        self.assertTrue(all(v == v for v in log.values()))

    def test_accumulators_reset_after_take_log(self):
        drive(self.metrics, self.env, [[1, 1], [1, 1]])
        self.metrics.take_log()
        drive(self.metrics, self.env, [[0, 0]])
        log = self.metrics.take_log()
        self.assertAlmostEqual(log["Gait/flight_frac"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
