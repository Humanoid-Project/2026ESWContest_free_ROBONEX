import ast
import math
import types
import unittest
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
REWARDS = ROOT / "mdp/rewards.py"
CFG = ROOT / "robonex_walking_v2_env_cfg.py"
ISAAC_DICT = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab/utils/dict.py")
DT = 0.02
GRACE_STEPS = math.ceil(1.5 / DT)


class Base:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load():
    tree = ast.parse(REWARDS.read_text())
    nodes = [n for n in tree.body if getattr(n, "name", None) in {"_saturate", "feet_touchdown_descent_l2"}]
    ns = {"torch": torch, "math": math, "ManagerTermBase": Base, "ManagerBasedRLEnv": object, "SceneEntityCfg": object,
          "Articulation": object, "ContactSensor": object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "rewards", "exec"), ns)
    return ns["feet_touchdown_descent_l2"], ns["_saturate"]


class Sensor:
    def __init__(self, n):
        self.data = types.SimpleNamespace(current_contact_time=torch.zeros(n, 2))

    def compute_first_contact(self, dt, abs_tol=1.0e-8):
        t = self.data.current_contact_time
        return (t > 0.0) & (t < dt + abs_tol)


class Env:
    def __init__(self, n=1):
        self.step_dt = DT
        self.common_step_counter = 0
        self.vel = torch.zeros(n, 2, 3)
        self.robot = types.SimpleNamespace(data=types.SimpleNamespace(body_lin_vel_w=self.vel))
        self.sensor = Sensor(n)
        self.scene = types.SimpleNamespace(sensors={"contact_forces": self.sensor})
        self.scene_items = {"robot": self.robot}

    def step(self, vz, contact_time):
        self.common_step_counter += 1
        self.vel[:, :, 2] = torch.as_tensor(vz)
        self.sensor.data.current_contact_time[:] = torch.as_tensor(contact_time)


class Commands:
    def __init__(self, n=1):
        self.command = torch.zeros(n, 3)

    def get_command(self, name):
        assert name == "base_velocity"
        return self.command


class SceneDict(types.SimpleNamespace):
    def __getitem__(self, key):
        return self.items[key]


def make():
    env = Env()
    env.scene = SceneDict(sensors={"contact_forces": env.sensor}, items={"robot": env.robot})
    return env


def make_gated(command=(0.3, 0.0, 0.0), episode_steps=1000):
    env = make()
    env.command_manager = Commands()
    env.command_manager.command[:] = torch.as_tensor(command)
    env.episode_length_buf = torch.full((1,), episode_steps, dtype=torch.long)
    env.common_step_counter = 500
    return env


def mark(env, age):
    env._sway_last_push_step = torch.full((1,), env.common_step_counter + 3 - age, dtype=torch.long)


def touchdown(term, env, call):
    env.step([[-0.4, 0.0]], [[0.0, 0.0]])
    call(term, env)
    env.step([[-0.6, 0.0]], [[0.0, 0.0]])
    call(term, env)
    env.step([[-0.1, 0.0]], [[0.005, 0.0]])
    return call(term, env).item()


def expected_touchdown():
    speed = 0.6 + (0.6 - 0.4) * (DT - 0.005) / DT
    return 1.0 - math.exp(-((speed - 0.25) ** 2) / 0.16)


SENSOR = types.SimpleNamespace(name="contact_forces", body_ids=[0, 1])
ASSET = types.SimpleNamespace(name="robot", body_ids=[0, 1])


class TouchdownDescentTest(unittest.TestCase):
    def setUp(self):
        self.cls, self.sat = load()

    def call(self, term, env):
        return term(env, sensor_cfg=SENSOR, asset_cfg=ASSET, deadband=0.25, scale=0.16)

    def test_extrapolates_to_the_contact_instant_and_penalises_only_first_contact(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.4, 0.0]], [[0.0, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)
        env.step([[-0.6, 0.0]], [[0.0, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)
        env.step([[-0.1, 0.0]], [[0.005, 0.0]])
        lag = DT - 0.005
        speed = 0.6 + (0.6 - 0.4) * lag / DT
        expected = 1.0 - math.exp(-((speed - 0.25) ** 2) / 0.16)
        self.assertAlmostEqual(self.call(term, env).item(), expected, places=5)
        env.step([[0.0, 0.0]], [[0.025, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)

    def test_slow_landing_inside_the_deadband_is_free(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.1, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[-0.2, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[0.0, 0.0]], [[0.015, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)

    def test_first_step_after_reset_is_masked_and_history_reseeded(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.9, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        term.reset([0])
        env.step([[-0.9, 0.0]], [[0.01, 0.0]])
        self.assertEqual(self.call(term, env).item(), 0.0)

    def test_value_is_computed_once_per_step(self):
        env = make()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.4, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[-0.6, 0.0]], [[0.0, 0.0]])
        self.call(term, env)
        env.step([[-0.1, 0.0]], [[0.005, 0.0]])
        first = self.call(term, env).item()
        self.assertEqual(self.call(term, env).item(), first)
        self.assertGreater(first, 0.0)

    def test_cfg_term_is_inert_with_a_float_weight(self):
        text = CFG.read_text()
        block = text[text.index("feet_touchdown_descent = RewTerm("):]
        block = block[: block.index("    )\n") + 6]
        self.assertIn("weight=0.0,", block)
        self.assertIn('"deadband": 0.25', block)
        self.assertIn('"scale": 0.16', block)


class TouchdownDescentGateTest(unittest.TestCase):
    def setUp(self):
        self.cls, _ = load()

    def gated(self, command_deadband=0.0, push_grace_s=0.0):
        def call(term, env):
            return term(env, sensor_cfg=SENSOR, asset_cfg=ASSET, deadband=0.25, scale=0.16,
                        command_name="base_velocity", command_deadband=command_deadband, push_grace_s=push_grace_s)
        return call

    def plain(self, term, env):
        return term(env, sensor_cfg=SENSOR, asset_cfg=ASSET, deadband=0.25, scale=0.16)

    def run_touchdown(self, env, call):
        return touchdown(self.cls(types.SimpleNamespace(params={}), env), env, call)

    def test_default_path_is_unchanged(self):
        reference = self.run_touchdown(make(), self.plain)
        self.assertAlmostEqual(reference, expected_touchdown(), places=6)
        self.assertEqual(self.run_touchdown(make(), self.gated()), reference)
        env = make()
        env._sway_last_push_step = torch.full((1,), 2, dtype=torch.long)
        self.assertEqual(self.run_touchdown(env, self.gated()), reference)
        env = make_gated(command=(0.0, 0.0, 0.0))
        mark(env, 1)
        self.assertEqual(self.run_touchdown(env, self.gated()), reference)

    def test_command_gate_zeroes_at_or_below_the_deadband(self):
        call = self.gated(command_deadband=0.05)
        for command in ((0.0, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, 0.0, 0.049)):
            self.assertEqual(self.run_touchdown(make_gated(command), call), 0.0, command)
        for command in ((0.06, 0.0, 0.0), (0.0, -0.2, 0.0), (0.0, 0.0, 0.3)):
            self.assertAlmostEqual(self.run_touchdown(make_gated(command), call), expected_touchdown(), places=6)

    def test_push_grace_zeroes_for_its_policy_steps(self):
        call = self.gated(push_grace_s=1.5)
        self.assertEqual(GRACE_STEPS, 75)
        for age, graced in ((1, True), (GRACE_STEPS, True), (GRACE_STEPS + 1, False), (400, False)):
            env = make_gated()
            mark(env, age)
            value = self.run_touchdown(env, call)
            if graced:
                self.assertEqual(value, 0.0, age)
            else:
                self.assertAlmostEqual(value, expected_touchdown(), places=6)

    def test_push_grace_ignores_an_unmarked_env_and_a_missing_marker(self):
        call = self.gated(push_grace_s=1.5)
        env = make_gated()
        env._sway_last_push_step = torch.full((1,), -1, dtype=torch.long)
        self.assertAlmostEqual(self.run_touchdown(env, call), expected_touchdown(), places=6)
        self.assertAlmostEqual(self.run_touchdown(make_gated(), call), expected_touchdown(), places=6)

    def test_push_grace_does_not_carry_over_a_reset(self):
        call = self.gated(push_grace_s=1.5)
        env = make_gated(episode_steps=10)
        mark(env, 11)
        self.assertAlmostEqual(self.run_touchdown(env, call), expected_touchdown(), places=6)
        env = make_gated(episode_steps=10)
        mark(env, 10)
        self.assertEqual(self.run_touchdown(env, call), 0.0)

    def test_reset_still_masks_the_first_contact_with_gates_on(self):
        call = self.gated(command_deadband=0.05, push_grace_s=1.5)
        env = make_gated()
        term = self.cls(types.SimpleNamespace(params={}), env)
        env.step([[-0.9, 0.0]], [[0.0, 0.0]])
        call(term, env)
        term.reset([0])
        env.step([[-0.9, 0.0]], [[0.01, 0.0]])
        self.assertEqual(call(term, env).item(), 0.0)

    def test_a_gated_touchdown_keeps_the_velocity_history(self):
        def sequence(call, env):
            term = self.cls(types.SimpleNamespace(params={}), env)
            out = []
            for vz, t in (([-0.4, -0.3], [0.0, 0.0]), ([-0.6, -0.5], [0.0, 0.0]), ([-0.1, -0.7], [0.005, 0.0]),
                          ([0.0, -0.1], [0.025, 0.004])):
                env.step([vz], [t])
                out.append(call(term, env).item())
            return out

        env = make_gated()
        mark(env, 3)
        graced = sequence(self.gated(push_grace_s=0.05), env)
        plain = sequence(self.plain, make())
        self.assertEqual(graced[2], 0.0)
        self.assertGreater(plain[2], 0.0)
        self.assertEqual(graced[3], plain[3])
        self.assertGreater(graced[3], 0.0)

    def test_invalid_gate_parameters_raise(self):
        for kwargs in ({"command_deadband": -0.01}, {"command_deadband": float("nan")}, {"push_grace_s": -1.0},
                       {"push_grace_s": float("inf")}):
            env = make_gated()
            term = self.cls(types.SimpleNamespace(params={}), env)
            env.step([[0.0, 0.0]], [[0.0, 0.0]])
            with self.assertRaises(ValueError):
                self.gated(**kwargs)(term, env)

    def test_cfg_gates_are_inert_floats(self):
        text = CFG.read_text()
        block = text[text.index("feet_touchdown_descent = RewTerm("):]
        block = block[: block.index("    )\n") + 6]
        self.assertIn('"command_name": "base_velocity"', block)
        self.assertIn('"command_deadband": 0.0', block)
        self.assertIn('"push_grace_s": 0.0', block)


OVERRIDES = (
    "env.rewards.feet_touchdown_descent.weight=-1.5",
    "env.rewards.feet_touchdown_descent.weight=-2.0",
    "env.rewards.feet_touchdown_descent.params.command_deadband=0.05",
    "env.rewards.feet_touchdown_descent.params.push_grace_s=1.5",
)


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
class TouchdownDescentOverrideTest(unittest.TestCase):
    def update(self):
        node = next(n for n in ast.parse(ISAAC_DICT.read_text()).body if getattr(n, "name", None) == "update_class_from_dict")
        ns = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": object, "string_to_callable": lambda s: s}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "dict", "exec"), ns)
        return ns["update_class_from_dict"]

    def term(self):
        tree = ast.parse(CFG.read_text())
        rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        node = next(n for n in rewards.body if isinstance(n, ast.Assign) and n.targets[0].id == "feet_touchdown_descent")
        kw = {k.arg: k.value for k in node.value.keywords}
        params = {}
        for key, value in zip(kw["params"].keys, kw["params"].values):
            try:
                params[ast.literal_eval(key)] = ast.literal_eval(value)
            except ValueError:
                params[ast.literal_eval(key)] = types.SimpleNamespace()
        term = types.SimpleNamespace(weight=ast.literal_eval(kw["weight"]), params=params)
        return types.SimpleNamespace(rewards=types.SimpleNamespace(feet_touchdown_descent=term))

    def parse(self, text):
        from hydra.core.override_parser.overrides_parser import OverridesParser

        override = OverridesParser.create().parse_overrides([text])[0]
        data = override.value()
        for key in reversed(override.key_or_group.split(".")[1:]):
            data = {key: data}
        return data

    def read(self, env, text):
        node = env
        for key in text.split("=")[0].split(".")[1:]:
            node = node[key] if isinstance(node, dict) else getattr(node, key)
        return node

    def test_launch_overrides_are_accepted_and_restore_from_the_saved_yaml(self):
        update = self.update()
        for text in OVERRIDES:
            env = self.term()
            update(env, self.parse(text))
            saved = self.read(env, text)
            self.assertIsInstance(saved, float, text)
            self.assertEqual(saved, float(text.split("=")[1]), text)
            restored = self.term()
            data = saved
            for key in reversed(text.split("=")[0].split(".")[1:]):
                data = {key: data}
            update(restored, data)
            self.assertEqual(self.read(restored, text), saved, text)

    def test_an_int_onto_a_float_gate_is_rejected(self):
        update = self.update()
        for text in ("env.rewards.feet_touchdown_descent.params.push_grace_s=1",
                     "env.rewards.feet_touchdown_descent.params.command_deadband=0"):
            with self.assertRaises(ValueError, msg=text):
                update(self.term(), self.parse(text))


if __name__ == "__main__":
    unittest.main()
