import ast
import types
import unittest
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
ISAAC_DICT = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab/utils/dict.py")

REWARD_OVERRIDES = (
    "env.rewards.slew_lag.weight=0.0",
    "env.rewards.stand_still.weight=-2.0",
    "env.rewards.track_lin_vel_y_cycle.weight=1.0",
    "env.rewards.com_forward_margin.weight=-0.5",
    "env.rewards.com_forward_margin.params.target=0.055",
    "env.rewards.com_forward_margin.params.standing=cap",
    "env.rewards.com_forward_margin.params.standing_cap=0.05",
    "env.rewards.com_forward_margin.params.push_grace_s=1.7",
    "env.rewards.com_forward_margin.params.standing_floor=0.045",
    "env.rewards.feet_width.params.lateral_deadband=0.44",
    "env.rewards.joint_load.weight=-0.02",
    "env.rewards.joint_load.weight=-0.04",
    "env.rewards.feet_touchdown_descent.weight=-1.0",
    "env.rewards.feet_touchdown_descent.weight=-3.0",
)


def reward_terms():
    tree = ast.parse(CFG.read_text())
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass

    def value(node):
        try:
            return eval(compile(ast.Expression(node), "cfg", "eval"), {"__builtins__": {}}, consts)
        except Exception:
            return types.SimpleNamespace()

    rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
    terms = {}
    for node in rewards.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            kw = {k.arg: k.value for k in node.value.keywords}
            params = {}
            if isinstance(kw.get("params"), ast.Dict):
                params = {ast.literal_eval(k): value(v) for k, v in zip(kw["params"].keys, kw["params"].values)}
            terms[node.targets[0].id] = types.SimpleNamespace(weight=value(kw["weight"]), params=params)
    return terms


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
class OverrideTypeTest(unittest.TestCase):
    def update(self):
        node = next(n for n in ast.parse(ISAAC_DICT.read_text()).body
                    if getattr(n, "name", None) == "update_class_from_dict")
        ns = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": object,
              "string_to_callable": lambda s: s}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "dict", "exec"), ns)
        return ns["update_class_from_dict"]

    def parse(self, text):
        from hydra.core.override_parser.overrides_parser import OverridesParser

        override = OverridesParser.create().parse_overrides([text])[0]
        data = override.value()
        for key in reversed(override.key_or_group.split(".")[1:]):
            data = {key: data}
        return data

    def env(self):
        return types.SimpleNamespace(rewards=types.SimpleNamespace(**reward_terms()))

    def read(self, env, text):
        node = env
        for key in text.split("=")[0].split(".")[1:]:
            node = node[key] if isinstance(node, dict) else getattr(node, key)
        return node

    def test_reward_overrides_accept_a_launch_override(self):
        update = self.update()
        for text in REWARD_OVERRIDES:
            env = self.env()
            data = self.parse(text)
            update(env, data)
            for key in text.split("=")[0].split(".")[1:]:
                data = data[key]
            self.assertEqual(self.read(env, text), data, text)

    def test_reward_overrides_restore_from_a_saved_env_yaml(self):
        update = self.update()
        for text in REWARD_OVERRIDES:
            env = self.env()
            update(env, self.parse(text))
            saved = self.read(env, text)
            restored = self.env()
            path = text.split("=")[0].split(".")[1:]
            data = saved
            for key in reversed(path):
                data = {key: data}
            update(restored, data)
            self.assertEqual(self.read(restored, text), saved, text)

    def test_an_int_onto_a_float_default_is_rejected(self):
        with self.assertRaises(ValueError):
            self.update()(self.env(), self.parse("env.rewards.slew_lag.weight=0"))

    def test_actuation_delay_list_override_and_restore(self):
        update = self.update()
        for data in (self.parse("env.actions.joint_pos.actuation_delay_substeps=[0,8]"),
                     {"actions": {"joint_pos": {"actuation_delay_substeps": [0, 8]}}}):
            env = types.SimpleNamespace(actions=types.SimpleNamespace(
                joint_pos=types.SimpleNamespace(actuation_delay_substeps=None)))
            update(env, data)
            self.assertEqual(list(env.actions.joint_pos.actuation_delay_substeps), [0, 8])


if __name__ == "__main__":
    unittest.main()
