import ast
import importlib.util
import sys
import tempfile
import types
import unittest
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
ISAAC_DICT = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab/utils/dict.py")

DR = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}
ZERO = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def effects():
    return load_module("deploy_effects_com_pin", EFFECTS)


def term(com_range):
    return types.SimpleNamespace(func="offset", mode="reset", params={
        "asset_cfg": types.SimpleNamespace(name="robot", body_names=["base_link"]),
        "com_range": dict(com_range)})


def env_cfg(com_term):
    return types.SimpleNamespace(events=types.SimpleNamespace(randomize_base_com=com_term))


class FakeTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


def load_event():
    node = next(n for n in ast.parse(EVENTS.read_text()).body
                if getattr(n, "name", None) == "randomize_rigid_body_com_offset")
    ns = {"torch": torch, "SceneEntityCfg": object, "Articulation": object, "ManagerTermBase": FakeTermBase}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "events", "exec"), ns)
    return ns["randomize_rigid_body_com_offset"]


class FakeView:
    def __init__(self, coms):
        self.coms = coms.clone()
        self.sets = 0

    def get_coms(self):
        return self.coms.clone()

    def set_coms(self, coms, env_ids):
        self.sets += 1
        self.coms[env_ids] = coms[env_ids]


class PinTest(unittest.TestCase):
    def test_default_pins_a_trained_range_to_nominal(self):
        cfg = env_cfg(term(DR))
        record = effects().apply_base_com_pin(cfg)
        com_term = cfg.events.randomize_base_com
        self.assertEqual(com_term.params["com_range"], ZERO)
        self.assertEqual((com_term.func, com_term.mode), ("offset", "reset"))
        self.assertEqual(com_term.params["asset_cfg"].body_names, ["base_link"])
        self.assertEqual(record, {"source": "default_nominal", "pin_m": [0.0, 0.0, 0.0], "event_present": True,
                                  "range_m": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}, "nominal": True})

    def test_explicit_pin_is_a_zero_width_range_at_the_offset(self):
        cfg = env_cfg(term(DR))
        record = effects().apply_base_com_pin(cfg, [0.02, -0.01, 0.0])
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"],
                         {"x": (0.02, 0.02), "y": (-0.01, -0.01), "z": (0.0, 0.0)})
        self.assertEqual(record["source"], "pin")
        self.assertEqual(record["pin_m"], [0.02, -0.01, 0.0])
        self.assertEqual(record["range_m"], {"x": [0.02, 0.02], "y": [-0.01, -0.01], "z": [0.0, 0.0]})
        self.assertFalse(record["nominal"])

    def test_keep_training_leaves_the_range_and_records_it(self):
        cfg = env_cfg(term(DR))
        record = effects().apply_base_com_pin(cfg, keep_training=True)
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"], DR)
        self.assertEqual(record, {"source": "training", "pin_m": None, "event_present": True,
                                  "range_m": {"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [0.0, 0.0]},
                                  "nominal": False})

    def test_pin_and_keep_training_exclude_each_other(self):
        with self.assertRaises(ValueError):
            effects().apply_base_com_pin(env_cfg(term(DR)), [0.0, 0.0, 0.0], keep_training=True)

    def test_bad_offsets_are_refused(self):
        for offset in ([0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]):
            with self.assertRaises(ValueError):
                effects().apply_base_com_pin(env_cfg(term(DR)), offset)

    def test_a_run_without_the_event_stays_without_it_at_nominal(self):
        for offset in (None, [0.0, 0.0, 0.0]):
            cfg = env_cfg(None)
            record = effects().apply_base_com_pin(cfg, offset, template=term(ZERO))
            self.assertIsNone(cfg.events.randomize_base_com)
            self.assertFalse(record["event_present"])
            self.assertIsNone(record["range_m"])
            self.assertTrue(record["nominal"])
        cfg = types.SimpleNamespace(events=types.SimpleNamespace())
        self.assertTrue(effects().apply_base_com_pin(cfg)["nominal"])
        self.assertFalse(hasattr(cfg.events, "randomize_base_com"))

    def test_a_non_zero_pin_on_a_run_without_the_event_installs_a_copy_of_the_template(self):
        template = term(ZERO)
        cfg = env_cfg(None)
        record = effects().apply_base_com_pin(cfg, [0.02, 0.0, 0.0], template=template)
        installed = cfg.events.randomize_base_com
        self.assertIsNot(installed, template)
        self.assertEqual(template.params["com_range"], ZERO)
        self.assertEqual(installed.params["com_range"], {"x": (0.02, 0.02), "y": (0.0, 0.0), "z": (0.0, 0.0)})
        self.assertEqual((installed.func, installed.mode), ("offset", "reset"))
        self.assertTrue(record["event_present"])
        with self.assertRaises(KeyError):
            effects().apply_base_com_pin(env_cfg(None), [0.02, 0.0, 0.0], template=None)


class EventUnderPinTest(unittest.TestCase):
    def setup_env(self, num_envs=6, num_bodies=3):
        torch.manual_seed(7)
        coms = torch.zeros(num_envs, num_bodies, 7)
        coms[..., :3] = 0.01 * torch.randn(num_envs, num_bodies, 3)
        coms[..., 6] = 1.0
        asset = types.SimpleNamespace(root_physx_view=FakeView(coms), num_bodies=num_bodies)
        scene = type("Scene", (dict,), {"num_envs": num_envs})(robot=asset)
        return types.SimpleNamespace(scene=scene), asset, coms

    def test_pinned_ranges_give_the_same_com_on_every_env_and_reset(self):
        event_cls = load_event()
        env, asset, nominal = self.setup_env()
        cfg = env_cfg(term(DR))
        effects().apply_base_com_pin(cfg, [0.02, -0.01, 0.0])
        params = cfg.events.randomize_base_com.params
        params["asset_cfg"].body_ids = [0]
        event = event_cls(cfg.events.randomize_base_com, env)
        expected = nominal.clone()
        expected[:, 0, :3] += torch.tensor([0.02, -0.01, 0.0])
        for _ in range(3):
            state = torch.get_rng_state()
            event(env, None, **params)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            torch.testing.assert_close(asset.root_physx_view.coms, expected)

    def test_default_nominal_pin_touches_nothing(self):
        event_cls = load_event()
        env, asset, nominal = self.setup_env()
        cfg = env_cfg(term(DR))
        effects().apply_base_com_pin(cfg)
        params = cfg.events.randomize_base_com.params
        params["asset_cfg"].body_ids = [0]
        event_cls(cfg.events.randomize_base_com, env)(env, torch.arange(6), **params)
        self.assertEqual(asset.root_physx_view.sets, 0)
        self.assertTrue(torch.equal(asset.root_physx_view.coms, nominal))


SAVED_DR = """events:
  randomize_base_com:
    func: robonex_walking.tasks.manager_based.robonex_walking.mdp.events:randomize_rigid_body_com_offset
    mode: reset
    params:
      com_range:
        x: !!python/tuple
        - -0.02
        - 0.02
        y: !!python/tuple
        - -0.02
        - 0.02
        z: !!python/tuple
        - 0.0
        - 0.0
seed: 43
"""

SAVED_OLD = """events:
  randomize_mass:
    mode: reset
seed: 43
"""


class RestoreThenPinTest(unittest.TestCase):
    def setUp(self):
        node = next(n for n in ast.parse(ISAAC_DICT.read_text()).body
                    if getattr(n, "name", None) == "update_class_from_dict")
        ns = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": object,
              "string_to_callable": lambda s: s}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "dict", "exec"), ns)
        fake = types.ModuleType("isaaclab.utils.dict")
        fake.update_class_from_dict = ns["update_class_from_dict"]
        self.saved_modules = {k: sys.modules.get(k) for k in ("isaaclab", "isaaclab.utils", "isaaclab.utils.dict")}
        sys.modules["isaaclab"] = types.ModuleType("isaaclab")
        sys.modules["isaaclab.utils"] = types.ModuleType("isaaclab.utils")
        sys.modules["isaaclab.utils.dict"] = fake
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for key, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = module
        self.tmp.cleanup()

    def restore(self, text):
        run = Path(self.tmp.name)
        (run / "params").mkdir()
        (run / "params" / "env.yaml").write_text(text)
        mass = types.SimpleNamespace(mode="reset", params={})
        cfg = types.SimpleNamespace(
            events=types.SimpleNamespace(randomize_mass=mass, randomize_base_com=term(ZERO)),
            observations=types.SimpleNamespace(),
            scene=types.SimpleNamespace(robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=None))))
        module = effects()
        template = term(ZERO)
        module.apply_training_env_cfg(cfg, str(run / "model_999.pt"))
        return module, cfg, template

    def test_a_com_dr_run_restores_its_range_and_the_default_pins_it_to_nominal(self):
        module, cfg, template = self.restore(SAVED_DR)
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"], DR)
        record = module.apply_base_com_pin(cfg, None, template)
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"], ZERO)
        self.assertEqual(record["source"], "default_nominal")
        self.assertTrue(record["nominal"])

    def test_a_com_dr_run_keeps_its_range_when_asked(self):
        module, cfg, template = self.restore(SAVED_DR)
        record = module.apply_base_com_pin(cfg, None, template, keep_training=True)
        self.assertEqual(record["range_m"], {"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [0.0, 0.0]})
        self.assertFalse(record["nominal"])

    def test_an_older_run_without_the_event_restores_and_pins(self):
        module, cfg, template = self.restore(SAVED_OLD)
        self.assertIsNone(cfg.events.randomize_base_com)
        record = module.apply_base_com_pin(cfg, None, template)
        self.assertEqual((record["event_present"], record["nominal"]), (False, True))
        record = module.apply_base_com_pin(cfg, [0.02, 0.0, 0.0], template)
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"]["x"], (0.02, 0.02))
        self.assertEqual(record["source"], "pin")


class ScriptWiringTest(unittest.TestCase):
    def calls(self, path):
        tree = ast.parse(path.read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        order = []
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                order.append((node.lineno, node.func.id, node))
        return tree, sorted(order, key=lambda item: item[0])

    def flags(self, tree):
        found = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                    and isinstance(node.args[0], ast.Constant)):
                found[node.args[0].value] = {k.arg: k.value for k in node.keywords}
        return found

    def check(self, path, restore_name, later_names):
        tree, calls = self.calls(path)
        flags = self.flags(tree)
        self.assertEqual(ast.literal_eval(flags["--com_pin"]["nargs"]), 3)
        self.assertIsNone(ast.literal_eval(flags["--com_pin"]["default"]))
        self.assertEqual(flags["--com_pin"]["type"].id, "float")
        self.assertEqual(ast.literal_eval(flags["--com_keep_training"]["action"]), "store_true")
        names = [name for _, name, _ in calls]
        self.assertIn("apply_base_com_pin", names)
        pin_line = next(line for line, name, _ in calls if name == "apply_base_com_pin")
        pin_call = next(node for _, name, node in calls if name == "apply_base_com_pin")
        self.assertEqual([ast.unparse(a) for a in pin_call.args],
                         ["env_cfg", "args_cli.com_pin", "com_template", "args_cli.com_keep_training"])
        for name in [restore_name, *later_names]:
            line = next(line for line, n, _ in calls if n == name)
            self.assertLess(line, pin_line, name)
        make_line = next(node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and ast.unparse(node.func) == "gym.make")
        self.assertLess(pin_line, make_line)
        template_line = next(node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assign)
                             and getattr(node.targets[0], "id", None) == "com_template")
        restore_line = next(line for line, n, _ in calls if n == restore_name)
        self.assertLess(template_line, restore_line)

    def test_eval_commands_pins_after_restore_and_records_it(self):
        self.check(EVAL, "apply_training_env_cfg", ["pin_joint_friction"])
        text = EVAL.read_text()
        self.assertIn('"base_com": base_com,', text)

    def test_sim_joint_trace_pins_after_restore_and_randomization_off(self):
        self.check(TRACE, "apply_training_env_cfg", ["disable_randomization"])


if __name__ == "__main__":
    unittest.main()
