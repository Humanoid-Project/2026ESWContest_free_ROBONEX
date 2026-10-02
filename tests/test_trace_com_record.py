import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
ISAAC_DICT = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab/utils/dict.py")

DR = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}
ZERO = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
FUNC = "robonex_walking.tasks.manager_based.robonex_walking.mdp.events:randomize_rigid_body_com_offset"

SAVED_DR = """events:
  reset_base:
    mode: reset
  randomize_mass:
    mode: reset
  push_robot:
    mode: interval
  randomize_base_com:
    func: %s
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
""" % FUNC

SAVED_OLD = """events:
  reset_base:
    mode: reset
  randomize_mass:
    mode: reset
  push_robot:
    mode: interval
seed: 43
"""


def load_effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_trace_com", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def com_term(com_range):
    return types.SimpleNamespace(func=FUNC, mode="reset", params={
        "asset_cfg": types.SimpleNamespace(name="robot", body_names=["base_link"]),
        "com_range": dict(com_range)})


def registry_cfg():
    events = types.SimpleNamespace(
        reset_base=types.SimpleNamespace(mode="reset", params={}),
        randomize_mass=types.SimpleNamespace(mode="reset", params={}),
        push_robot=types.SimpleNamespace(mode="interval", params={}),
        randomize_base_com=com_term(ZERO))
    return types.SimpleNamespace(
        events=events, observations=types.SimpleNamespace(),
        scene=types.SimpleNamespace(num_envs=4096,
                                    robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=None))))


def main_prefix():
    tree = ast.parse(TRACE.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = []
    for stmt in main.body:
        body.append(stmt)
        if any(isinstance(n, ast.Call) and ast.unparse(n.func) == "apply_base_com_pin" for n in ast.walk(stmt)):
            return ast.Module(body=body, type_ignores=[])
    raise AssertionError("sim_joint_trace.main never calls apply_base_com_pin")


class TraceCallOrderTest(unittest.TestCase):
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

    def run_trace_setup(self, saved, com_pin=None, keep_training=False, keep_randomization=False):
        run = Path(self.tmp.name) / f"run{len(list(Path(self.tmp.name).iterdir()))}"
        (run / "params").mkdir(parents=True)
        (run / "params" / "env.yaml").write_text(saved)
        cfg = registry_cfg()
        effects = load_effects()
        args = types.SimpleNamespace(task="RoboNex-Walking-V2-Edu-v0", device="cpu", num_envs=1,
                                     checkpoint=str(run / "model_999.pt"), registry_cfg=False,
                                     keep_randomization=keep_randomization, set=[],
                                     com_pin=com_pin, com_keep_training=keep_training)
        records = []

        def recording_pin(*a, **k):
            records.append(effects.apply_base_com_pin(*a, **k))
            return records[-1]

        ns = {name: getattr(effects, name) for name in
              ("BASE_COM_EVENT", "apply_training_env_cfg", "disable_randomization")}
        ns["apply_base_com_pin"] = recording_pin
        ns.update(copy=copy, args_cli=args, print=lambda *a, **k: None,
                  parse_env_cfg=lambda task, device, num_envs: cfg)
        exec(compile(main_prefix(), str(TRACE), "exec"), ns)
        self.assertEqual(len(records), 1)
        return ns["env_cfg"], records[0]

    def test_default_runs_nominal_without_the_event(self):
        for saved in (SAVED_DR, SAVED_OLD):
            cfg, record = self.run_trace_setup(saved)
            self.assertIsNone(cfg.events.randomize_base_com)
            self.assertEqual(record, {"source": "default_nominal", "pin_m": [0.0, 0.0, 0.0], "event_present": False,
                                      "range_m": None, "nominal": True})
            self.assertIsNone(cfg.events.randomize_mass)
            self.assertIsNone(cfg.events.push_robot)
            self.assertIsNotNone(cfg.events.reset_base)

    def test_keep_training_survives_randomization_off_and_nothing_else_does(self):
        cfg, record = self.run_trace_setup(SAVED_DR, keep_training=True)
        com = cfg.events.randomize_base_com
        self.assertEqual(com.params["com_range"], DR)
        self.assertEqual((com.func, com.mode), (FUNC, "reset"))
        self.assertEqual(record, {"source": "training", "pin_m": None, "event_present": True,
                                  "range_m": {"x": [-0.02, 0.02], "y": [-0.02, 0.02], "z": [0.0, 0.0]},
                                  "nominal": False})
        self.assertIsNone(cfg.events.randomize_mass)
        self.assertIsNone(cfg.events.push_robot)
        self.assertIsNotNone(cfg.events.reset_base)

    def test_keep_training_on_a_run_without_the_event_reports_nominal(self):
        cfg, record = self.run_trace_setup(SAVED_OLD, keep_training=True)
        self.assertIsNone(cfg.events.randomize_base_com)
        self.assertEqual((record["source"], record["event_present"], record["nominal"]), ("training", False, True))

    def test_non_zero_pin_is_applied_with_and_without_the_trained_event(self):
        for saved in (SAVED_DR, SAVED_OLD):
            cfg, record = self.run_trace_setup(saved, com_pin=[0.02, -0.01, 0.0])
            com = cfg.events.randomize_base_com
            self.assertEqual(com.params["com_range"], {"x": (0.02, 0.02), "y": (-0.01, -0.01), "z": (0.0, 0.0)})
            self.assertEqual((com.func, com.mode), (FUNC, "reset"))
            if saved is SAVED_OLD:
                self.assertEqual(com.params["asset_cfg"].body_names, ["base_link"])
            self.assertEqual(record, {"source": "pin", "pin_m": [0.02, -0.01, 0.0], "event_present": True,
                                      "range_m": {"x": [0.02, 0.02], "y": [-0.01, -0.01], "z": [0.0, 0.0]},
                                      "nominal": False})
            self.assertIsNone(cfg.events.randomize_mass)
            self.assertIsNone(cfg.events.push_robot)

    def test_zero_pin_on_a_com_dr_run_keeps_a_zero_width_event(self):
        cfg, record = self.run_trace_setup(SAVED_DR, com_pin=[0.0, 0.0, 0.0])
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"], ZERO)
        self.assertEqual((record["source"], record["event_present"], record["nominal"]), ("pin", True, True))

    def test_keep_randomization_keeps_everything(self):
        cfg, record = self.run_trace_setup(SAVED_DR, keep_training=True, keep_randomization=True)
        self.assertEqual(cfg.events.randomize_base_com.params["com_range"], DR)
        self.assertIsNotNone(cfg.events.randomize_mass)
        self.assertIsNotNone(cfg.events.push_robot)
        self.assertFalse(record["nominal"])

    def test_disable_randomization_keeps_only_named_events_that_exist(self):
        effects = load_effects()
        cfg = registry_cfg()
        cfg.events.randomize_base_com = None
        kept = effects.disable_randomization(cfg, ("randomize_base_com", "missing"))
        self.assertEqual(kept, ["reset_base"])
        cfg = registry_cfg()
        self.assertEqual(effects.disable_randomization(cfg, ("randomize_base_com",)),
                         ["reset_base", "randomize_base_com"])
        self.assertEqual(effects.disable_randomization(registry_cfg()), ["reset_base"])


def meta_dict(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and getattr(node.targets[0], "id", None) == "meta"):
            return node.value
    raise AssertionError(f"no meta dict in {path.name}")


class TraceSidecarTest(unittest.TestCase):
    def test_sidecar_records_the_pin_helper_result(self):
        meta = meta_dict(TRACE)
        entries = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(meta.keys, meta.values)}
        self.assertEqual(entries.get("base_com"), "base_com")
        tree = ast.parse(TRACE.read_text())
        targets = [ast.unparse(n.targets[0]) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "apply_base_com_pin"]
        self.assertEqual(targets, ["base_com"])
        self.assertIn('"base_com": base_com,', EVAL.read_text())

    def test_the_record_is_json_and_matches_the_eval_shape(self):
        effects = load_effects()
        cfg = types.SimpleNamespace(events=types.SimpleNamespace(randomize_base_com=com_term(DR)))
        record = effects.apply_base_com_pin(cfg, [0.02, 0.0, 0.0])
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertEqual(set(record), {"source", "pin_m", "event_present", "range_m", "nominal"})


if __name__ == "__main__":
    unittest.main()
