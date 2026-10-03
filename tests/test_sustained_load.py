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

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"
EVAL = ROOT / "scripts/rsl_rl/eval_commands.py"
TRACE = ROOT / "scripts/rsl_rl/sim_joint_trace.py"
ISAAC = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab")
ISAAC_DICT = ISAAC / "utils/dict.py"
FUNC = "robonex_walking.tasks.manager_based.robonex_walking.mdp.events:apply_sustained_base_load"

ZERO = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
FORCE_DR = {"x": (-2.0, 2.0), "y": (-3.0, 3.0), "z": (0.0, 0.0)}
TORQUE_DR = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (-0.4, 0.4)}


def load_event():
    wanted = {"_LOAD_AXES", "_sustained_load_bounds", "apply_sustained_base_load"}
    body = [n for n in ast.parse(EVENTS.read_text()).body
            if getattr(n, "name", None) in wanted
            or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) in wanted)]
    ns = {"torch": torch, "SceneEntityCfg": object, "Articulation": object}
    exec(compile(ast.Module(body=body, type_ignores=[]), "events", "exec"), ns)
    return ns["apply_sustained_base_load"]


def load_push_standing():
    node = next(n for n in ast.parse(EVENTS.read_text()).body
                if getattr(n, "name", None) == "push_standing_by_setting_velocity")
    ns = {"torch": torch, "SceneEntityCfg": lambda name: types.SimpleNamespace(name=name), "Articulation": object}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "events", "exec"), ns)
    return ns["push_standing_by_setting_velocity"]


def load_effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_sustained_load", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cfg_tree():
    return ast.parse(CFG.read_text())


def constant(name):
    for node in cfg_tree().body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def event_term(name):
    events = next(n for n in cfg_tree().body if isinstance(n, ast.ClassDef) and n.name == "EventCfg")
    for node in events.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == name:
            return {k.arg: k.value for k in node.value.keywords}
    raise KeyError(name)


class FakeComposer:
    def __init__(self, num_envs, num_bodies):
        self.force = torch.zeros(num_envs, num_bodies, 3)
        self.torque = torch.zeros(num_envs, num_bodies, 3)
        self.active = False
        self.calls = []

    def set_forces_and_torques(self, forces=None, torques=None, positions=None, body_ids=None, env_ids=None,
                               is_global=False):
        self.calls.append({"forces": forces.clone(), "torques": torques.clone(), "positions": positions,
                           "body_ids": list(body_ids), "env_ids": env_ids.clone(), "is_global": is_global})
        self.active = True
        rows = env_ids.long()[:, None]
        cols = torch.tensor(body_ids, dtype=torch.long)[None, :]
        self.force[rows, cols] = forces
        self.torque[rows, cols] = torques

    def reset(self, env_ids=None):
        if env_ids is None:
            self.force.zero_()
            self.torque.zero_()
            self.active = False
        else:
            self.force[env_ids] = 0.0
            self.torque[env_ids] = 0.0


class FakeAsset:
    def __init__(self, num_envs, num_bodies):
        self.num_bodies = num_bodies
        self.device = "cpu"
        self.permanent_wrench_composer = FakeComposer(num_envs, num_bodies)
        self.instantaneous_wrench_composer = FakeComposer(num_envs, num_bodies)
        self.data = types.SimpleNamespace(root_vel_w=torch.zeros(num_envs, 6))
        self.velocity_writes = []

    def reset(self, env_ids=None):
        self.instantaneous_wrench_composer.reset(env_ids)
        self.permanent_wrench_composer.reset(env_ids)

    def write_root_velocity_to_sim(self, vel, env_ids):
        self.velocity_writes.append(env_ids.clone())
        self.data.root_vel_w[env_ids] = vel


class Scene(dict):
    def __init__(self, asset, num_envs):
        super().__init__(robot=asset)
        self.num_envs = num_envs


class SustainedLoadEventTest(unittest.TestCase):
    NUM_ENVS = 64
    BASE = 0

    def setUp(self):
        self.asset = FakeAsset(self.NUM_ENVS, num_bodies=5)
        self.env = types.SimpleNamespace(scene=Scene(self.asset, self.NUM_ENVS))
        self.asset_cfg = types.SimpleNamespace(name="robot", body_ids=[self.BASE])
        self.event = load_event()
        self.composer = self.asset.permanent_wrench_composer

    def call(self, env_ids, force_range, torque_range, height=0.0):
        self.event(self.env, env_ids, self.asset_cfg, force_range, torque_range, height)

    def test_default_ranges_are_inert_and_draw_nothing(self):
        force, torque, height = (constant("SUSTAINED_LOAD_FORCE_RANGE"), constant("SUSTAINED_LOAD_TORQUE_RANGE"),
                                 constant("SUSTAINED_LOAD_HEIGHT_M"))
        rng = torch.get_rng_state()
        for env_ids in (None, torch.arange(self.NUM_ENVS), torch.tensor([3, 5])):
            self.call(env_ids, dict(force), dict(torque), height)
        self.call(None, dict(force), dict(torque), 0.3)
        self.call(None, {}, {}, 0.0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(self.composer.calls, [])
        self.assertFalse(self.composer.active)
        self.assertFalse(self.asset.instantaneous_wrench_composer.active)

    def test_draws_per_env_inside_the_ranges_in_the_body_frame(self):
        torch.manual_seed(0)
        self.call(None, FORCE_DR, TORQUE_DR)
        call = self.composer.calls[-1]
        self.assertFalse(call["is_global"])
        self.assertIsNone(call["positions"])
        self.assertEqual(call["body_ids"], [self.BASE])
        self.assertEqual(call["env_ids"].tolist(), list(range(self.NUM_ENVS)))
        self.assertEqual(tuple(call["forces"].shape), (self.NUM_ENVS, 1, 3))
        self.assertEqual(tuple(call["torques"].shape), (self.NUM_ENVS, 1, 3))
        self.assertTrue(call["forces"].is_contiguous() and call["torques"].is_contiguous())
        force = self.composer.force[:, self.BASE]
        torque = self.composer.torque[:, self.BASE]
        for values, axis, bound in ((force, 0, 2.0), (force, 1, 3.0), (torque, 2, 0.4)):
            self.assertTrue(bool(torch.all(values[:, axis].abs() <= bound + 1e-6)))
            self.assertGreater(float(values[:, axis].std()), 0.2 * bound)
        self.assertTrue(torch.equal(force[:, 2], torch.zeros(self.NUM_ENVS)))
        self.assertTrue(torch.equal(torque[:, :2], torch.zeros(self.NUM_ENVS, 2)))
        self.assertNotEqual(force[:, 0].tolist(), force[:, 1].tolist())
        others = [b for b in range(5) if b != self.BASE]
        self.assertTrue(torch.equal(self.composer.force[:, others], torch.zeros(self.NUM_ENVS, 4, 3)))
        self.assertTrue(torch.equal(self.composer.torque[:, others], torch.zeros(self.NUM_ENVS, 4, 3)))

    def test_many_resets_cover_the_range_without_accumulating(self):
        torch.manual_seed(1)
        seen = []
        for _ in range(100):
            self.asset.reset(None)
            self.call(None, FORCE_DR, TORQUE_DR)
            seen.append(self.composer.force[:, self.BASE, 1].clone())
        spread = torch.stack(seen)
        self.assertLessEqual(float(spread.abs().max()), 3.0 + 1e-6)
        self.assertGreater(float(spread.max()), 2.95)
        self.assertLess(float(spread.min()), -2.95)
        self.assertLess(abs(float(spread.mean())), 0.1)

    def test_application_height_adds_the_lever_torque(self):
        torch.manual_seed(2)
        height = 0.3
        self.call(None, FORCE_DR, TORQUE_DR, height)
        force = self.composer.force[:, self.BASE]
        torque = self.composer.torque[:, self.BASE]
        lever = torch.tensor([0.0, 0.0, height]).expand_as(force)
        drawn = torque - torch.linalg.cross(lever, force, dim=-1)
        torch.testing.assert_close(torque[:, 0], -height * force[:, 1])
        torch.testing.assert_close(torque[:, 1], height * force[:, 0])
        self.assertTrue(bool(torch.all(drawn[:, 2].abs() <= 0.4 + 1e-6)))
        torch.testing.assert_close(drawn[:, :2], torch.zeros(self.NUM_ENVS, 2), atol=1e-6, rtol=0.0)

    def test_rope_case_matches_the_mujoco_harness_wrench(self):
        self.call(None, {"y": (2.0, 2.0)}, {}, 0.3)
        torch.testing.assert_close(self.composer.force[0, self.BASE], torch.tensor([0.0, 2.0, 0.0]))
        torch.testing.assert_close(self.composer.torque[0, self.BASE], torch.tensor([-0.6, 0.0, 0.0]))

    def test_fixed_load_without_width_draws_nothing_and_is_the_same_on_every_env(self):
        rng = torch.get_rng_state()
        self.call(None, {"x": (0.0, 0.0), "y": (2.0, 2.0), "z": (0.0, 0.0)}, {"z": (0.25, 0.25)})
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        torch.testing.assert_close(self.composer.force[:, self.BASE],
                                   torch.tensor([0.0, 2.0, 0.0]).expand(self.NUM_ENVS, 3))
        torch.testing.assert_close(self.composer.torque[:, self.BASE],
                                   torch.tensor([0.0, 0.0, 0.25]).expand(self.NUM_ENVS, 3))

    def test_partial_reset_redraws_only_the_reset_envs(self):
        torch.manual_seed(3)
        self.call(None, FORCE_DR, TORQUE_DR, 0.3)
        before_f = self.composer.force.clone()
        before_t = self.composer.torque.clone()
        ids = torch.tensor([3, 7, 40])
        self.asset.reset(ids)
        self.assertTrue(torch.equal(self.composer.force[ids], torch.zeros(3, 5, 3)))
        self.call(ids, FORCE_DR, TORQUE_DR, 0.3)
        self.assertEqual(self.composer.calls[-1]["env_ids"].tolist(), [3, 7, 40])
        keep = [i for i in range(self.NUM_ENVS) if i not in (3, 7, 40)]
        self.assertTrue(torch.equal(self.composer.force[keep], before_f[keep]))
        self.assertTrue(torch.equal(self.composer.torque[keep], before_t[keep]))
        self.assertFalse(torch.equal(self.composer.force[ids], before_f[ids]))
        self.assertTrue(bool(torch.all(self.composer.force[ids, self.BASE, 1].abs() <= 3.0 + 1e-6)))
        self.assertTrue(bool(torch.all(self.composer.force[ids, self.BASE, 1] != 0.0)))

    def test_push_events_leave_the_load_alone(self):
        torch.manual_seed(4)
        self.call(None, FORCE_DR, TORQUE_DR)
        force = self.composer.force.clone()
        torque = self.composer.torque.clone()
        push = load_push_standing()
        self.env.command_manager = types.SimpleNamespace(get_command=lambda name: torch.zeros(self.NUM_ENVS, 3))
        push(self.env, torch.arange(self.NUM_ENVS), {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
             asset_cfg=types.SimpleNamespace(name="robot"))
        self.assertEqual(len(self.asset.velocity_writes), 1)
        self.assertTrue(torch.equal(self.composer.force, force))
        self.assertTrue(torch.equal(self.composer.torque, torque))
        self.assertEqual(len(self.composer.calls), 1)

    def test_bad_inputs_are_refused(self):
        for force, torque, height in (({"y": (3.0, -3.0)}, {}, 0.0), ({}, {"z": (0.4, -0.4)}, 0.0),
                                      ({"y": (float("nan"), 3.0)}, {}, 0.0), ({"y": (-3.0, 3.0)}, {}, float("inf")),
                                      ({}, {"z": (-float("inf"), 0.0)}, 0.0)):
            with self.assertRaises(ValueError):
                self.call(None, force, torque, height)
        self.asset_cfg.body_ids = slice(None)
        with self.assertRaises(ValueError):
            self.call(None, FORCE_DR, TORQUE_DR)
        self.asset_cfg.body_ids = [0, 1]
        with self.assertRaises(ValueError):
            self.call(None, FORCE_DR, TORQUE_DR)
        self.asset_cfg.body_ids = slice(2, 3)
        self.call(None, FORCE_DR, TORQUE_DR)
        self.assertEqual(self.composer.calls[-1]["body_ids"], [2])


@unittest.skipUnless((ISAAC / "utils/wrench_composer.py").exists(), "Isaac Lab source not present")
class IsaacLabContractTest(unittest.TestCase):
    def function(self, path, name, cls=None):
        tree = ast.parse(path.read_text())
        scope = tree.body if cls is None else next(n for n in tree.body if getattr(n, "name", None) == cls).body
        return next(n for n in scope if getattr(n, "name", None) == name)

    def test_composer_set_signature_and_body_frame_storage(self):
        node = self.function(ISAAC / "utils/wrench_composer.py", "set_forces_and_torques", "WrenchComposer")
        args = [a.arg for a in node.args.args]
        self.assertEqual(args, ["self", "forces", "torques", "positions", "body_ids", "env_ids", "is_global"])
        self.assertEqual(ast.literal_eval(node.args.defaults[-1]), False)

    def test_articulation_reset_clears_and_write_reapplies_the_permanent_wrench(self):
        reset = ast.unparse(self.function(ISAAC / "assets/articulation/articulation.py", "reset", "Articulation"))
        self.assertIn("self._permanent_wrench_composer.reset(env_ids)", reset)
        write = ast.unparse(self.function(ISAAC / "assets/articulation/articulation.py", "write_data_to_sim",
                                          "Articulation"))
        self.assertIn("self._permanent_wrench_composer.composed_force_as_torch", write)
        self.assertIn("is_global=False", write)
        self.assertNotIn("self._permanent_wrench_composer.reset()", write)

    def test_scene_reset_runs_before_the_reset_events(self):
        body = ast.unparse(self.function(ISAAC / "envs/manager_based_rl_env.py", "_reset_idx", "ManagerBasedRLEnv"))
        self.assertLess(body.index("self.scene.reset(env_ids)"), body.index("self.event_manager.apply(mode='reset'"))

    def test_stock_push_does_not_touch_wrenches(self):
        body = ast.unparse(self.function(ISAAC / "envs/mdp/events.py", "push_by_setting_velocity"))
        self.assertIn("write_root_velocity_to_sim", body)
        self.assertNotIn("wrench", body)


class WiringTest(unittest.TestCase):
    def test_constants_are_zero(self):
        self.assertEqual(constant("SUSTAINED_LOAD_FORCE_RANGE"), ZERO)
        self.assertEqual(constant("SUSTAINED_LOAD_TORQUE_RANGE"), ZERO)
        self.assertEqual(constant("SUSTAINED_LOAD_HEIGHT_M"), 0.0)

    def test_event_term_is_a_reset_event_on_base_link(self):
        term = event_term("sustained_base_load")
        self.assertEqual(ast.unparse(term["func"]), "mdp.apply_sustained_base_load")
        self.assertEqual(ast.literal_eval(term["mode"]), "reset")
        params = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(term["params"].keys, term["params"].values)}
        self.assertEqual(params, {
            "asset_cfg": "SceneEntityCfg('robot', body_names=['base_link'])",
            "force_range": "dict(SUSTAINED_LOAD_FORCE_RANGE)",
            "torque_range": "dict(SUSTAINED_LOAD_TORQUE_RANGE)",
            "application_height_m": "SUSTAINED_LOAD_HEIGHT_M",
        })

    def test_push_events_are_unchanged(self):
        self.assertEqual(ast.unparse(event_term("push_robot")["func"]), "mdp.push_by_setting_velocity")
        self.assertEqual(ast.unparse(event_term("push_standing")["func"]), "mdp.push_standing_by_setting_velocity")


def update_class_from_dict():
    node = next(n for n in ast.parse(ISAAC_DICT.read_text()).body
                if getattr(n, "name", None) == "update_class_from_dict")
    ns = {"Iterable": Iterable, "Mapping": Mapping, "Sized": Sized, "Any": object,
          "string_to_callable": lambda s: s}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "dict", "exec"), ns)
    return ns["update_class_from_dict"]


def load_term(force=ZERO, torque=ZERO, height=0.0):
    return types.SimpleNamespace(func=FUNC, mode="reset", params={
        "asset_cfg": types.SimpleNamespace(name="robot", body_names=["base_link"]),
        "force_range": dict(force), "torque_range": dict(torque), "application_height_m": height})


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
class HydraOverrideTest(unittest.TestCase):
    def parse(self, text):
        from hydra.core.override_parser.overrides_parser import OverridesParser

        override = OverridesParser.create().parse_overrides([text])[0]
        data = override.value()
        for key in reversed(override.key_or_group.split(".")[1:]):
            data = {key: data}
        return data

    def test_launch_overrides_round_trip_as_tuples(self):
        update = update_class_from_dict()
        env = types.SimpleNamespace(events=types.SimpleNamespace(sustained_base_load=load_term()))
        for text in ("env.events.sustained_base_load.params.force_range.y=[-3.0,3.0]",
                     "env.events.sustained_base_load.params.force_range.x=[-2.0,2.0]",
                     "env.events.sustained_base_load.params.torque_range.z=[-0.4,0.4]",
                     "env.events.sustained_base_load.params.application_height_m=0.3"):
            update(env, self.parse(text))
        params = env.events.sustained_base_load.params
        self.assertEqual(params["force_range"], FORCE_DR)
        self.assertEqual(params["torque_range"], TORQUE_DR)
        self.assertEqual(params["application_height_m"], 0.3)
        self.assertIsInstance(params["force_range"]["y"], tuple)
        self.assertIsInstance(params["torque_range"]["z"], tuple)
        self.assertEqual(constant("SUSTAINED_LOAD_FORCE_RANGE"), ZERO)
        torch.manual_seed(5)
        asset = FakeAsset(8, 3)
        env_obj = types.SimpleNamespace(scene=Scene(asset, 8))
        load_event()(env_obj, None, types.SimpleNamespace(name="robot", body_ids=[0]),
                     params["force_range"], params["torque_range"], params["application_height_m"])
        force = asset.permanent_wrench_composer.force[:, 0]
        self.assertTrue(bool(torch.all(force[:, 1].abs() <= 3.0)) and float(force[:, 1].abs().max()) > 0.0)

    def test_an_absent_term_cannot_be_added_by_override(self):
        update = update_class_from_dict()
        env = types.SimpleNamespace(events=types.SimpleNamespace(sustained_base_load=load_term()))
        with self.assertRaises(KeyError):
            update(env, self.parse("env.events.sustained_other_load.params.force_range.y=[-3.0,3.0]"))


class PinTest(unittest.TestCase):
    def cfg(self, term):
        return types.SimpleNamespace(events=types.SimpleNamespace(sustained_base_load=term))

    def test_default_pins_a_trained_range_to_zero(self):
        cfg = self.cfg(load_term(FORCE_DR, TORQUE_DR, 0.3))
        record = load_effects().apply_base_load_pin(cfg)
        params = cfg.events.sustained_base_load.params
        self.assertEqual(params["force_range"], ZERO)
        self.assertEqual(params["torque_range"], ZERO)
        self.assertEqual(params["application_height_m"], 0.0)
        self.assertEqual(params["asset_cfg"].body_names, ["base_link"])
        zero = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}
        self.assertEqual(record, {"source": "default_zero", "pin": [0.0] * 6, "event_present": True,
                                  "force_range_n": zero, "torque_range_nm": zero, "application_height_m": 0.0,
                                  "zero_load": True})

    def test_explicit_pin_is_a_zero_width_wrench_at_the_link_origin(self):
        cfg = self.cfg(load_term(FORCE_DR, TORQUE_DR, 0.3))
        record = load_effects().apply_base_load_pin(cfg, [0.0, 2.0, 0.0, -0.6, 0.0, 0.3])
        params = cfg.events.sustained_base_load.params
        self.assertEqual(params["force_range"], {"x": (0.0, 0.0), "y": (2.0, 2.0), "z": (0.0, 0.0)})
        self.assertEqual(params["torque_range"], {"x": (-0.6, -0.6), "y": (0.0, 0.0), "z": (0.3, 0.3)})
        self.assertEqual(params["application_height_m"], 0.0)
        self.assertEqual((record["source"], record["pin"], record["zero_load"]),
                         ("pin", [0.0, 2.0, 0.0, -0.6, 0.0, 0.3], False))

    def test_keep_training_leaves_the_range_and_records_it(self):
        cfg = self.cfg(load_term(FORCE_DR, TORQUE_DR, 0.3))
        record = load_effects().apply_base_load_pin(cfg, keep_training=True)
        params = cfg.events.sustained_base_load.params
        self.assertEqual((params["force_range"], params["torque_range"]), (FORCE_DR, TORQUE_DR))
        self.assertEqual(record["source"], "training")
        self.assertIsNone(record["pin"])
        self.assertEqual(record["force_range_n"]["y"], [-3.0, 3.0])
        self.assertEqual(record["torque_range_nm"]["z"], [-0.4, 0.4])
        self.assertEqual(record["application_height_m"], 0.3)
        self.assertFalse(record["zero_load"])

    def test_pin_and_keep_training_exclude_each_other_and_bad_values_are_refused(self):
        effects = load_effects()
        with self.assertRaises(ValueError):
            effects.apply_base_load_pin(self.cfg(load_term()), [0.0] * 6, keep_training=True)
        for load in ([0.0] * 5, [0.0] * 7, [float("nan")] + [0.0] * 5, [0.0] * 5 + [float("inf")]):
            with self.assertRaises(ValueError):
                effects.apply_base_load_pin(self.cfg(load_term()), load)

    def test_a_run_without_the_event_stays_without_it(self):
        effects = load_effects()
        for load in (None, [0.0] * 6):
            cfg = self.cfg(None)
            record = effects.apply_base_load_pin(cfg, load, template=load_term())
            self.assertIsNone(cfg.events.sustained_base_load)
            self.assertEqual((record["event_present"], record["zero_load"], record["force_range_n"]),
                             (False, True, None))
        cfg = types.SimpleNamespace(events=types.SimpleNamespace())
        self.assertTrue(effects.apply_base_load_pin(cfg)["zero_load"])
        self.assertFalse(hasattr(cfg.events, "sustained_base_load"))

    def test_a_non_zero_pin_on_a_run_without_the_event_installs_a_copy_of_the_template(self):
        effects = load_effects()
        template = load_term()
        cfg = self.cfg(None)
        record = effects.apply_base_load_pin(cfg, [0.0, 0.0, 0.0, 0.0, 0.0, 0.3], template=template)
        installed = cfg.events.sustained_base_load
        self.assertIsNot(installed, template)
        self.assertEqual(template.params["torque_range"], ZERO)
        self.assertEqual(installed.params["torque_range"]["z"], (0.3, 0.3))
        self.assertEqual((installed.func, installed.mode), (FUNC, "reset"))
        self.assertTrue(record["event_present"])
        with self.assertRaises(KeyError):
            effects.apply_base_load_pin(self.cfg(None), [0.0, 2.0, 0.0, 0.0, 0.0, 0.0], template=None)

    def test_the_record_is_json(self):
        record = load_effects().apply_base_load_pin(self.cfg(load_term(FORCE_DR, TORQUE_DR, 0.3)),
                                                    [1.0, 2.0, 0.0, 0.0, 0.0, 0.1])
        self.assertEqual(json.loads(json.dumps(record)), record)

    def test_pinned_cfg_drives_the_event_to_the_same_wrench_on_every_env_and_reset(self):
        cfg = self.cfg(load_term(FORCE_DR, TORQUE_DR, 0.3))
        load_effects().apply_base_load_pin(cfg, [0.0, 2.0, 0.0, -0.6, 0.0, 0.3])
        params = dict(cfg.events.sustained_base_load.params)
        params.pop("asset_cfg")
        asset = FakeAsset(6, 3)
        env = types.SimpleNamespace(scene=Scene(asset, 6))
        for _ in range(3):
            state = torch.get_rng_state()
            asset.reset(None)
            load_event()(env, None, types.SimpleNamespace(name="robot", body_ids=[0]), **params)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            torch.testing.assert_close(asset.permanent_wrench_composer.force[:, 0],
                                       torch.tensor([0.0, 2.0, 0.0]).expand(6, 3))
            torch.testing.assert_close(asset.permanent_wrench_composer.torque[:, 0],
                                       torch.tensor([-0.6, 0.0, 0.3]).expand(6, 3))

    def test_default_zero_pin_touches_nothing(self):
        cfg = self.cfg(load_term(FORCE_DR, TORQUE_DR, 0.3))
        load_effects().apply_base_load_pin(cfg)
        params = dict(cfg.events.sustained_base_load.params)
        params.pop("asset_cfg")
        asset = FakeAsset(6, 3)
        load_event()(types.SimpleNamespace(scene=Scene(asset, 6)), torch.arange(6),
                     types.SimpleNamespace(name="robot", body_ids=[0]), **params)
        self.assertEqual(asset.permanent_wrench_composer.calls, [])


SAVED_LOAD = """events:
  reset_base:
    mode: reset
  randomize_mass:
    mode: reset
  push_robot:
    mode: interval
  sustained_base_load:
    func: %s
    mode: reset
    params:
      force_range:
        x: !!python/tuple
        - -2.0
        - 2.0
        y: !!python/tuple
        - -3.0
        - 3.0
        z: !!python/tuple
        - 0.0
        - 0.0
      torque_range:
        x: !!python/tuple
        - 0.0
        - 0.0
        y: !!python/tuple
        - 0.0
        - 0.0
        z: !!python/tuple
        - -0.4
        - 0.4
      application_height_m: 0.3
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


def registry_cfg():
    events = types.SimpleNamespace(
        reset_base=types.SimpleNamespace(mode="reset", params={}),
        randomize_mass=types.SimpleNamespace(mode="reset", params={}),
        push_robot=types.SimpleNamespace(mode="interval", params={}),
        randomize_base_com=None,
        sustained_base_load=load_term())
    return types.SimpleNamespace(
        events=events, observations=types.SimpleNamespace(),
        scene=types.SimpleNamespace(num_envs=4096,
                                    robot=types.SimpleNamespace(spawn=types.SimpleNamespace(usd_path=None))))


class IsaacDictStub(unittest.TestCase):
    def setUp(self):
        fake = types.ModuleType("isaaclab.utils.dict")
        fake.update_class_from_dict = update_class_from_dict()
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

    def write_run(self, saved):
        run = Path(self.tmp.name) / f"run{len(list(Path(self.tmp.name).iterdir()))}"
        (run / "params").mkdir(parents=True)
        (run / "params" / "env.yaml").write_text(saved)
        return run


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
class RestoreThenPinTest(IsaacDictStub):
    def restore(self, saved):
        run = self.write_run(saved)
        cfg = registry_cfg()
        effects = load_effects()
        template = copy.deepcopy(cfg.events.sustained_base_load)
        effects.apply_training_env_cfg(cfg, str(run / "model_999.pt"))
        return effects, cfg, template

    def test_a_load_dr_run_restores_its_range_and_the_default_pins_it_to_zero(self):
        effects, cfg, template = self.restore(SAVED_LOAD)
        params = cfg.events.sustained_base_load.params
        self.assertEqual((params["force_range"], params["torque_range"]), (FORCE_DR, TORQUE_DR))
        self.assertEqual(params["application_height_m"], 0.3)
        record = effects.apply_base_load_pin(cfg, None, template)
        self.assertEqual((record["source"], record["zero_load"]), ("default_zero", True))
        self.assertEqual(cfg.events.sustained_base_load.params["force_range"], ZERO)

    def test_a_load_dr_run_keeps_its_range_when_asked(self):
        effects, cfg, template = self.restore(SAVED_LOAD)
        record = effects.apply_base_load_pin(cfg, None, template, keep_training=True)
        self.assertEqual(record["force_range_n"], {"x": [-2.0, 2.0], "y": [-3.0, 3.0], "z": [0.0, 0.0]})
        self.assertEqual(record["application_height_m"], 0.3)
        self.assertFalse(record["zero_load"])

    def test_an_older_run_without_the_term_restores_without_it(self):
        effects, cfg, template = self.restore(SAVED_OLD)
        self.assertIsNone(cfg.events.sustained_base_load)
        record = effects.apply_base_load_pin(cfg, None, template)
        self.assertEqual((record["event_present"], record["zero_load"]), (False, True))
        record = effects.apply_base_load_pin(cfg, [0.0, 2.0, 0.0, 0.0, 0.0, 0.0], template)
        self.assertEqual(cfg.events.sustained_base_load.params["force_range"]["y"], (2.0, 2.0))
        self.assertEqual(record["source"], "pin")


def main_prefix(path, last_call):
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = []
    for stmt in main.body:
        body.append(stmt)
        if any(isinstance(n, ast.Call) and ast.unparse(n.func) == last_call for n in ast.walk(stmt)):
            return ast.Module(body=body, type_ignores=[])
    raise AssertionError(f"{path.name} main never calls {last_call}")


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
class TraceSetupTest(IsaacDictStub):
    def run_setup(self, saved, load_pin=None, keep_training=False, keep_randomization=False):
        run = self.write_run(saved)
        cfg = registry_cfg()
        effects = load_effects()
        args = types.SimpleNamespace(task="RoboNex-Walking-V2-Edu-v0", device="cpu", num_envs=1,
                                     checkpoint=str(run / "model_999.pt"), registry_cfg=False,
                                     keep_randomization=keep_randomization, set=[], com_pin=None,
                                     com_keep_training=False, load_pin=load_pin, load_keep_training=keep_training)
        ns = {name: getattr(effects, name) for name in
              ("BASE_COM_EVENT", "BASE_LOAD_EVENT", "apply_training_env_cfg", "disable_randomization",
               "apply_base_com_pin", "apply_base_load_pin")}
        ns.update(copy=copy, args_cli=args, print=lambda *a, **k: None,
                  parse_env_cfg=lambda task, device, num_envs: cfg)
        exec(compile(main_prefix(TRACE, "apply_base_load_pin"), str(TRACE), "exec"), ns)
        return ns["env_cfg"], ns["base_load"]

    def test_default_trace_runs_without_load(self):
        for saved in (SAVED_LOAD, SAVED_OLD):
            cfg, record = self.run_setup(saved)
            self.assertIsNone(cfg.events.sustained_base_load)
            self.assertEqual((record["source"], record["event_present"], record["zero_load"]),
                             ("default_zero", False, True))
            self.assertIsNone(cfg.events.push_robot)

    def test_keep_training_survives_randomization_off(self):
        cfg, record = self.run_setup(SAVED_LOAD, keep_training=True)
        self.assertEqual(cfg.events.sustained_base_load.params["force_range"], FORCE_DR)
        self.assertEqual((record["source"], record["zero_load"]), ("training", False))
        self.assertIsNone(cfg.events.randomize_mass)

    def test_non_zero_pin_with_and_without_the_trained_event(self):
        for saved in (SAVED_LOAD, SAVED_OLD):
            cfg, record = self.run_setup(saved, load_pin=[0.0, 2.0, 0.0, -0.6, 0.0, 0.0])
            term = cfg.events.sustained_base_load
            self.assertEqual(term.params["force_range"]["y"], (2.0, 2.0))
            self.assertEqual(term.params["torque_range"]["x"], (-0.6, -0.6))
            self.assertEqual(term.params["application_height_m"], 0.0)
            self.assertEqual((term.func, term.mode), (FUNC, "reset"))
            self.assertEqual((record["source"], record["event_present"], record["zero_load"]), ("pin", True, False))
            self.assertIsNone(cfg.events.randomize_mass)


class ScriptWiringTest(unittest.TestCase):
    def flags(self, tree):
        found = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                    and isinstance(node.args[0], ast.Constant)):
                found[node.args[0].value] = {k.arg: k.value for k in node.keywords}
        return found

    def check(self, path, before_names):
        tree = ast.parse(path.read_text())
        flags = self.flags(tree)
        self.assertEqual(ast.literal_eval(flags["--load_pin"]["nargs"]), 6)
        self.assertIsNone(ast.literal_eval(flags["--load_pin"]["default"]))
        self.assertEqual(flags["--load_pin"]["type"].id, "float")
        self.assertEqual(ast.literal_eval(flags["--load_keep_training"]["action"]), "store_true")
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = sorted(((n.lineno, ast.unparse(n.func), n) for n in ast.walk(main) if isinstance(n, ast.Call)),
                       key=lambda item: item[0])
        pin = [(line, node) for line, name, node in calls if name == "apply_base_load_pin"]
        self.assertEqual(len(pin), 1)
        pin_line, pin_call = pin[0]
        self.assertEqual([ast.unparse(a) for a in pin_call.args],
                         ["env_cfg", "args_cli.load_pin", "load_template", "args_cli.load_keep_training"])
        for name in before_names:
            self.assertLess(next(line for line, n, _ in calls if n == name), pin_line, name)
        self.assertLess(pin_line, next(line for line, n, _ in calls if n == "gym.make"))
        template_line = next(n.lineno for n in ast.walk(main) if isinstance(n, ast.Assign)
                             and getattr(n.targets[0], "id", None) == "load_template")
        self.assertLess(template_line, next(line for line, n, _ in calls if n == "apply_training_env_cfg"))
        targets = [ast.unparse(n.targets[0]) for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == "apply_base_load_pin"]
        self.assertEqual(targets, ["base_load"])
        return tree

    def test_eval_commands_pins_after_restore_and_records_in_deploy_effects(self):
        tree = self.check(EVAL, ["apply_training_env_cfg", "pin_joint_friction", "apply_base_com_pin"])
        applied = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and getattr(n.targets[0], "id", None) == "applied" and isinstance(n.value, ast.Dict))
        effects = next(v for k, v in zip(applied.keys, applied.values) if ast.literal_eval(k) == "deploy_effects")
        entries = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(effects.keys, effects.values)}
        self.assertEqual(entries.get("base_load"), "base_load")

    def test_sim_joint_trace_pins_after_randomization_off_and_records_in_the_sidecar(self):
        tree = self.check(TRACE, ["apply_training_env_cfg", "disable_randomization", "apply_base_com_pin"])
        meta = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], "id", None) == "meta" and isinstance(n.value, ast.Dict))
        entries = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(meta.keys, meta.values)}
        self.assertEqual(entries.get("base_load"), "base_load")


if __name__ == "__main__":
    unittest.main()
