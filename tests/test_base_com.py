import ast
import types
import unittest
from collections.abc import Iterable, Mapping, Sized
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
EVENTS = TASK / "mdp/events.py"
CFG = TASK / "robonex_walking_v2_env_cfg.py"
ISAAC_DICT = Path("/home/polygon/IsaacLab/source/isaaclab/isaaclab/utils/dict.py")


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


class FakeView:
    def __init__(self, coms):
        self.coms = coms.clone()
        self.gets = 0
        self.sets = []

    def get_coms(self):
        self.gets += 1
        return self.coms.clone()

    def set_coms(self, coms, env_ids):
        self.sets.append(env_ids.clone())
        self.coms[env_ids] = coms[env_ids]


class FakeAsset:
    def __init__(self, num_envs, num_bodies):
        torch.manual_seed(123)
        coms = torch.zeros(num_envs, num_bodies, 7)
        coms[..., :3] = 0.01 * torch.randn(num_envs, num_bodies, 3)
        coms[..., 6] = 1.0
        self.nominal = coms.clone()
        self.root_physx_view = FakeView(coms)
        self.num_bodies = num_bodies


class Scene(dict):
    def __init__(self, asset, num_envs):
        super().__init__(robot=asset)
        self.num_envs = num_envs


class ComOffsetEventTest(unittest.TestCase):
    NUM_ENVS = 64
    BASE = 0
    WIDE = {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}

    def setUp(self):
        self.asset = FakeAsset(self.NUM_ENVS, num_bodies=5)
        self.env = types.SimpleNamespace(scene=Scene(self.asset, self.NUM_ENVS))
        self.asset_cfg = types.SimpleNamespace(name="robot", body_ids=[self.BASE])
        self.term = load_event()(cfg=None, env=self.env)

    def call(self, env_ids, com_range):
        self.term(self.env, env_ids, self.asset_cfg, com_range)
        return self.asset.root_physx_view.coms

    def offset(self, coms):
        return coms[:, self.BASE, :3] - self.asset.nominal[:, self.BASE, :3]

    def test_default_range_is_inert_and_draws_nothing(self):
        default = constant("BASE_COM_RANGE")
        rng = torch.get_rng_state()
        for _ in range(3):
            coms = self.call(None, dict(default))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(self.asset.root_physx_view.gets, 0)
        self.assertEqual(self.asset.root_physx_view.sets, [])
        self.assertTrue(torch.equal(coms, self.asset.nominal))

    def test_widened_range_draws_inside_the_range_per_env(self):
        torch.manual_seed(0)
        coms = self.call(torch.arange(self.NUM_ENVS), self.WIDE)
        offset = self.offset(coms)
        for axis in (0, 1):
            self.assertTrue(bool(torch.all(offset[:, axis].abs() <= 0.02 + 1e-7)))
            self.assertGreater(float(offset[:, axis].std()), 0.005)
        self.assertNotEqual(offset[:, 0].tolist(), offset[:, 1].tolist())
        self.assertTrue(torch.allclose(offset[:, 2], torch.zeros(self.NUM_ENVS), atol=1e-7))
        self.assertTrue(torch.equal(coms[:, self.BASE, 3:], self.asset.nominal[:, self.BASE, 3:]))
        others = [b for b in range(5) if b != self.BASE]
        self.assertTrue(torch.equal(coms[:, others], self.asset.nominal[:, others]))

    def test_repeated_calls_never_accumulate(self):
        torch.manual_seed(1)
        seen = []
        for _ in range(200):
            coms = self.call(None, self.WIDE)
            offset = self.offset(coms)
            self.assertTrue(bool(torch.all(offset[:, :2].abs() <= 0.02 + 1e-7)))
            seen.append(offset[:, 0].clone())
        spread = torch.stack(seen)
        self.assertGreater(float(spread.max()), 0.019)
        self.assertLess(float(spread.min()), -0.019)
        self.assertLess(abs(float(spread.mean())), 0.002)

    def test_env_ids_subset_only_touches_those_envs(self):
        torch.manual_seed(2)
        ids = torch.tensor([3, 7, 40])
        coms = self.call(ids, self.WIDE)
        self.assertEqual(self.asset.root_physx_view.sets[-1].tolist(), [3, 7, 40])
        untouched = [i for i in range(self.NUM_ENVS) if i not in (3, 7, 40)]
        self.assertTrue(torch.equal(coms[untouched], self.asset.nominal[untouched]))
        self.assertTrue(bool(torch.all(self.offset(coms)[ids, :2].abs() > 0.0)))
        before = coms.clone()
        coms = self.call(torch.tensor([7]), self.WIDE)
        changed = [i for i in range(self.NUM_ENVS) if not torch.equal(coms[i], before[i])]
        self.assertEqual(changed, [7])
        self.assertTrue(bool(torch.all(self.offset(coms)[7, :2].abs() <= 0.02 + 1e-7)))

    def test_nominal_is_snapshotted_once_and_zero_range_restores_it(self):
        self.call(None, self.WIDE)
        snapshot = self.term.nominal_coms.clone()
        self.call(None, self.WIDE)
        self.assertTrue(torch.equal(self.term.nominal_coms, snapshot))
        self.assertTrue(torch.equal(snapshot, self.asset.nominal))
        coms = self.call(None, {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)})
        self.assertTrue(torch.equal(coms, self.asset.nominal))

    def test_fixed_offset_without_width(self):
        coms = self.call(None, {"x": (0.01, 0.01), "y": (0.0, 0.0), "z": (0.0, 0.0)})
        self.assertTrue(torch.allclose(self.offset(coms)[:, 0], torch.full((self.NUM_ENVS,), 0.01), atol=1e-7))
        coms = self.call(None, {"x": (0.01, 0.01), "y": (0.0, 0.0), "z": (0.0, 0.0)})
        self.assertTrue(torch.allclose(self.offset(coms)[:, 0], torch.full((self.NUM_ENVS,), 0.01), atol=1e-7))

    def test_slice_body_ids_and_bad_range(self):
        self.asset_cfg.body_ids = slice(None)
        torch.manual_seed(3)
        coms = self.call(None, self.WIDE)
        delta = coms[..., :3] - self.asset.nominal[..., :3]
        self.assertTrue(torch.allclose(delta, delta[:, :1].expand_as(delta)))
        with self.assertRaises(ValueError):
            self.call(None, {"x": (0.02, -0.02)})


class WiringTest(unittest.TestCase):
    def test_range_constant_is_zero_width_on_every_axis(self):
        self.assertEqual(constant("BASE_COM_RANGE"), {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)})

    def test_event_term_is_the_offset_class_on_base_link(self):
        term = event_term("randomize_base_com")
        self.assertEqual(ast.unparse(term["func"]), "mdp.randomize_rigid_body_com_offset")
        self.assertEqual(ast.literal_eval(term["mode"]), "reset")
        params = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(term["params"].keys, term["params"].values)}
        self.assertEqual(set(params), {"asset_cfg", "com_range"})
        self.assertEqual(params["asset_cfg"], "SceneEntityCfg('robot', body_names=['base_link'])")
        self.assertEqual(params["com_range"], "dict(BASE_COM_RANGE)")
        self.assertNotIn("randomize_rigid_body_com,", CFG.read_text())

    def test_mass_event_is_unchanged(self):
        term = event_term("randomize_mass")
        self.assertEqual(ast.unparse(term["func"]), "mdp.randomize_rigid_body_mass")
        params = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(term["params"].keys, term["params"].values)}
        self.assertEqual(params["mass_distribution_params"], "(-0.3, 0.3)")


@unittest.skipUnless(ISAAC_DICT.exists(), "Isaac Lab source not present")
class HydraOverrideTest(unittest.TestCase):
    def update_class_from_dict(self):
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
        term = types.SimpleNamespace(func="f", mode="reset", params={
            "asset_cfg": types.SimpleNamespace(name="robot"),
            "com_range": dict(constant("BASE_COM_RANGE"))})
        return types.SimpleNamespace(events=types.SimpleNamespace(randomize_base_com=term))

    def test_launch_overrides_widen_x_and_y_as_tuples(self):
        update = self.update_class_from_dict()
        env = self.env()
        for text in ("env.events.randomize_base_com.params.com_range.x=[-0.02,0.02]",
                     "env.events.randomize_base_com.params.com_range.y=[-0.02,0.02]"):
            update(env, self.parse(text))
        com_range = env.events.randomize_base_com.params["com_range"]
        self.assertEqual(com_range, {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)})
        self.assertIsInstance(com_range["x"], tuple)

    def test_an_absent_term_cannot_be_added_by_override(self):
        update = self.update_class_from_dict()
        with self.assertRaises(KeyError):
            update(self.env(), self.parse("env.events.randomize_other_com.params.com_range.x=[-0.02,0.02]"))


if __name__ == "__main__":
    unittest.main()
