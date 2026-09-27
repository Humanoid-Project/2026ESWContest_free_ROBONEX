import ast
import math
import unittest
from pathlib import Path

import numpy as np
import torch

try:
    from robonex_common.foot_roll import clip_foot_roll as deploy_clip
    from robonex_common.models import VER2_EDU
except ImportError:
    deploy_clip = None

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"


def load_functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[name] for name in names]


def load_contract_constants():
    namespace = {"math": math}
    tree = ast.parse((TASK / "robot_contract_v2.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").startswith("FOOT_ROLL")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "robot_contract_v2.py", "exec"), namespace)
    return namespace["FOOT_ROLL_COEFFS"], namespace["FOOT_ROLL_LIMIT_RAD"], namespace["FOOT_ROLL_PAIRS"]


@unittest.skipIf(deploy_clip is None, "robonex-common without the Ver.2 profile (< 0.6.0)")
class DeployRollParityTest(unittest.TestCase):
    def setUp(self):
        (self.train_clip,) = load_functions(TASK / "mdp/actions.py", ["foot_roll", "clip_foot_roll"], {"torch": torch})[1:]
        self.coeffs, self.limit, self.pairs = load_contract_constants()
        margin = 0.01
        self.upper_range = (math.radians(-16.0) + margin, math.radians(50.0) - margin)
        self.lower_range = (math.radians(-50.0) + margin, math.radians(30.0) - margin)

    def test_common_profile_carries_the_training_constants(self):
        roll = VER2_EDU.foot_roll
        self.assertEqual(tuple(roll.coeffs), tuple(self.coeffs))
        self.assertEqual(roll.limit, self.limit)
        self.assertEqual(tuple(tuple(p) for p in roll.pairs), tuple(tuple(p) for p in self.pairs))

    def test_deploy_clip_matches_training_clip(self):
        generator = torch.Generator().manual_seed(0)
        u = torch.empty(20000, dtype=torch.float64).uniform_(-0.6, 1.1, generator=generator)
        l = torch.empty(20000, dtype=torch.float64).uniform_(-1.1, 0.8, generator=generator)
        u = torch.clamp(u, *self.upper_range)
        l = torch.clamp(l, *self.lower_range)
        tu, tl = self.train_clip(u, l, self.upper_range, self.lower_range, self.coeffs, self.limit)
        du, dl = deploy_clip(u.numpy(), l.numpy(), self.upper_range, self.lower_range, self.coeffs, self.limit)
        np.testing.assert_allclose(du, tu.numpy(), atol=1e-12)
        np.testing.assert_allclose(dl, tl.numpy(), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
