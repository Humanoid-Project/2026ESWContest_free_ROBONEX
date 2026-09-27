import ast
import json
import math
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"
SCAN = ROOT / "etc/diag/2026-09-27_ver2_ankle_limits/ankle_workspace_l.json"


def load_functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[name] for name in names]


def load_contract_constants():
    namespace = {"math": math}
    tree = ast.parse((TASK / "robot_contract_v2.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").startswith("FOOT_ROLL")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "robot_contract_v2.py", "exec"), namespace)
    return namespace["FOOT_ROLL_COEFFS"], namespace["FOOT_ROLL_LIMIT_RAD"]


class FootRollClipTest(unittest.TestCase):
    def setUp(self):
        self.roll, self.clip = load_functions(TASK / "mdp/actions.py", ["foot_roll", "clip_foot_roll"], {"torch": torch})
        self.coeffs, self.limit = load_contract_constants()
        margin = 0.01
        self.upper_range = (math.radians(-16.0) + margin, math.radians(50.0) - margin)
        self.lower_range = (math.radians(-50.0) + margin, math.radians(30.0) - margin)
        grid_u = torch.linspace(*self.upper_range, 67, dtype=torch.float64)
        grid_l = torch.linspace(*self.lower_range, 81, dtype=torch.float64)
        self.u, self.l = (g.reshape(-1) for g in torch.meshgrid(grid_u, grid_l, indexing="ij"))

    def clipped(self, u, l):
        return self.clip(u, l, self.upper_range, self.lower_range, self.coeffs, self.limit)

    def test_targets_inside_are_unchanged(self):
        inside = self.roll(self.u, self.l, self.coeffs).abs() <= self.limit
        u, l = self.clipped(self.u[inside], self.l[inside])
        self.assertTrue(torch.equal(u, self.u[inside]))
        self.assertTrue(torch.equal(l, self.l[inside]))

    def test_every_box_target_ends_inside_roll_and_box(self):
        u, l = self.clipped(self.u, self.l)
        roll = self.roll(u, l, self.coeffs)
        self.assertLessEqual(float(roll.abs().max()), self.limit + math.radians(0.05))
        self.assertGreaterEqual(float(u.min()), self.upper_range[0])
        self.assertLessEqual(float(u.max()), self.upper_range[1])
        self.assertGreaterEqual(float(l.min()), self.lower_range[0])
        self.assertLessEqual(float(l.max()), self.lower_range[1])

    def test_clipped_targets_are_feasible_in_the_scan(self):
        if not SCAN.exists():
            self.skipTest("ankle workspace scan not found")
        rows = {(r["upper"], r["lower"]): r for r in json.loads(SCAN.read_text())["rows"]}
        u, l = self.clipped(self.u, self.l)
        bad = []
        for a, b in zip(torch.rad2deg(u).tolist(), torch.rad2deg(l).tolist()):
            for ga in (2 * math.floor(a / 2), 2 * math.ceil(a / 2)):
                for gb in (2 * math.floor(b / 2), 2 * math.ceil(b / 2)):
                    row = rows[(float(ga), float(gb))]
                    if not row["ok"] and abs(row["roll"]) <= 14.0:
                        bad.append((ga, gb))
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
