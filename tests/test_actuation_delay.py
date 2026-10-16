import math
import sys
import types
import unittest
from collections.abc import Sequence
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sent_tracking_excess import ACTIONS, FakeAsset, FakeJointPositionAction, load_nodes

JOINT_IDS = [15, 2, 9, 0, 11, 4, 13, 6, 1, 8, 3, 10]
DECIMATION = 8


def make_term(delay, num_envs=16, default=None):
    ns = {"torch": torch, "JointPositionAction": FakeJointPositionAction, "ManagerBasedEnv": object,
          "Sequence": Sequence, "SlewLimitedJointPositionActionCfg": object, "Ver2JointPositionActionCfg": object}
    load_nodes(ACTIONS, {"slew_limit_step", "foot_roll", "clip_foot_roll",
                         "SlewLimitedJointPositionAction", "Ver2JointPositionAction"}, ns)
    asset = FakeAsset(num_envs, 16)
    if default is not None:
        asset.data.default_joint_pos[:] = default
    env = types.SimpleNamespace(scene={"robot": asset}, joint_ids=JOINT_IDS, num_envs=num_envs, step_dt=0.02)
    cfg = types.SimpleNamespace(
        max_speed=6.0, max_accel=120.0, slew_enabled=True, foot_roll_limit=math.radians(12.0),
        foot_roll_coeffs=(-0.001195, 0.498159, 0.491024, -0.105182, -0.031960, 0.006356),
        foot_roll_pairs=(("l_ankle_upper_joint", "l_ankle_lower_joint", 1.0),
                         ("r_ankle_upper_joint", "r_ankle_lower_joint", -1.0)),
        actuation_delay_substeps=delay,
    )
    return ns["Ver2JointPositionAction"](cfg, env), asset


def run(term, asset, actions):
    sent = []
    for a in actions:
        term.process_actions(a)
        step = []
        for _ in range(DECIMATION):
            term.apply_actions()
            step.append(asset.data.joint_pos_target[:, JOINT_IDS].clone())
        sent.append((term.processed_actions.clone(), torch.stack(step, dim=1)))
    return sent


def random_actions(num_envs, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [4.0 * torch.randn(num_envs, 12, generator=g) for _ in range(steps)]


class ActuationDelayTest(unittest.TestCase):
    def test_none_is_bit_identical_to_the_current_term(self):
        actions = random_actions(16, 20)
        torch.manual_seed(0)
        off, asset_off = make_term(None)
        torch.manual_seed(0)
        zero, asset_zero = make_term((0, 0))
        for (p0, s0), (p1, s1) in zip(run(off, asset_off, actions), run(zero, asset_zero, actions)):
            torch.testing.assert_close(p0, p1, rtol=0, atol=0)
            torch.testing.assert_close(s0, s1, rtol=0, atol=0)
            torch.testing.assert_close(s0, p0.unsqueeze(1).expand_as(s0), rtol=0, atol=0)

    def test_previous_post_slew_target_is_held_for_k_substeps(self):
        torch.manual_seed(1)
        term, asset = make_term((0, 8), num_envs=64)
        k = term._delay_substeps.clone()
        self.assertTrue(bool((k >= 0).all() and (k <= 8).all()))
        self.assertGreater(len(set(k.tolist())), 4)
        out = run(term, asset, random_actions(64, 10, seed=2))
        previous = asset.data.default_joint_pos[:, JOINT_IDS].clone()
        for processed, sent in out:
            for sub in range(DECIMATION):
                expected = torch.where((sub < k).unsqueeze(1), previous, processed)
                torch.testing.assert_close(sent[:, sub], expected, rtol=0, atol=0)
            previous = processed

    def test_slew_and_roll_path_unchanged_by_the_delay(self):
        actions = random_actions(16, 15, seed=4)
        torch.manual_seed(0)
        off, asset_off = make_term(None)
        torch.manual_seed(0)
        on, asset_on = make_term((3, 8))
        for (p0, _), (p1, _) in zip(run(off, asset_off, actions), run(on, asset_on, actions)):
            torch.testing.assert_close(p0, p1, rtol=0, atol=0)
        torch.testing.assert_close(off._slew_position, on._slew_position, rtol=0, atol=0)
        torch.testing.assert_close(off.slew_lag, on.slew_lag, rtol=0, atol=0)

    def test_reset_seeds_the_hold_with_the_default_pose_and_redraws(self):
        default = torch.linspace(-0.3, 0.3, 16)
        torch.manual_seed(5)
        term, asset = make_term((8, 8), num_envs=4, default=default)
        run(term, asset, random_actions(4, 3, seed=6))
        term.reset(torch.tensor([1, 3]))
        actions = random_actions(4, 1, seed=7)
        before = term._delay_last.clone()
        (processed, sent), = run(term, asset, actions)
        for sub in range(DECIMATION):
            torch.testing.assert_close(sent[[1, 3], sub], default[JOINT_IDS].expand(2, 12), rtol=0, atol=0)
            torch.testing.assert_close(sent[[0, 2], sub], before[[0, 2]], rtol=0, atol=0)
        self.assertFalse(torch.equal(processed[[0, 2]], before[[0, 2]]))

    def test_full_step_delay_sends_only_the_previous_target(self):
        torch.manual_seed(0)
        term, asset = make_term((8, 8), num_envs=8)
        out = run(term, asset, random_actions(8, 4, seed=8))
        for (prev, _), (_, sent) in zip(out, out[1:]):
            torch.testing.assert_close(sent, prev.unsqueeze(1).expand_as(sent), rtol=0, atol=0)

    def test_range_is_validated(self):
        for bad in ((-1, 2), (4, 2)):
            with self.assertRaises(ValueError):
                make_term(bad)


if __name__ == "__main__":
    unittest.main()
