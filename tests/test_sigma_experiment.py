import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class BaseWrapper:
    def __init__(self, env, clip_actions=None):
        self.unwrapped = env
        self.device = "cpu"
        self.num_envs = 2
        self.num_actions = 12
        self.clip_actions = clip_actions

    def step(self, actions):
        return actions.clamp(-self.clip_actions, self.clip_actions) if self.clip_actions else actions


class BaseRunner:
    def log(self, locs, width=80, pad=35):
        pass


class SigmaExperimentTests(unittest.TestCase):
    def setUp(self):
        self.rewards = load_definitions(
            TASK / "mdp/rewards.py", {"_bounded_square", "_saturate", "action_rate_l2_bounded"},
            {"torch": torch, "ManagerBasedRLEnv": object},
        )
        self.diagnostics = load_definitions(
            ROOT / "scripts/rsl_rl/train.py",
            {"DiagnosticVecEnvWrapper", "DiagnosticOnPolicyRunner"},
            {"torch": torch, "math": math, "RslRlVecEnvWrapper": BaseWrapper,
             "OnPolicyRunner": BaseRunner},
        )
        self.names = tuple(f"joint_{i}" for i in reversed(range(12)))
        self.scales = torch.linspace(0.02, 0.12, 12)
        offsets = torch.zeros(12)
        clip = torch.stack((torch.full((12,), -1.0), torch.full((12,), 1.0)), dim=-1)
        term = SimpleNamespace(
            _joint_names=self.names,
            _scale=self.scales.repeat(2, 1),
            _offset=offsets.repeat(2, 1),
            _clip=clip.unsqueeze(0).repeat(2, 1, 1),
        )
        env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _: term))
        self.wrapper = self.diagnostics["DiagnosticVecEnvWrapper"](env, clip_actions=14.0)

    def _rate_env(self, action, prev_action):
        term = SimpleNamespace(_joint_names=self.names, _scale=self.scales.unsqueeze(0))
        return SimpleNamespace(
            device="cpu",
            action_manager=SimpleNamespace(
                action=action, prev_action=prev_action, get_term=lambda _: term
            ),
        )

    def test_cost_is_measured_in_joint_target_radians(self):
        delta = torch.eye(12)
        scale = 0.5
        actual = self.rewards["action_rate_l2_bounded"](self._rate_env(delta, 0))
        expected = 1.0 - torch.exp(-self.scales.square() / scale)
        torch.testing.assert_close(actual, expected)

    def test_equal_physical_delta_has_equal_cost(self):
        # one radian of target change on each joint costs the same regardless of scale
        delta = torch.eye(12) / self.scales
        actual = self.rewards["action_rate_l2_bounded"](self._rate_env(delta, 0))
        torch.testing.assert_close(actual, torch.full((12,), 1.0 - torch.exp(torch.tensor(-1.0 / 0.5)).item()))

    def test_physical_delta_is_clamped_at_the_cap(self):
        # one joint saturates the 1.0 rad cap, so the raw cost is exactly 1.0
        # and the kernel maps it to 1 - exp(-1/scale), still short of 1.0
        delta = torch.eye(12) * 1000.0
        actual = self.rewards["action_rate_l2_bounded"](self._rate_env(delta, 0))
        expected = 1.0 - torch.exp(torch.tensor(-1.0 / 0.5))
        torch.testing.assert_close(actual, expected.expand(12))
        self.assertTrue(bool((actual < 1.0).all()))

    def test_nonfinite_guard(self):
        actual = self.rewards["_bounded_square"](torch.tensor([math.nan, math.inf, -math.inf]), 30)
        torch.testing.assert_close(actual, torch.full((3,), 900.0))

    def test_clip_logging_does_not_change_actions(self):
        actions = torch.zeros(2, 12)
        actions[0, 0], actions[1, 1] = 15.0, -14.0
        original = actions.clone()
        result = self.wrapper.step(actions)
        torch.testing.assert_close(actions, original)
        torch.testing.assert_close(result, original.clamp(-14, 14))
        self.wrapper.step(torch.zeros_like(actions))
        metrics = self.wrapper.take_action_diagnostics()
        self.assertEqual(metrics[f"Policy/runner_clip_fraction/{self.names[0]}"], 0.25)
        self.assertEqual(metrics["Policy/runner_clip_any_fraction"], 0.5)
        self.assertAlmostEqual(metrics["Policy/runner_clip_element_fraction"], 2 / 48)
        self.assertEqual(self.wrapper.take_action_diagnostics()["Policy/runner_clip_any_fraction"], 0)

    def test_per_joint_std_and_units(self):
        runner = self.diagnostics["DiagnosticOnPolicyRunner"]()
        runner.env = self.wrapper
        runner.writer = SimpleNamespace(add_scalar=Mock())
        runner.alg = SimpleNamespace(policy=SimpleNamespace(noise_std_type="log", log_std=torch.zeros(12)))
        runner.log({"it": 123})
        metrics = {call.args[0]: call.args[1] for call in runner.writer.add_scalar.call_args_list}
        for index, name in enumerate(self.names):
            self.assertEqual(metrics[f"Policy/std/{name}"], 1)
            self.assertAlmostEqual(metrics[f"Policy/preclip_target_std_deg/{name}"], self.scales[index].item() * 180 / math.pi, places=5)
        self.assertEqual(metrics["Policy/std_gate_pass"], 1)
        self.assertTrue(all(call.args[2] == 123 for call in runner.writer.add_scalar.call_args_list))
        runner.alg.policy.log_std[0] = math.log(1000)
        runner.log({"it": 124})
        metrics = {call.args[0]: call.args[1] for call in runner.writer.add_scalar.call_args_list}
        self.assertEqual(metrics["Policy/std_gate_pass"], 0)

    def test_experiment_configuration(self):
        namespace = load_definitions(
            TASK / "agents/rsl_rl_ppo_cfg.py", {"PPORunnerCfg"},
            {"configclass": lambda cls: cls, "RslRlOnPolicyRunnerCfg": object,
             "RUNNER_ACTION_CLIP": 14.0, "RslRlPpoActorCriticCfg": SimpleNamespace,
             "RslRlPpoAlgorithmCfg": SimpleNamespace, "RslRlSymmetryCfg": SimpleNamespace,
             "symmetry": SimpleNamespace(compute_symmetric_states=lambda **kw: None)},
        )
        cfg = namespace["PPORunnerCfg"]
        self.assertEqual((cfg.policy.init_noise_std, cfg.policy.noise_std_type), (1.0, "log"))
        self.assertEqual((cfg.algorithm.entropy_coef, cfg.clip_actions), (0.008, 14))
        self.assertTrue(cfg.algorithm.symmetry_cfg.use_data_augmentation)
        self.assertFalse(cfg.algorithm.symmetry_cfg.use_mirror_loss)
        tree = ast.parse((TASK / "robonex_walking_env_cfg.py").read_text())
        rewards = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")
        terms = {node.targets[0].id: node.value for node in rewards.body if isinstance(node, ast.Assign)}
        weight = next(item.value for item in terms["action_rate"].keywords if item.arg == "weight")
        self.assertEqual(ast.literal_eval(weight), -0.2)
        weight = next(item.value for item in terms["joint_pos_limits"].keywords if item.arg == "weight")
        self.assertEqual(ast.literal_eval(weight), -5.0)
        weight = next(item.value for item in terms["feet_clearance"].keywords if item.arg == "weight")
        self.assertEqual(ast.literal_eval(weight), -0.5)
        weight = next(item.value for item in terms["feet_gait"].keywords if item.arg == "weight")
        self.assertEqual(ast.literal_eval(weight), 1.0)


if __name__ == "__main__":
    unittest.main()
