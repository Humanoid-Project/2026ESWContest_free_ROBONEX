import ast
import importlib.util
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "source/robonex_walking/robonex_walking/tasks/manager_based/robonex_walking/robonex_walking_v2_env_cfg.py"
EFFECTS = ROOT / "scripts/rsl_rl/deploy_effects.py"


def load_effects():
    spec = importlib.util.spec_from_file_location("deploy_effects_under_test", EFFECTS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def obs_terms(group):
    tree = ast.parse(CFG.read_text())
    obs = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObservationsCfg")
    cls = next(n for n in obs.body if isinstance(n, ast.ClassDef) and n.name == group)
    terms = {}
    for node in cls.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            terms[node.targets[0].id] = {k.arg: k.value for k in node.value.keywords}
    return terms


def constant(name):
    tree = ast.parse(CFG.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


class FakeDelayedTerm:
    def __init__(self, delay, max_delay=1):
        self._delay = delay
        self._max_delay = max_delay

    def reset(self, env_ids=None):
        self._delay[:] = torch.randint(0, self._max_delay + 1, self._delay.shape)


def fake_env(terms):
    manager = types.SimpleNamespace(
        _group_obs_term_names={"policy": [name for name, _ in terms]},
        _group_obs_term_cfgs={"policy": [types.SimpleNamespace(func=func) for _, func in terms]},
    )
    return types.SimpleNamespace(observation_manager=manager)


class WiringTest(unittest.TestCase):
    def test_joint_terms_are_delayed_and_the_rest_is_not(self):
        terms = obs_terms("PolicyCfg")
        self.assertEqual(list(terms), ["joint_pos_rel", "joint_vel_rel", "imu_ang_vel", "projected_gravity",
                                       "velocity_commands", "gait_phase", "actions"])
        for name, field in (("joint_pos_rel", "pos"), ("joint_vel_rel", "vel")):
            self.assertEqual(ast.unparse(terms[name]["func"]), "mdp.delayed_joint_state")
            params = {ast.literal_eval(k): v for k, v in zip(terms[name]["params"].keys, terms[name]["params"].values)}
            self.assertEqual(ast.literal_eval(params["field"]), field)
            self.assertEqual(ast.unparse(params["max_delay_steps"]), "JOINT_OBS_MAX_DELAY_STEPS")
            self.assertIn("LEG_JOINTS", ast.unparse(params["asset_cfg"]))
        self.assertEqual(ast.unparse(terms["imu_ang_vel"]["func"]), "mdp.delayed_imu_ang_vel")
        self.assertEqual(ast.unparse(terms["projected_gravity"]["func"]), "mdp.delayed_imu_projected_gravity")
        self.assertEqual(constant("JOINT_OBS_MAX_DELAY_STEPS"), 0)

    def test_critic_keeps_the_undelayed_state(self):
        critic = obs_terms("CriticCfg")
        self.assertIn("base_lin_vel", critic)
        self.assertFalse(any("delayed" in ast.unparse(term["func"]) for term in critic.values()))


class PinTest(unittest.TestCase):
    def test_pin_fixes_the_shared_delay_through_resets(self):
        effects = load_effects()
        shared = torch.tensor([0, 1, 1, 0])
        pos, vel = FakeDelayedTerm(shared), FakeDelayedTerm(shared)
        env = fake_env([("joint_pos_rel", pos), ("joint_vel_rel", vel), ("imu_ang_vel", len)])
        self.assertEqual(effects.pin_joint_obs_delay(env, 0), ["joint_pos_rel", "joint_vel_rel"])
        self.assertEqual(shared.tolist(), [0, 0, 0, 0])
        for _ in range(20):
            pos.reset(torch.arange(2))
            vel.reset()
            self.assertEqual(shared.tolist(), [0, 0, 0, 0])
        effects.pin_joint_obs_delay(env, 1)
        pos.reset()
        self.assertEqual(shared.tolist(), [1, 1, 1, 1])

    def test_pin_rejects_a_delay_the_buffer_does_not_hold(self):
        effects = load_effects()
        env = fake_env([("joint_pos_rel", FakeDelayedTerm(torch.zeros(2, dtype=torch.long)))])
        with self.assertRaises(ValueError):
            effects.pin_joint_obs_delay(env, 2)

    def test_pin_on_an_undelayed_policy_is_an_error(self):
        effects = load_effects()
        with self.assertRaises(KeyError):
            effects.pin_joint_obs_delay(fake_env([("joint_pos_rel", len)]), 0)

    def test_eval_delay_refuses_to_stack_on_the_trained_delay(self):
        effects = load_effects()
        env = fake_env([("joint_pos_rel", FakeDelayedTerm(torch.zeros(2, dtype=torch.long))),
                        ("joint_vel_rel", FakeDelayedTerm(torch.zeros(2, dtype=torch.long)))])
        with self.assertRaises(ValueError):
            effects.install_joint_obs_delay(env, 1)


class RestoreTest(unittest.TestCase):
    def test_an_undelayed_run_restores_without_the_new_params(self):
        effects = load_effects()
        term = types.SimpleNamespace(params={"asset_cfg": object(), "field": "pos", "max_delay_steps": 1})
        effects._drop_untrained_params(term, {"func": "isaaclab.envs.mdp.observations:joint_pos_rel",
                                              "params": {"asset_cfg": {"name": "robot"}}})
        self.assertEqual(list(term.params), ["asset_cfg"])


if __name__ == "__main__":
    unittest.main()
