from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn
from torch.distributions import Normal

import rsl_rl.runners.on_policy_runner as _on_policy_runner
from rsl_rl.algorithms import PPO

__all__ = [
    "RSL_RL_VERSION",
    "SmoothPPO",
    "closed_loop_transfer",
    "hf_closed_loop_penalty",
    "hf_directions",
    "hf_gain_penalty",
    "hf_phasors",
    "input_jacobian",
    "resolve_action_scales",
    "resolve_obs_term_slice",
]

RSL_RL_VERSION = "3.1.2"


def installed_rsl_rl_version() -> str:
    return importlib.metadata.version("rsl-rl-lib")


def check_rsl_rl_version() -> None:
    found = installed_rsl_rl_version()
    if found != RSL_RL_VERSION:
        raise RuntimeError(
            f"SmoothPPO.update is a copy of rsl_rl {RSL_RL_VERSION} PPO.update, but rsl-rl-lib {found} is installed; "
            "re-diff rsl_rl/algorithms/ppo.py and update the copy before training"
        )


def hf_directions(freqs: Sequence[float], step_dt: float, history: int, axes: int) -> torch.Tensor:
    rows = []
    for freq in freqs:
        omega = 2.0 * math.pi * float(freq) * step_dt
        for trig in (math.cos, math.sin):
            weights = [trig(omega * (history - 1 - k)) for k in range(history)]
            if max(abs(w) for w in weights) < 1e-9:
                continue
            for axis in range(axes):
                row = torch.zeros(history * axes, dtype=torch.float32)
                for k, w in enumerate(weights):
                    row[axes * k + axis] = w
                rows.append(row)
    return torch.stack(rows)


def hf_gain_penalty(mean_fn, obs: torch.Tensor, lo: int, hi: int, dirs: torch.Tensor, eps: float,
                    scales: torch.Tensor) -> torch.Tensor:
    num, count = obs.shape[0], dirs.shape[0]
    pert = obs.new_zeros(count, obs.shape[1])
    pert[:, lo:hi] = eps * dirs.to(obs)
    rows = torch.cat(((obs[:, None] + pert).flatten(0, 1), (obs[:, None] - pert).flatten(0, 1)))
    plus, minus = mean_fn(rows).view(2, num, count, -1)
    gain = (plus - minus) / (2.0 * eps) * scales.to(plus)
    return gain.square().sum((1, 2)).mean()


def hf_phasors(freqs: Sequence[float], step_dt: float, history: int, lag: int) -> torch.Tensor:
    ages = torch.arange(history, dtype=torch.float64).flip(0) + lag
    omega = 2.0 * math.pi * torch.tensor([float(f) for f in freqs], dtype=torch.float64) * step_dt
    return torch.polar(torch.ones(len(freqs), history, dtype=torch.float64), -omega[:, None] * ages[None])


def input_jacobian(mean_fn, obs: torch.Tensor) -> torch.Tensor:
    return torch.func.vmap(torch.func.jacrev(lambda row: mean_fn(row[None])[0]))(obs)


def closed_loop_transfer(jac: torch.Tensor, gyro: tuple[int, int], action: tuple[int, int], history: int,
                         freqs: Sequence[float], step_dt: float, damping: float) -> torch.Tensor:
    ctype = torch.complex128 if jac.dtype == torch.float64 else torch.complex64
    jg = jac[..., gyro[0]:gyro[1]].unflatten(-1, (history, -1)).to(ctype)
    ja = jac[..., action[0]:action[1]].unflatten(-1, (history, -1)).to(ctype)
    if ja.shape[-1] != jac.shape[-2]:
        raise RuntimeError(f"action history has {ja.shape[-1]} values per frame for {jac.shape[-2]} actions")
    b = torch.einsum("njkc,fk->fnjc", jg, hf_phasors(freqs, step_dt, history, 0).to(ctype).to(jac.device))
    a = torch.einsum("njkc,fk->fnjc", ja, hf_phasors(freqs, step_dt, history, 1).to(ctype).to(jac.device))
    eye = torch.eye(a.shape[-1], dtype=ctype, device=jac.device)
    m = eye - a
    if damping > 0.0:
        mh = m.mH
        return torch.linalg.solve(mh @ m + damping**2 * eye, mh @ b)
    return torch.linalg.solve(m, b)


def hf_closed_loop_penalty(mean_fn, obs: torch.Tensor, gyro: tuple[int, int], action: tuple[int, int], history: int,
                           freqs: Sequence[float], step_dt: float, scales: torch.Tensor,
                           damping: float) -> torch.Tensor:
    transfer = closed_loop_transfer(input_jacobian(mean_fn, obs), gyro, action, history, freqs, step_dt, damping)
    gain = transfer * scales.to(obs)[:, None]
    return (gain.real.square() + gain.imag.square()).sum((0, 2, 3)).mean()


def resolve_obs_term_slice(manager, groups: Sequence[str], term: str) -> tuple[int, int, int, int]:
    found = None
    offset = 0
    for group in groups:
        names = list(manager.active_terms[group])
        dims = [tuple(int(v) for v in d) for d in manager.group_obs_term_dim[group]]
        if any(len(d) != 1 for d in dims):
            raise RuntimeError(f"group {group!r} has unflattened term shapes {dims}; the actor input layout is unknown")
        widths = [d[0] for d in dims]
        width = int(manager.group_obs_dim[group][0])
        if sum(widths) != width:
            raise RuntimeError(f"group {group!r} terms cover {sum(widths)} of {width} values")
        if term in names:
            if found is not None:
                raise RuntimeError(f"term {term!r} appears in more than one actor group")
            index = names.index(term)
            cfg = manager._group_obs_term_cfgs[group][index]
            history = int(cfg.history_length or 1)
            if cfg.history_length and not cfg.flatten_history_dim:
                raise RuntimeError(f"term {term!r} keeps an unflattened history dimension")
            if widths[index] % history:
                raise RuntimeError(f"term {term!r} width {widths[index]} is not a multiple of history {history}")
            start = offset + sum(widths[:index])
            found = (start, start + widths[index], history)
        offset += width
    if found is None:
        raise RuntimeError(f"term {term!r} is not in the actor observation groups {list(groups)}")
    return found + (offset,)


def resolve_action_scales(action_manager, term: str, expected=None) -> torch.Tensor:
    action_term = action_manager.get_term(term)
    names = list(action_term._joint_names)
    if int(action_manager.total_action_dim) != len(names):
        raise RuntimeError(
            f"action vector has {action_manager.total_action_dim} values but term {term!r} drives {len(names)} joints"
        )
    live = torch.as_tensor(action_term._scale, dtype=torch.float32).detach().cpu()
    if live.dim() == 0:
        live = live.expand(len(names))
    elif live.dim() > 1:
        if (live - live[:1]).abs().max().item() > 0.0:
            raise RuntimeError(f"action term {term!r} has per-environment scales")
        live = live[0]
    live = live.reshape(-1).clone()
    if expected is None:
        return live
    if isinstance(expected, Mapping):
        if set(expected) != set(names):
            raise RuntimeError(f"action_scales names {sorted(expected)} differ from the action joints {sorted(names)}")
        wanted = torch.tensor([float(expected[name]) for name in names], dtype=torch.float32)
    else:
        wanted = torch.tensor([float(v) for v in expected], dtype=torch.float32)
        if wanted.numel() != len(names):
            raise RuntimeError(f"action_scales has {wanted.numel()} values for {len(names)} action joints")
    residual = (wanted - live).abs().max().item()
    if residual > 1e-6:
        raise RuntimeError(f"action_scales differ from the live action term scales (max diff {residual:.3e})")
    return wanted


class SmoothPPO(PPO):
    def __init__(
        self,
        policy,
        hf_gyro_coef: float = 0.0,
        lcp_coef: float = 0.0,
        action_scales=None,
        hf_freqs: Sequence[float] = (12.5, 18.75, 25.0),
        step_dt: float = 0.02,
        hf_eps: float = 0.05,
        hf_samples: int = 2048,
        hf_closed_loop: bool = False,
        hf_damping: float = 0.05,
        gyro_term: str = "imu_ang_vel",
        action_term: str = "joint_pos",
        last_action_term: str = "actions",
        **kwargs,
    ) -> None:
        check_rsl_rl_version()
        super().__init__(policy, **kwargs)
        self.hf_gyro_coef = float(hf_gyro_coef)
        self.lcp_coef = float(lcp_coef)
        self.action_scales = action_scales
        self.hf_freqs = tuple(float(f) for f in hf_freqs)
        self.step_dt = float(step_dt)
        self.hf_eps = float(hf_eps)
        self.hf_samples = int(hf_samples)
        self.hf_closed_loop = bool(hf_closed_loop)
        self.hf_damping = float(hf_damping)
        self.gyro_term = gyro_term
        self.action_term = action_term
        self.last_action_term = last_action_term
        self.hf_slice = None
        self.hf_action_slice = None
        self.hf_history = None
        self.hf_dirs = None
        self.hf_scales = None
        self.hf_generator = None
        if self.hf_gyro_coef < 0.0 or self.lcp_coef < 0.0 or self.hf_damping < 0.0:
            raise ValueError("hf_gyro_coef, lcp_coef and hf_damping must be non-negative")
        if (self.hf_gyro_coef > 0.0 or self.lcp_coef > 0.0) and self.policy.is_recurrent:
            raise ValueError("SmoothPPO penalties support feed-forward policies only")
        if self.hf_gyro_coef > 0.0:
            env = self.symmetry.get("_env") if self.symmetry else None
            if env is None:
                raise ValueError("hf_gyro_coef > 0 needs the environment (symmetry_cfg) to resolve the gyro layout")
            self.configure_hf_gyro(env)

    def _actor_in_features(self) -> int:
        return next(m for m in self.policy.actor.modules() if isinstance(m, nn.Linear)).in_features

    def configure_hf_gyro(self, env) -> None:
        unwrapped = env.unwrapped
        groups = self.policy.obs_groups["policy"]
        lo, hi, history, width = resolve_obs_term_slice(unwrapped.observation_manager, groups, self.gyro_term)
        if width != self._actor_in_features():
            raise RuntimeError(f"actor input is {self._actor_in_features()} wide, observation groups give {width}")
        if abs(float(unwrapped.step_dt) - self.step_dt) > 1e-9:
            raise RuntimeError(f"step_dt {self.step_dt} differs from the environment step {unwrapped.step_dt}")
        axes = (hi - lo) // history
        if axes != 3:
            raise RuntimeError(f"term {self.gyro_term!r} has {axes} values per frame, expected 3")
        scales = resolve_action_scales(unwrapped.action_manager, self.action_term, self.action_scales)
        with torch.no_grad():
            num_actions = self._actor_mean_normalized(torch.zeros(1, width, device=self.device)).shape[-1]
        if scales.numel() != num_actions:
            raise RuntimeError(f"{scales.numel()} action scales for {num_actions} actions")
        form = "open loop"
        if self.hf_closed_loop:
            a_lo, a_hi, a_history, a_width = resolve_obs_term_slice(
                unwrapped.observation_manager, groups, self.last_action_term
            )
            if a_width != width or a_history != history:
                raise RuntimeError(
                    f"term {self.last_action_term!r} has history {a_history} in a {a_width}-wide input, "
                    f"expected history {history} in {width}"
                )
            if (a_hi - a_lo) != history * num_actions:
                raise RuntimeError(
                    f"term {self.last_action_term!r} has {(a_hi - a_lo) // history} values per frame, "
                    f"expected {num_actions}"
                )
            self.hf_action_slice = (a_lo, a_hi)
            form = f"closed loop through '{self.last_action_term}' at {a_lo}:{a_hi} (damping {self.hf_damping})"
        self.hf_slice = (lo, hi)
        self.hf_history = history
        self.hf_dirs = hf_directions(self.hf_freqs, self.step_dt, history, axes).to(self.device)
        self.hf_scales = scales.to(self.device)
        self.hf_generator = torch.Generator(device=self.device)
        self.hf_generator.manual_seed(torch.initial_seed())
        print(
            f"[SmoothPPO] hf_gyro coef {self.hf_gyro_coef}: '{self.gyro_term}' at actor inputs {lo}:{hi} "
            f"({history} frames x {axes}, oldest first), {self.hf_dirs.shape[0]} directions at {self.hf_freqs} Hz, "
            f"{form}, scales {[round(v, 6) for v in self.hf_scales.tolist()]}",
            flush=True,
        )

    def _actor_mean_normalized(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.policy.actor(obs)
        if self.policy.state_dependent_std:
            return out[..., 0, :]
        return out

    def actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self._actor_mean_normalized(self.policy.actor_obs_normalizer(obs))

    def hf_gyro_penalty_on(self, obs: torch.Tensor) -> torch.Tensor:
        if self.hf_closed_loop:
            return hf_closed_loop_penalty(self.actor_mean, obs, self.hf_slice, self.hf_action_slice, self.hf_history,
                                          self.hf_freqs, self.step_dt, self.hf_scales, self.hf_damping)
        lo, hi = self.hf_slice
        return hf_gain_penalty(self.actor_mean, obs, lo, hi, self.hf_dirs, self.hf_eps, self.hf_scales)

    def _hf_gyro_penalty(self, obs_batch, original_batch_size: int, num_aug: int) -> torch.Tensor:
        obs = self.policy.get_actor_obs(obs_batch).detach()
        count = min(original_batch_size, max(1, self.hf_samples // num_aug))
        pick = torch.randperm(original_batch_size, generator=self.hf_generator, device=self.hf_generator.device)
        pick = pick[:count].to(obs.device)
        rows = torch.cat([pick + a * original_batch_size for a in range(num_aug)])
        return self.hf_gyro_penalty_on(obs[rows])

    def _lcp_penalty(self, obs_batch, actions_batch: torch.Tensor) -> torch.Tensor:
        obs = self.policy.actor_obs_normalizer(self.policy.get_actor_obs(obs_batch)).detach().requires_grad_(True)
        mean = self._actor_mean_normalized(obs)
        sigma = self.policy.action_std.detach()
        log_prob = Normal(mean, sigma).log_prob(actions_batch).sum(dim=-1)
        grad = torch.autograd.grad(log_prob.sum(), obs, create_graph=True)[0]
        return grad.square().sum(dim=-1).mean()

    def update(self) -> dict[str, float]:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None
        mean_hf_gyro = 0 if self.hf_gyro_coef > 0.0 else None
        mean_lcp = 0 if self.lcp_coef > 0.0 else None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            penalty_obs_batch, penalty_actions_batch, penalty_num_aug = obs_batch, actions_batch, num_aug

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if mean_hf_gyro is not None:
                hf_gyro_loss = self._hf_gyro_penalty(penalty_obs_batch, original_batch_size, penalty_num_aug)
                loss = loss + self.hf_gyro_coef * hf_gyro_loss

            if mean_lcp is not None:
                lcp_loss = self._lcp_penalty(penalty_obs_batch, penalty_actions_batch)
                loss = loss + self.lcp_coef * lcp_loss

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if mean_hf_gyro is not None:
                mean_hf_gyro += hf_gyro_loss.item()
            if mean_lcp is not None:
                mean_lcp += lcp_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_hf_gyro is not None:
            mean_hf_gyro /= num_updates
        if mean_lcp is not None:
            mean_lcp /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if mean_hf_gyro is not None:
            loss_dict["hf_gyro"] = mean_hf_gyro
        if mean_lcp is not None:
            loss_dict["lcp"] = mean_lcp

        return loss_dict


_on_policy_runner.SmoothPPO = SmoothPPO
