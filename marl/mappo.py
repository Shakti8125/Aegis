"""Hand-rolled MAPPO: centralized critic over the pooled graph embedding, decentralized actors, GAE advantages.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.

No RLlib, on purpose (PLAN.md section 3 Phase 4).  What is here:

* **CTDE.**  One *shared-parameter* actor evaluated on each agent's own local
  observation - the 12 service agents are homogeneous, so sharing weights is
  both correct and a 12x sample multiplier.  One centralized critic that sees
  the global state, which is never available at execution time.
* **GAE(lambda)** with correct terminal-vs-truncation bootstrapping.
* Clipped surrogate objective, clipped value loss, entropy bonus, advantage
  normalisation, multiple minibatch epochs, global grad-norm clipping,
  optional approximate-KL early stop and linear LR annealing.

=====================================================================
THE PHASE 3 SEAM  (read this before wiring the GraphSAGE encoder in)
=====================================================================
The trainer never touches raw observation tensors directly.  Every input to the
actor and the critic goes through an :class:`ObservationEncoder`:

    actor_input  = encoder.actor_features(obs)      # (..., N, actor_dim)
    critic_input = encoder.critic_features(state)   # (..., N, critic_dim)

The default :class:`VectorObsEncoder` is a **pass-through**: it forwards the
simulator's fixed-length per-agent observation vector
(``env.observation_space(agent)``, ``obs_dim = 35 + n_tiers``) to the actor and
the simulator's global state vector (``env.state()``,
``state_dim = 10*n_services + 3*n_nodes + 5``) to the critic.  It holds no
parameters, so with it MAPPO is the no-GNN ablation the simulator docstring
anticipates.

Phase 3's ``encoder/gnn_model.py`` plugs in as a second implementation of the
same two methods.  Its ``AegisGraphEncoder.forward`` already returns exactly the
two heads this seam wants - an ``EncoderOutput`` whose ``agent_observations``
(= ``node_embeddings["Service"]``, one row per Service, in simulator index
order) is the actor input and whose ``global_embedding`` is the pooled critic
input - so the adapter is:

    class GraphEncoderSeam(ObservationEncoder):
        def __init__(self, gnn, n_agents):
            self.gnn = gnn
            self.actor_input_dim  = gnn.hidden_dim
            self.critic_input_dim = gnn.global_dim + n_agents
        # actor_features  -> out.agent_observations
        # critic_features -> out.global_embedding, tiled + agent-id one-hot

plus feeding ``env.graph_snapshot()`` (not the obs vector) into the rollout
buffer.  Nothing else moves:

* :class:`ObservationEncoder` is an ``nn.Module``, so a *learned* encoder's
  parameters are picked up automatically - :class:`MAPPO` optimises
  ``encoder.parameters()`` together with the actor and critic, and checkpoints
  its ``state_dict``.
* Only ``actor_input_dim`` / ``critic_input_dim`` feed the MLP constructors, so
  changing the embedding width needs no edit here.
* The rollout buffer stores whatever the env hands back (obs vectors today,
  graph batches later would be stored by the same call sites); the encoder is
  applied inside :meth:`MAPPO.act` and :meth:`MAPPO._evaluate`, which are the
  only two places tensors are built.

**Agent identity in the critic.**  ``critic_features`` tiles the global state
across agents and appends an agent-id one-hot.  This is the MAPPO paper's
agent-specific global state: rewards here are *not* fully shared (``action_cost``
and ``invalid_action`` are per agent), so a critic that could not tell agents
apart would regress every agent's return onto the team mean and blur exactly the
credit assignment that ``marl/reward.py`` splits out.  It stays CTDE - the
one-hot is metadata, the state is still centralized and still unavailable to a
decentralized actor.  Set ``agent_id_in_critic=False`` for the pure-shared-state
variant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


# ==========================================================================
# Config
# ==========================================================================
@dataclass(frozen=True)
class MAPPOConfig:
    """Hyperparameters. Frozen; serialised next to every checkpoint."""

    hidden_dim: int = 128
    n_hidden_layers: int = 2

    gamma: float = 0.99
    gae_lambda: float = 0.95

    clip_coef: float = 0.2
    clip_vloss: bool = True
    vf_clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 4
    norm_adv: bool = True
    target_kl: float | None = 0.03

    agent_id_in_critic: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================================================
# Encoder seam (see the module docstring)
# ==========================================================================
class ObservationEncoder(nn.Module):
    """Interface between "what the env hands back" and "what the nets consume".

    Subclasses must set ``actor_input_dim`` / ``critic_input_dim`` and implement
    both feature methods.  Being an ``nn.Module`` means a learned Phase 3
    encoder's parameters are optimised and checkpointed with no trainer change.
    """

    actor_input_dim: int
    critic_input_dim: int

    def actor_features(self, obs: torch.Tensor) -> torch.Tensor:
        """``(..., N, obs_dim)`` -> ``(..., N, actor_input_dim)``."""
        raise NotImplementedError

    def critic_features(self, state: torch.Tensor) -> torch.Tensor:
        """``(..., state_dim)`` -> ``(..., N, critic_input_dim)``."""
        raise NotImplementedError


class VectorObsEncoder(ObservationEncoder):
    """Pass-through encoder over the simulator's vector observation surfaces.

    This is the Phase 3 stand-in described in the module docstring: the actor
    sees ``env.observation_space(agent)`` unchanged and the critic sees
    ``env.state()`` tiled per agent (plus an agent-id one-hot).  Parameterless.
    """

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        agent_id_in_critic: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_agents = int(n_agents)
        self.agent_id_in_critic = bool(agent_id_in_critic)

        self.actor_input_dim = self.obs_dim
        self.critic_input_dim = self.state_dim + (
            self.n_agents if self.agent_id_in_critic else 0
        )
        # Buffer (not a parameter) so it moves with .to(device) and round-trips
        # through state_dict without ever being optimised.
        self.register_buffer(
            "_agent_ids", torch.eye(self.n_agents), persistent=False
        )

    def actor_features(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def critic_features(self, state: torch.Tensor) -> torch.Tensor:
        tiled = state.unsqueeze(-2).expand(
            *state.shape[:-1], self.n_agents, self.state_dim
        )
        if not self.agent_id_in_critic:
            return tiled
        ids = self._agent_ids.expand(*state.shape[:-1], self.n_agents, self.n_agents)
        return torch.cat((tiled, ids), dim=-1)


# ==========================================================================
# Networks
# ==========================================================================
def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(in_dim: int, hidden: int, layers: int, out_dim: int, out_std: float) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [_layer_init(nn.Linear(d, hidden)), nn.Tanh()]
        d = hidden
    mods.append(_layer_init(nn.Linear(d, out_dim), std=out_std))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Decentralized policy. One set of weights, shared by all N agents."""

    def __init__(self, in_dim: int, n_actions: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden, layers, n_actions, out_std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """Centralized value function over the global state (CTDE)."""

    def __init__(self, in_dim: int, hidden: int, layers: int) -> None:
        super().__init__()
        self.net = _mlp(in_dim, hidden, layers, 1, out_std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ==========================================================================
# GAE
# ==========================================================================
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    final_values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Advantage Estimation with correct time-limit handling.

    All arrays are ``(T, ...)`` with a common trailing shape; ``last_values`` is
    that trailing shape alone.

    Parameters
    ----------
    rewards, values
        ``r_t`` and ``V(s_t)`` for the collected steps.
    terminated
        True where the episode reached a **real** terminal state (the env's
        ``recovered`` / ``collapsed``).  The MDP has no future beyond it, so the
        bootstrap is exactly 0.
    truncated
        True where the episode was cut off by ``max_cycles``.  The MDP *does*
        continue; cutting it is an artefact of the harness, so the bootstrap is
        ``V(s_final)``.  Treating this as a terminal is the classic time-limit
        bug - it teaches the policy that the world ends at tick 200 and biases
        every value estimate near the horizon downwards.
    final_values
        ``V(s_final)`` of the episode that ended at step ``t`` (only read where
        ``truncated``; anything at ``terminated`` steps is overridden with 0).
        Needed because the env auto-resets, so ``values[t+1]`` belongs to a
        *new* episode.
    last_values
        ``V(s_T)`` for the step after the final collected one, used when the
        rollout simply ran out of budget mid-episode.

    Returns
    -------
    ``(advantages, returns)``, both shaped like ``rewards``.  ``returns`` is
    ``advantages + values`` (the standard GAE value target).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    final_values = np.asarray(final_values, dtype=np.float64)
    last_values = np.asarray(last_values, dtype=np.float64)
    terminated = np.asarray(terminated, dtype=bool)
    truncated = np.asarray(truncated, dtype=bool)

    n_steps = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros_like(last_values, dtype=np.float64)

    done = terminated | truncated
    not_done = (~done).astype(np.float64)
    not_terminated = (~terminated).astype(np.float64)

    for t in range(n_steps - 1, -1, -1):
        in_flight = values[t + 1] if t < n_steps - 1 else last_values
        # Where the episode ended at t the successor is that episode's final
        # state, not the freshly reset one sitting in values[t + 1].
        next_value = np.where(done[t], final_values[t], in_flight)
        # ...and where it *terminated* there is no successor at all.
        next_value = next_value * not_terminated[t]
        delta = rewards[t] + gamma * next_value - values[t]
        # The lambda-chain never crosses an episode boundary, terminal or not.
        last_gae = delta + gamma * gae_lambda * not_done[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


# ==========================================================================
# Rollout buffer
# ==========================================================================
class RolloutBuffer:
    """Fixed ``(T, E, N)`` storage for one on-policy rollout.

    ``env.agents`` empties at episode end (PettingZoo requires it), but the
    roster is otherwise constant for the whole episode and identical across
    episodes - ``simulator/cluster_env.py`` says so explicitly.  So ``N`` is a
    compile-time constant here and no ragged batching or index remapping is
    needed; the buffer just never consults ``env.agents``.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        component_names: tuple[str, ...],
    ) -> None:
        self.n_steps, self.n_envs, self.n_agents = n_steps, n_envs, n_agents
        ten = (n_steps, n_envs, n_agents)
        self.obs = np.zeros((*ten, obs_dim), dtype=np.float32)
        self.states = np.zeros((n_steps, n_envs, state_dim), dtype=np.float32)
        self.final_states = np.zeros((n_steps, n_envs, state_dim), dtype=np.float32)
        self.actions = np.zeros(ten, dtype=np.int64)
        self.logprobs = np.zeros(ten, dtype=np.float32)
        self.values = np.zeros(ten, dtype=np.float32)
        self.rewards = np.zeros(ten, dtype=np.float32)
        self.terminated = np.zeros((n_steps, n_envs), dtype=bool)
        self.truncated = np.zeros((n_steps, n_envs), dtype=bool)
        # Every reward component kept separately for the whole rollout, so the
        # training curve can be decomposed after the fact (CLAUDE.md).
        self.components = {
            k: np.zeros(ten, dtype=np.float32) for k in component_names
        }
        self.ptr = 0

    def reset(self) -> None:
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        logprob: np.ndarray,
        value: np.ndarray,
        reward: np.ndarray,
        components: dict[str, np.ndarray],
        terminated: np.ndarray,
        truncated: np.ndarray,
        final_state: np.ndarray,
    ) -> None:
        t = self.ptr
        self.obs[t] = obs
        self.states[t] = state
        self.actions[t] = action
        self.logprobs[t] = logprob
        self.values[t] = value
        self.rewards[t] = reward
        for k, v in components.items():
            self.components[k][t] = v
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.final_states[t] = final_state
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.n_steps


# ==========================================================================
# The algorithm
# ==========================================================================
class MAPPO:
    """Shared-parameter decentralized actors + a centralized critic."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int,
        config: MAPPOConfig | None = None,
        device: str | torch.device = "cpu",
        encoder: ObservationEncoder | None = None,
    ) -> None:
        self.cfg = config or MAPPOConfig()
        self.device = torch.device(device)
        self.obs_dim, self.state_dim = int(obs_dim), int(state_dim)
        self.n_agents, self.n_actions = int(n_agents), int(n_actions)

        self.encoder = (
            encoder
            if encoder is not None
            else VectorObsEncoder(
                obs_dim, state_dim, n_agents, self.cfg.agent_id_in_critic
            )
        ).to(self.device)
        self.actor = Actor(
            self.encoder.actor_input_dim,
            n_actions,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
        ).to(self.device)
        self.critic = Critic(
            self.encoder.critic_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
        ).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.actor.parameters())
            + list(self.critic.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.lr, eps=1e-5)
        self._params = params

    # ------------------------------------------------------------- acting
    @torch.no_grad()
    def act(
        self, obs: np.ndarray, state: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (or argmax) actions. ``obs`` ``(E,N,obs_dim)``, ``state`` ``(E,state_dim)``.

        Returns ``(actions, logprobs, values)``, each ``(E, N)``.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.actor(self.encoder.actor_features(obs_t))
        dist = Categorical(logits=logits)
        actions = logits.argmax(-1) if deterministic else dist.sample()
        logprobs = dist.log_prob(actions)
        values = self.critic(self.encoder.critic_features(state_t))
        return (
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def value(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        return self.critic(self.encoder.critic_features(state_t)).cpu().numpy()

    @torch.no_grad()
    def act_single(
        self, obs: np.ndarray, state: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """One env's worth of observations: ``(N, obs_dim)`` -> ``(N,)`` actions."""
        actions, _, _ = self.act(obs[None], state[None], deterministic=deterministic)
        return actions[0]

    # ------------------------------------------------------------ learning
    def _evaluate(
        self, obs: torch.Tensor, state: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(self.encoder.actor_features(obs))
        dist = Categorical(logits=logits)
        values = self.critic(self.encoder.critic_features(state))
        return dist.log_prob(actions), dist.entropy(), values

    def update(
        self,
        buffer: RolloutBuffer,
        advantages: np.ndarray,
        returns: np.ndarray,
        progress: float = 0.0,
    ) -> dict[str, float]:
        """One PPO update over the rollout. ``progress`` in [0,1] drives LR anneal."""
        cfg = self.cfg
        if cfg.anneal_lr:
            lr = cfg.lr * max(1.0 - float(progress), 0.0)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        n_steps, n_envs, n_agents = buffer.n_steps, buffer.n_envs, buffer.n_agents
        batch = n_steps * n_envs * n_agents
        dev = self.device

        # The critic consumes one row per (timestep, env, agent); the encoder
        # expands the per-env state across agents, so states are flattened over
        # (T, E) only and re-expanded inside critic_features.
        b_obs = torch.as_tensor(
            buffer.obs.reshape(n_steps * n_envs, n_agents, -1), device=dev
        )
        b_state = torch.as_tensor(
            buffer.states.reshape(n_steps * n_envs, -1), device=dev
        )
        b_actions = torch.as_tensor(buffer.actions.reshape(-1), device=dev)
        b_logprobs = torch.as_tensor(buffer.logprobs.reshape(-1), device=dev)
        b_values = torch.as_tensor(buffer.values.reshape(-1), device=dev)
        b_adv = torch.as_tensor(advantages.reshape(-1), device=dev)
        b_ret = torch.as_tensor(returns.reshape(-1), device=dev)

        n_rows = n_steps * n_envs
        minibatch_rows = max(1, n_rows // cfg.num_minibatches)
        idx = np.arange(n_rows)
        rng = np.random.default_rng()

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
        }
        n_batches = 0
        stopped_early = False

        for _ in range(cfg.update_epochs):
            rng.shuffle(idx)
            for start in range(0, n_rows, minibatch_rows):
                rows = idx[start : start + minibatch_rows]
                if rows.size == 0:
                    continue
                rows_t = torch.as_tensor(rows, device=dev)
                # (rows, N) -> flat sample indices into the (T*E*N,) tensors.
                flat = (
                    rows_t[:, None] * n_agents
                    + torch.arange(n_agents, device=dev)[None, :]
                ).reshape(-1)

                new_logp, entropy, new_values = self._evaluate(
                    b_obs[rows_t], b_state[rows_t], b_actions[flat].view(-1, n_agents)
                )
                new_logp = new_logp.reshape(-1)
                entropy = entropy.reshape(-1)
                new_values = new_values.reshape(-1)

                logratio = new_logp - b_logprobs[flat]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                    clip_frac = (
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                mb_adv = b_adv[flat]
                if cfg.norm_adv and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv
                    * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef),
                ).mean()

                mb_ret = b_ret[flat]
                if cfg.clip_vloss:
                    unclipped = (new_values - mb_ret) ** 2
                    v_clipped = b_values[flat] + torch.clamp(
                        new_values - b_values[flat],
                        -cfg.vf_clip_coef,
                        cfg.vf_clip_coef,
                    )
                    clipped = (v_clipped - mb_ret) ** 2
                    v_loss = 0.5 * torch.max(unclipped, clipped).mean()
                else:
                    v_loss = 0.5 * ((new_values - mb_ret) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self._params, cfg.max_grad_norm)
                self.optimizer.step()

                stats["policy_loss"] += float(pg_loss.item())
                stats["value_loss"] += float(v_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["grad_norm"] += float(grad_norm)
                n_batches += 1

            if cfg.target_kl is not None and n_batches:
                if stats["approx_kl"] / n_batches > cfg.target_kl:
                    stopped_early = True
                    break

        for key in (
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "clip_frac",
            "grad_norm",
        ):
            stats[key] /= max(n_batches, 1)

        # Explained variance: the standard "is the critic doing anything" check.
        y_pred = buffer.values.reshape(-1)
        y_true = returns.reshape(-1)
        var_y = float(np.var(y_true))
        stats["explained_variance"] = (
            float("nan") if var_y == 0 else 1.0 - float(np.var(y_true - y_pred)) / var_y
        )
        stats["lr"] = float(self.optimizer.param_groups[0]["lr"])
        stats["early_stop"] = float(stopped_early)
        stats["batch_size"] = float(batch)
        return stats

    # --------------------------------------------------------- persistence
    def state_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, payload: dict[str, Any], load_optimizer: bool = True) -> None:
        self.encoder.load_state_dict(payload["encoder"], strict=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if load_optimizer and "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])

    def train(self) -> None:
        self.encoder.train()
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._params if p.requires_grad)
