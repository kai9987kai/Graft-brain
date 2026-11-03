"""The grafted connectome as a running dynamical system, coupled to an artificial neural network
and biological sub-circuits (Central Complex ring attractor compass, Mushroom Body associative
learning with dopaminergic modulation, and intrinsic curiosity via RND).

Dynamics (rate model, one microstep):
    x   = W_conn r + W_graft r + b + P_s * I_sens
    r  <- (1 - a) r + a * relu(tanh(x)) + noise

Biological & algorithmic innovations:
1. **Homeostatic Activity Regulation**: intrinsic excitability (b_i) continuously adapts
   to keep the 12,000-neuron recurrent graph within a stable dynamic range (target r* ~ 0.12).
2. **Central Complex (CX) Compass**: 16-wedge ring attractor path-integrates angular velocity
   and anchors to visual cues, providing a biological heading reference.
3. **Mushroom Body (MB) Circuit**: sparse Kenyon Cells (top-k WTA) with plastic KC->MBON
   synapses modulated by Dopaminergic Neurons (DANs) on TD error.
4. **PPO with Generalized Advantage Estimation (GAE)**: clipped surrogate objective with
   multi-step lambda returns to stabilize policy learning through the recurrent graph.
5. **Intrinsic Curiosity (RND)**: exploration bonus from prediction error in sparse-reward tasks.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graft import GraftedGraph
from .neuro_modules import CentralComplexCompass, CuriosityRND, MushroomBody


class GraftedBrain(nn.Module):
    def __init__(self, g: GraftedGraph, n_sensory=256, n_motor=128, n_probe=256, leak=0.35,
                 noise=0.01, microsteps=4, target_rate=0.12, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        N = g.n
        self.n, self.leak, self.noise, self.microsteps = N, leak, noise, microsteps
        self.target_rate = target_rate

        self.register_buffer("base_pre", torch.tensor(g.base_pre, dtype=torch.long))
        self.register_buffer("base_post", torch.tensor(g.base_post, dtype=torch.long))
        self.register_buffer("base_w", torch.tensor(g.base_w, dtype=torch.float32))
        self.register_buffer("graft_pre", torch.tensor(g.graft_pre, dtype=torch.long))
        self.register_buffer("graft_post", torch.tensor(g.graft_post, dtype=torch.long))
        self.register_buffer("graft_site", torch.tensor(g.graft_site, dtype=torch.long))
        self.graft_w = nn.Parameter(torch.tensor(g.graft_w0))
        self.register_buffer("bias", torch.tensor(rng.normal(0.02, 0.02, N), dtype=torch.float32))
        self.register_buffer("r", torch.zeros(N))
        self.register_buffer("activity_ema", torch.full((N,), target_rate, dtype=torch.float32))
        self.register_buffer("elig", torch.zeros(len(g.graft_pre)))
        self.register_buffer("heb", torch.zeros(len(g.graft_pre)))

        def choose(roles, k):
            pool = np.flatnonzero(np.isin(g.role, roles))
            if len(pool) < k:  # top up with central neurons
                pool = np.concatenate([pool, rng.choice(np.flatnonzero(g.role == "central"), k - len(pool), replace=False)])
            # spread the ports evenly across all grafted connectomes
            per = [pool[g.origin[pool] == c] for c in range(len(g.names))]
            out, i = [], 0
            while len(out) < k:
                p = per[i % len(per)]
                if len(p):
                    out.append(p[(i // len(per)) % len(p)])
                i += 1
            return torch.tensor(np.unique(out)[:k], dtype=torch.long)

        self.register_buffer("sensory_idx", choose(["sensory", "visual"], n_sensory))
        self.register_buffer("motor_idx", choose(["motor", "descending"], n_motor))
        self.register_buffer("probe_idx", torch.tensor(rng.choice(N, min(n_probe, N), replace=False), dtype=torch.long))

    def graft_current(self, r):
        return torch.zeros_like(r).index_add(0, self.graft_post, self.graft_w * r[self.graft_pre])

    def forward(self, i_sens: torch.Tensor):
        r = self.r.detach().clone()
        drive = torch.zeros(self.n).index_add(0, self.sensory_idx, i_sens)
        for _ in range(self.microsteps):
            x = (torch.zeros_like(r)
                 .index_add(0, self.base_post, self.base_w * r[self.base_pre])
                 .index_add(0, self.graft_post, self.graft_w * r[self.graft_pre])) + self.bias + drive
            r = (1 - self.leak) * r + self.leak * F.relu(torch.tanh(x))
            if self.noise:
                r = r + self.noise * torch.randn_like(r)
            r = r.clamp(0.0, 1.0)
        with torch.no_grad():
            self.elig.mul_(0.9).add_(r[self.graft_post].detach() * self.r[self.graft_pre])
            self.r.copy_(r.detach())
            # Homeostatic activity regulation (drift bias toward target firing rate)
            self.activity_ema.mul_(0.99).add_(0.01 * r.detach())
            self.bias.add_(0.0005 * (self.target_rate - self.activity_ema)).clamp_(-0.5, 0.5)
        return r

    @torch.no_grad()
    def hebbian(self, td_error: float, eta=1e-3):
        """Accumulate a three-factor update; applied in :meth:`apply_hebbian` (after backprop)."""
        self.heb.add_(eta * float(np.clip(td_error, -2, 2)) * (self.elig - self.elig.mean()))

    @torch.no_grad()
    def apply_hebbian(self, clip=0.5):
        self.graft_w.add_(self.heb).clamp_(-clip, clip)
        self.heb.zero_()

    @torch.no_grad()
    def reset_state(self):
        self.r.zero_(); self.elig.zero_(); self.heb.zero_()


class NeuralInterface(nn.Module):
    """ANN <-> connectome coupling with Central Complex (CX) compass and Mushroom Body (MB) links."""

    def __init__(self, obs_dim, cam_shape, n_sensory, n_motor, n_probe, act_dim, hidden=256):
        super().__init__()
        # Camera visual pathway
        self.cam = nn.Sequential(nn.Conv2d(2, 8, 5, stride=2), nn.ReLU(), nn.Conv2d(8, 16, 5, stride=2), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(16 * 12, 32), nn.ReLU())
        # Encoder: receives raw obs + cam (32) + CX compass bump (16) + MBON readout (2)
        in_dim = obs_dim + 32 + 16 + 2
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_sensory))
        # Decoder: reads motor neurons + probe neurons + MBON readout (2)
        dec_in = n_motor + n_probe + 2
        self.dec = nn.Sequential(nn.Linear(dec_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def encode(self, obs, cam, compass, mbon):
        cam_feat = self.cam(cam.unsqueeze(0)).squeeze(0)
        h_in = torch.cat([obs, cam_feat, compass, mbon])
        return 2.0 * torch.tanh(self.enc(h_in))

    def decode(self, motor_rates, probe_rates, mbon):
        h = self.dec(torch.cat([motor_rates, probe_rates, mbon]))
        return torch.tanh(self.mu(h)), self.v(h).squeeze(-1)


class Agent:
    """Embodied connectomic agent with PPO, GAE, Central Complex compass, MB, and RND curiosity."""

    def __init__(self, graph: GraftedGraph, obs_dim, cam_shape, act_dim, lr=3e-4, rollout=32,
                 gamma=0.98, gae_lambda=0.95, ppo_clip=0.2, microsteps=4, seed=0):
        torch.manual_seed(seed)
        self.brain = GraftedBrain(graph, microsteps=microsteps, seed=seed)
        b = self.brain

        self.compass = CentralComplexCompass(n_wedges=16)
        self.mb = MushroomBody(in_dim=4, n_kc=64, k_sparse=4, n_mbon=2)
        self.curiosity = CuriosityRND(obs_dim=obs_dim)

        self.iface = NeuralInterface(obs_dim, cam_shape, len(b.sensory_idx), len(b.motor_idx), len(b.probe_idx), act_dim)
        self.opt = torch.optim.Adam([{"params": self.iface.parameters()},
                                     {"params": [b.graft_w], "lr": lr * 0.3}], lr=lr)

        self.rollout = rollout
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_clip = ppo_clip
        self.buf = []
        self.last_value = 0.0
        self.last_obs = np.zeros(obs_dim, dtype=np.float32)
        self.last_compass = torch.zeros(16)
        self.last_mbon = torch.zeros(2)

        self.stats = {
            "updates": 0, "loss": 0.0, "td": 0.0, "curiosity": 0.0,
            "graft_w_abs": float(b.graft_w.detach().abs().mean()),
            "compass_bump": 0.0, "kc_sparsity": 0.0,
        }

    def act(self, obs: np.ndarray, cam: np.ndarray):
        self.last_obs = obs
        o = torch.tensor(obs, dtype=torch.float32)
        c = torch.tensor(cam, dtype=torch.float32)

        # 1. Update Central Complex ring attractor with angular velocity and retina contrast
        ang_vel = float(obs[33 * 4 + 5]) * 2.5
        compass_act = self.compass.step(ang_vel, retina_rays=obs[:33 * 4])

        # 2. Update Mushroom Body sparse Kenyon Cells with current task cue (color + active)
        cue_offset = 33 * 4 + 8 + 9
        cue_tensor = torch.tensor(obs[cue_offset:cue_offset + 4], dtype=torch.float32)
        mbon_act = self.mb(cue_tensor)

        self.last_compass = compass_act.detach()
        self.last_mbon = mbon_act.detach()
        self.stats["compass_bump"] = float(self.compass.bump_angle)
        self.stats["kc_sparsity"] = float((self.mb.kc_act > 0).float().mean())

        # 3. Step connectome dynamics
        i_sens = self.iface.encode(o, c, compass_act, mbon_act)
        r = self.brain(i_sens)

        # 4. Decode motor output and value
        mu, v = self.iface.decode(r[self.brain.motor_idx], r[self.brain.probe_idx], mbon_act)
        std = self.iface.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()

        # Buffer: [log_prob, value, entropy, reward (placeholder), done (placeholder)]
        self.buf.append([dist.log_prob(a).sum(), v, dist.entropy().sum(), None, None])
        return a.clamp(-1, 1).numpy(), float(v.detach())

    def feedback(self, reward: float, done: bool):
        # 1. Add intrinsic curiosity bonus
        o_t = torch.tensor(self.last_obs, dtype=torch.float32)
        curiosity_bonus = self.curiosity.compute_bonus(o_t, train=True)
        r_total = float(reward) + 0.05 * curiosity_bonus
        self.stats["curiosity"] = 0.95 * self.stats["curiosity"] + 0.05 * curiosity_bonus

        self.buf[-1][3] = r_total
        self.buf[-1][4] = done

        # 2. TD error for 3-factor Hebbian graft plasticity and MB dopaminergic modulation
        if len(self.buf) >= 2:
            prev_v = float(self.buf[-2][1].detach())
            next_v = float(self.buf[-1][1].detach())
            td = self.buf[-2][3] + (0.0 if self.buf[-2][4] else self.gamma * next_v) - prev_v
            self.brain.hebbian(td)
            self.mb.dopaminergic_update(td)
            self.stats["td"] = 0.95 * self.stats["td"] + 0.05 * td

        # 3. Multi-step PPO / GAE update
        if done:
            self._update(bootstrap=0.0)
        elif len(self.buf) > self.rollout:
            last = self.buf.pop()
            self._update(bootstrap=float(last[1]))

    def _update(self, bootstrap: float):
        n_steps = len(self.buf)
        rewards = [b[3] for b in self.buf]
        dones = [b[4] for b in self.buf]
        values = [float(b[1]) for b in self.buf] + [bootstrap]
        old_lps = torch.stack([b[0] for b in self.buf]).detach()
        old_vs = torch.stack([b[1] for b in self.buf])
        ents = torch.stack([b[2] for b in self.buf])

        # Generalized Advantage Estimation (GAE)
        advantages = []
        gae = 0.0
        for t in reversed(range(n_steps)):
            next_val = values[t + 1]
            non_terminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + old_vs.detach()

        if len(advantages) > 1 and advantages.std() > 1e-6:
            adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        else:
            adv_norm = advantages

        # PPO clipped surrogate objective
        lps = torch.stack([b[0] for b in self.buf])
        ratio = torch.exp(lps - old_lps)
        surr1 = ratio * adv_norm
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * adv_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * F.smooth_l1_loss(old_vs, returns)
        entropy_loss = -0.01 * ents.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        self.opt.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(self.iface.parameters()) + [self.brain.graft_w], 1.0)
        self.opt.step()
        self.brain.apply_hebbian()
        self.buf.clear()

        self.stats.update(
            updates=self.stats["updates"] + 1,
            loss=float(total_loss),
            graft_w_abs=float(self.brain.graft_w.detach().abs().mean())
        )

    # ---- persistence ----
    def state_dict(self):
        return {
            "brain": self.brain.state_dict(),
            "iface": self.iface.state_dict(),
            "mb": self.mb.state_dict(),
            "curiosity": self.curiosity.state_dict(),
            "opt": self.opt.state_dict(),
            "stats": self.stats,
        }

    def load_state_dict(self, sd):
        self.brain.load_state_dict(sd["brain"], strict=False)
        self.iface.load_state_dict(sd["iface"], strict=False)
        if "mb" in sd:
            self.mb.load_state_dict(sd["mb"], strict=False)
        if "curiosity" in sd:
            self.curiosity.load_state_dict(sd["curiosity"], strict=False)
        self.opt.load_state_dict(sd["opt"])
        self.stats.update(sd.get("stats", {}))
        self.buf.clear()
