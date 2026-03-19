import deprl
import sconegym
import gym
import argparse
from trainFM import EMGTransformer, ReplayBuffer, QNetwork, compute_impedance_torque, soft_update
import os
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from visualizer import TrainingVisualizer


# ──────────────────────────────────────────────────────────────────────────────
# Noise Configuration + Helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NoiseConfig:
    """
    Sim-to-real noise configuration. All magnitude params are UPPER BOUNDS —
    actual noise is domain-randomized fresh each draw so the policy learns
    robustness across the full distribution [0, max], including the clean case.

    Signal noise: each channel/dim draws its own (μ_c, std_c) independently,
    giving natural inter-channel variance without explicit per-channel config.

    Temporal jitter: EMG and kin are jittered independently each step/sample,
    modeling desynchronized sensor latencies between the EMG processing pipeline
    and joint encoders.

    noise_on_rollout — gates both signal noise AND temporal jitter at act time.
                       Clean observations are always stored; noise only affects
                       what the policy sees during the forward pass.
    noise_on_replay  — gates both signal noise AND temporal jitter at SAC
                       sample time (done-signal-guarded buffer lookback for
                       jitter; fresh noise draw for signal noise each sample).
    """
    # Signal noise upper bounds (0.0 = disabled)
    emg_noise_std_max:  float = 0.0   # per-channel std ~ U[0, std_max]
    emg_noise_mean_max: float = 0.0   # per-channel μ ~ U[-mean_max, mean_max]
    kin_noise_std_max:  float = 0.0
    kin_noise_mean_max: float = 0.0
    # Temporal jitter upper bounds in env steps (0 = disabled)
    emg_jitter_max: int = 0           # δ ~ U[0, jitter_max]; never forward in time
    kin_jitter_max: int = 0
    # Jitter curriculum: linearly ramp 0 → configured max over N steps (0 = off)
    jitter_warmup_steps: int = 0
    # Application gates
    noise_on_rollout: bool = False
    noise_on_replay:  bool = False

    def effective_jitter(self, curr_step: int) -> Tuple[int, int]:
        """Return (emg_jitter, kin_jitter) scaled by warmup curriculum."""
        if self.jitter_warmup_steps <= 0:
            return self.emg_jitter_max, self.kin_jitter_max
        scale = min(1.0, curr_step / self.jitter_warmup_steps)
        return (
            int(round(self.emg_jitter_max * scale)),
            int(round(self.kin_jitter_max * scale)),
        )

    @property
    def any_emg_noise(self) -> bool:
        return self.emg_noise_std_max > 0.0 or self.emg_noise_mean_max > 0.0

    @property
    def any_kin_noise(self) -> bool:
        return self.kin_noise_std_max > 0.0 or self.kin_noise_mean_max > 0.0


# ── Single-sample noise (rollout) ─────────────────────────────────────────────

def _signal_noise_emg_single(emg: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    """
    Per-channel domain-randomized signal noise for a single EMG window.

    Args:
        emg: (13, 100) float32 tensor on any device.

    Returns:
        Noisy (13, 100) tensor, clipped to [-1, 1]. Input unchanged if noise
        params are zero.
    """
    if not cfg.any_emg_noise:
        return emg
    C = emg.shape[0]  # 13 channels
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.emg_noise_std_max, C),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)                                          # (13, 1) → broadcasts over time
    means = torch.tensor(
        np.random.uniform(-cfg.emg_noise_mean_max, cfg.emg_noise_mean_max, C),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)                                          # (13, 1)
    noise = means + sigmas * torch.randn_like(emg)
    return torch.clamp(emg + noise, -1.0, 1.0)


def _signal_noise_kin_single(kin: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    """
    Per-dim domain-randomized signal noise for a single kin vector.

    Args:
        kin: (27,) float32 tensor on any device.
    """
    if not cfg.any_kin_noise:
        return kin
    D = kin.shape[0]  # 27 dims
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.kin_noise_std_max, D),
        dtype=torch.float32, device=kin.device,
    )
    means = torch.tensor(
        np.random.uniform(-cfg.kin_noise_mean_max, cfg.kin_noise_mean_max, D),
        dtype=torch.float32, device=kin.device,
    )
    return kin + means + sigmas * torch.randn_like(kin)


def _emg_window_from_frame_buffer(emg_frame_buf: deque, delta: int) -> torch.Tensor:
    """
    Reconstruct a genuine (13, 100) EMG window ending `delta` steps in the past.

    The buffer holds individual (13,) frames in chronological order, newest last,
    with depth = 100 + emg_jitter_max so every valid δ can be reconstructed from
    real historical data — no padding, no approximation.

        delta=0  →  buf[-100:]        (current window, same as emg_windows)
        delta=δ  →  buf[-(100+δ):-δ]  (window as it was δ steps ago)

    Args:
        emg_frame_buf: deque of (13,) CPU float32 tensors, newest last.
        delta:         steps to look back; 0 returns the current window.

    Returns:
        (13, 100) float32 tensor on CPU.
    """
    buf = list(emg_frame_buf) # 100,13 shape 
    if delta == 0:
        frames = buf[-100:]
    else:
        frames = buf[-(100 + delta):-delta]

    return torch.stack(frames, dim=1)  # (13, 100)


def _rollout_emg_noise(
    emg_frame_buf: deque,
    cfg: NoiseConfig,
    eff_jitter: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Sample temporal jitter δ ~ U[0, eff_jitter], reconstruct the genuine
    historical EMG window from the frame buffer, then apply per-channel
    domain-randomized signal noise.

    Args:
        emg_frame_buf: deque of (13,) CPU frames, depth = 100 + emg_jitter_max.
        cfg:           NoiseConfig.
        eff_jitter:    warmup-scaled effective jitter max for this step.
        device:        target device for the returned tensor.

    Returns:
        Noisy (13, 100) tensor on `device`.
    """
    delta = int(np.random.randint(0, eff_jitter + 1)) if eff_jitter > 0 else 0
    w = _emg_window_from_frame_buffer(emg_frame_buf, delta).to(device)
    return _signal_noise_emg_single(w, cfg)


def _rollout_kin_noise(
    kin: torch.Tensor,
    cfg: NoiseConfig,
    kin_buffer: deque,
    eff_jitter: int,
) -> torch.Tensor:
    """
    Sample a stale kin snapshot from the circular buffer (temporal jitter),
    then apply per-dim signal noise.

    Args:
        kin:        current clean kin tensor (27,) on device — used if δ=0.
        kin_buffer: deque of recent clean kin CPU tensors, newest last.
        eff_jitter: warmup-scaled effective jitter max for this step.
    """
    k = kin
    if eff_jitter > 0 and len(kin_buffer) > 1:
        max_lookback = min(eff_jitter, len(kin_buffer) - 1)
        delta = int(np.random.randint(0, max_lookback + 1))
        if delta > 0:
            k = kin_buffer[-(delta + 1)].to(kin.device)
    return _signal_noise_kin_single(k, cfg)


# ── Batch noise (replay, applied after sample_with_jitter) ───────────────────

def _batch_signal_noise_emg(emg: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    """
    Per-element, per-channel domain-randomized signal noise for a batch.

    Args:
        emg: (B, 13, 100) float32 tensor on device.

    Returns:
        Noisy (B, 13, 100), clipped to [-1, 1].
    """
    if not cfg.any_emg_noise:
        return emg
    B, C, _ = emg.shape
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.emg_noise_std_max, (B, C)),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)                                          # (B, 13, 1)
    means = torch.tensor(
        np.random.uniform(-cfg.emg_noise_mean_max, cfg.emg_noise_mean_max, (B, C)),
        dtype=torch.float32, device=emg.device,
    ).unsqueeze(-1)                                          # (B, 13, 1)
    noise = means + sigmas * torch.randn_like(emg)
    return torch.clamp(emg + noise, -1.0, 1.0)


def _batch_signal_noise_kin(kin: torch.Tensor, cfg: NoiseConfig) -> torch.Tensor:
    """
    Per-element, per-dim domain-randomized signal noise for a batch.

    Args:
        kin: (B, 27) float32 tensor on device.
    """
    if not cfg.any_kin_noise:
        return kin
    B, D = kin.shape
    sigmas = torch.tensor(
        np.random.uniform(0.0, cfg.kin_noise_std_max, (B, D)),
        dtype=torch.float32, device=kin.device,
    )
    means = torch.tensor(
        np.random.uniform(-cfg.kin_noise_mean_max, cfg.kin_noise_mean_max, (B, D)),
        dtype=torch.float32, device=kin.device,
    )
    return kin + means + sigmas * torch.randn_like(kin)


# ── NoisyReplayBuffer ─────────────────────────────────────────────────────────

class NoisyReplayBuffer(ReplayBuffer):
    """
    Extends ReplayBuffer with sample_with_jitter — applies independent
    per-element temporal jitter (done-signal guarded) at sample time.
    Signal noise is applied separately in train_sac after tensor conversion.

    Parent attribute mapping (from trainFM.ReplayBuffer):
        self.mem_size        ← max_size arg
        self.ptr             ← circular write pointer
        self.size            ← current fill level (capped at mem_size)
        self.state_memory    ← (mem_size, state_dim) float32
        self.new_state_memory← (mem_size, state_dim) float32
        self.action_memory   ← (mem_size, action_dim) float32
        self.reward_memory   ← (mem_size,) float32
        self.terminal_memory ← (mem_size,) bool
    """

    def _clamp_to_episode(self, idx: int, delta: int) -> int:
        """
        Reduce delta until the lookback window stays within the current episode.

        Two safety conditions:
          1. Buffer hasn't fully wrapped: indices below 0 would alias to stale/
             uninitialised entries at the top of the array, so clamp delta to
             at most idx (the number of filled entries before this one).
          2. A terminal_memory[j]=True at lookback position j means j ended an
             episode, so j+1 is a new episode start — stop before crossing it.
        """
        # Clamp to filled region when buffer hasn't wrapped
        if self.size < self.mem_size:
            delta = min(delta, idx)

        for d in range(1, delta + 1):
            lookback = (idx - d) % self.mem_size
            if self.terminal_memory[lookback]:
                return d - 1   # last safe step before the episode boundary
        return delta

    def sample_with_jitter(
        self,
        batch_size: int,
        noise_cfg: 'NoiseConfig',
        bilateral: bool = False,
        curr_step: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch with per-element independently sampled temporal jitter.

        For each element b in the batch:
          δ_emg ~ U[0, eff_emg_jitter], clamped to not cross a done boundary
          δ_kin ~ U[0, eff_kin_jitter],  sampled independently from δ_emg

        The EMG slice of states[b] is substituted from state_memory[b - δ_emg],
        and the kin slice from state_memory[b - δ_kin], while action, reward,
        next_state, and done always come from the base index unchanged.

        Signal noise is NOT applied here — it is applied in train_sac after
        tensor conversion via _batch_signal_noise_emg / _kin.

        State layout (matches parse_sides):
            unilateral: [emg(1300) | kin(27)]
            bilateral:  [emg_R(1300) | kin_R(27) | emg_L(1300) | kin_L(27)]
        """
        max_mem = min(self.size, self.mem_size)
        assert max_mem > 0, 'Buffer is empty!'
        batch_size = min(batch_size, max_mem)
        base_indices = np.random.choice(max_mem, batch_size, replace=(max_mem < batch_size))

        eff_emg_jitter, eff_kin_jitter = noise_cfg.effective_jitter(curr_step)

        states  = self.state_memory[base_indices].copy()
        states_ = self.new_state_memory[base_indices].copy()

        # ── temporal jitter — per element, EMG and kin independently ──────────
        if eff_emg_jitter > 0 or eff_kin_jitter > 0:
            # (emg_start, emg_end, kin_start, kin_end) per side
            sides_slices = (
                [(0, 1300, 1300, 1327), (1327, 2627, 2627, 2654)]
                if bilateral else
                [(0, 1300, 1300, 1327)]
            )
            for b, idx in enumerate(base_indices):
                for es, ee, ks, ke in sides_slices:
                    if eff_emg_jitter > 0:
                        d_emg = self._clamp_to_episode(
                            int(idx), int(np.random.randint(0, eff_emg_jitter + 1))
                        )
                        if d_emg > 0:
                            src = (int(idx) - d_emg) % self.mem_size
                            states[b, es:ee] = self.state_memory[src, es:ee]

                    if eff_kin_jitter > 0:
                        d_kin = self._clamp_to_episode(
                            int(idx), int(np.random.randint(0, eff_kin_jitter + 1))
                        )
                        if d_kin > 0:
                            src = (int(idx) - d_kin) % self.mem_size
                            states[b, ks:ke] = self.state_memory[src, ks:ke]

        actions = self.action_memory[base_indices]
        rewards = self.reward_memory[base_indices]
        dones   = self.terminal_memory[base_indices]

        return states, states_, actions, rewards, dones


#TODO isometric actuation zeroing debug: default_activation, minimum_activation 
#TODO minimum replay buffer size 10k-50k (25k?)
#TODO prioritized experience replay -> binary tree? 
#NOTE^^ training on different (t) policies actions by acting in the environment
#NOTE kinematic and impedance masks are applied at log pdf calculation and state variable representation(before Q parameterization) to prevent non used index gradient noise
#NOTE jitter is happening uniformly per given emg sample 
#TODO github reformat
#TODO noise options
#TODO arg paramd saving and loading functionality :: RL save paths of policy, replayBuff and critic


# ──────────────────────────────────────────────────────────────────────────────
# SAC Training
# ──────────────────────────────────────────────────────────────────────────────

def parse_sides(states, actions=None, bilateral=False):
    """
    Returns sides: list of (emg, kin) tuples — length 1 (unilateral) or 2 (bilateral).
    If actions provided, also returns act_list aligned to sides.
    State layout:
        unilateral:  [emg(1300) | kin(27)]
        bilateral:   [emg_R(1300) | kin_R(27) | emg_L(1300) | kin_L(27)]
    Action layout:
        unilateral:  [action(54)]
        bilateral:   [action_R(54) | action_L(54)]
    """
    B = states.shape[0]
    if bilateral:
        emg_R = states[:, :1300].reshape(B, 13, 100)
        kin_R = states[:, 1300:1327]
        emg_L = states[:, 1327:2627].reshape(B, 13, 100)
        kin_L = states[:, 2627:2654]
        sides = [(emg_R, kin_R), (emg_L, kin_L)]
        if actions is not None:
            return sides, [actions[:, :54], actions[:, 54:]]
    else:
        emg = states[:, :1300].reshape(B, 13, 100)
        kin = states[:, 1300:1327]
        sides = [(emg, kin)]
        if actions is not None:
            return sides, [actions]
    return sides


def train_sac(optimizer_and_scheduler, policy_args, critic_args, Policy,
              QNetwork_base1, QNetwork_base2,
              QNetwork_target1, QNetwork_target2,
              replay_buff, training_epochs, training_losses,
              bilateral=False, sample_batch_size=256,
              noise_cfg: Optional[NoiseConfig] = None,
              curr_step: int = 0):

    gamma, tau = 0.99, 0.05
    training_iterations = 0

    if replay_buff.size < sample_batch_size:
        return

    while training_iterations < training_epochs:

        # ── sample batch — with jitter if noise_on_replay, else standard ──────
        if noise_cfg is not None and noise_cfg.noise_on_replay and \
                isinstance(replay_buff, NoisyReplayBuffer):
            states, states_, actions, rewards, dones = replay_buff.sample_with_jitter(
                sample_batch_size, noise_cfg, bilateral=bilateral, curr_step=curr_step
            )
        else:
            states, states_, actions, rewards, dones = replay_buff.sample_buffer(sample_batch_size)

        states  = torch.tensor(states,  dtype=torch.float32).to('cuda')
        states_ = torch.tensor(states_, dtype=torch.float32).to('cuda')
        actions = torch.tensor(actions, dtype=torch.float32).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda').unsqueeze(-1)
        dones   = torch.tensor(dones,   dtype=torch.float32).to('cuda').unsqueeze(-1)

        #TODO why does parse_sides need an action?
        sides,  act_list = parse_sides(states,  actions,  bilateral)
        sides_           = parse_sides(states_,           bilateral=bilateral)

        # ── replay signal noise — per-element, per-channel domain randomized ──
        if noise_cfg is not None and noise_cfg.noise_on_replay:
            sides = [
                (_batch_signal_noise_emg(emg, noise_cfg),
                 _batch_signal_noise_kin(kin, noise_cfg))
                for emg, kin in sides
            ]
            sides_ = [
                (_batch_signal_noise_emg(emg, noise_cfg),
                 _batch_signal_noise_kin(kin, noise_cfg))
                for emg, kin in sides_
            ]

        # ── Critic Update ─────────────────────────────────────────────────────
        for p in QNetwork_base1.parameters(): p.requires_grad = True
        for p in QNetwork_base2.parameters(): p.requires_grad = True

        q1_loss = q2_loss = torch.tensor(0.0, device='cuda')
        cq1_list, cq2_list = [], []

        with torch.no_grad():
            ys = []
            for (emg_, kin_) in sides_:
                out_ = Policy(emg_.to(Policy.device), kin_.to(Policy.device), sample=True)
                next_act = torch.cat([
                    out_['pred_kin_state']  * Policy.kinematic_mask.unsqueeze(0),
                    out_['pred_impedance']  * Policy.kinematic_mask.unsqueeze(0)
                ], dim=-1)
                tq = torch.min(
                    QNetwork_target1(emg_, kin_, next_act),
                    QNetwork_target2(emg_, kin_, next_act)
                )
                log_pdf_ = out_['pred_kin_log_pdf'] + out_['pred_impedance_log_pdf']
                ys.append(rewards + gamma * (1 - dones) * (tq - Policy.log_alpha.exp().detach() * log_pdf_))

        for (emg, kin), act, y in zip(sides, act_list, ys):
            cq1 = QNetwork_base1(emg, kin, act)
            cq2 = QNetwork_base2(emg, kin, act)
            cq1_list.append(cq1)
            cq2_list.append(cq2)
            q1_loss += F.huber_loss(cq1, y)
            q2_loss += F.huber_loss(cq2, y)

        optimizer_and_scheduler['q1b']['optimizer'].zero_grad()
        optimizer_and_scheduler['q2b']['optimizer'].zero_grad()
        q1_loss.backward(retain_graph=True)
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(QNetwork_base1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(QNetwork_base2.parameters(), 1.0)
        optimizer_and_scheduler['q1b']['optimizer'].step()
        optimizer_and_scheduler['q2b']['optimizer'].step()

        # ── Actor Update ──────────────────────────────────────────────────────
        for p in QNetwork_base1.parameters(): p.requires_grad = False
        for p in QNetwork_base2.parameters(): p.requires_grad = False

        actor_loss = torch.tensor(0.0, device='cuda')
        log_pdfs_all = []

        for (emg, kin) in sides:
            out = Policy(emg.to(Policy.device), kin.to(Policy.device), sample=True)
            sampled_act = torch.cat([
                out['pred_kin_state']  * Policy.kinematic_mask.unsqueeze(0),
                out['pred_impedance']  * Policy.kinematic_mask.unsqueeze(0)
            ], dim=-1)
            q = torch.min(QNetwork_base1(emg, kin, sampled_act),
                          QNetwork_base2(emg, kin, sampled_act))
            log_pdf = out['pred_kin_log_pdf'] + out['pred_impedance_log_pdf']
            log_pdfs_all.append(log_pdf)
            actor_loss += (Policy.log_alpha.exp() * log_pdf - q).mean()

        optimizer_and_scheduler['policy']['optimizer'].zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(Policy.parameters(), 1.0)
        optimizer_and_scheduler['policy']['optimizer'].step()

        for p in QNetwork_base1.parameters(): p.requires_grad = True
        for p in QNetwork_base2.parameters(): p.requires_grad = True

        # ── Alpha Update ──────────────────────────────────────────────────────
        avg_log_pdf = torch.stack(log_pdfs_all).mean(0).detach()
        alpha_loss = -(Policy.log_alpha * (avg_log_pdf + Policy.target_entropy)).mean()

        optimizer_and_scheduler['policy_log_alpha']['optimizer'].zero_grad()
        alpha_loss.backward()
        optimizer_and_scheduler['policy_log_alpha']['optimizer'].step()

        # ── Soft Updates + Logging ────────────────────────────────────────────
        soft_update(QNetwork_base1, QNetwork_target1, tau)
        soft_update(QNetwork_base2, QNetwork_target2, tau)

        training_losses['actor_loss'].append(actor_loss.item())
        training_losses['q1_loss'].append(q1_loss.item())
        training_losses['q2_loss'].append(q2_loss.item())
        training_losses['alpha_loss'].append(alpha_loss.item())
        training_losses['q1_mean'].append(torch.stack(cq1_list).mean().item())
        training_losses['q2_mean'].append(torch.stack(cq2_list).mean().item())

        training_iterations += 1

    print("\n--- Training Phase Complete ---")
    print(f"  Actor Loss: {np.mean(training_losses['actor_loss']):.4f}")
    print(f"  Q1 Loss:    {np.mean(training_losses['q1_loss']):.4f}")
    print(f"  Q2 Loss:    {np.mean(training_losses['q2_loss']):.4f}")

    Policy.save_checkpoint(
        optimizer_and_scheduler['policy']['optimizer'],
        optimizer_and_scheduler['policy']['scheduler'],
        policy_args, np.mean(training_losses['actor_loss']),
        training_iterations,
        optimizer_and_scheduler['policy_log_alpha']['optimizer'],
        optimizer_and_scheduler['policy_log_alpha']['scheduler']
    )
    QNetwork_base1.save_checkpoint('q1b', critic_args, optimizer_and_scheduler['q1b']['optimizer'], optimizer_and_scheduler['q1b']['scheduler'])
    QNetwork_base2.save_checkpoint('q2b', critic_args, optimizer_and_scheduler['q2b']['optimizer'], optimizer_and_scheduler['q2b']['scheduler'])
    QNetwork_target1.save_checkpoint('q1t', critic_args, optimizer_and_scheduler['q1t']['optimizer'], optimizer_and_scheduler['q1t']['scheduler'])
    QNetwork_target2.save_checkpoint('q2t', critic_args, optimizer_and_scheduler['q2t']['optimizer'], optimizer_and_scheduler['q2t']['scheduler'])
    replay_buff.save()
    print("Checkpoints saved.")


# ──────────────────────────────────────────────────────────────────────────────
# Amputation Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AmputationConfig:
    name: str
    env_id: str
    action_indices: List[int]       # isometric zeroing indices passed to env
    obs_rearrange_tag: str          # passed to rearrange_obs
    concat_tag: str                 # passed to concatenate_actions
    bilateral: bool
    agent_obs_slices: Tuple         # slices for rebuilding agent obs w/o prosthetic channels
    n_prosthetic_joints: int        # number of actuated prosthetic joints
    emg_side_slices: Tuple          # slices into excitation_buffer per side (r, l)
    tibial: bool = False            # True → populate below-knee EMG channels
    subfolder: str = ''             # checkpoint subdirectory for this config


AMPUTATION_CONFIGS = {
    'transtibial_left': AmputationConfig(
        name='transtibial_left',
        env_id='sconewalk_h0111_osim-v1',
        subfolder='tb_left',
        action_indices=[0],
        obs_rearrange_tag='tibial_left',
        concat_tag='tibial_left',
        bilateral=False,
        agent_obs_slices=(slice(0, 45), slice(46, None)),
        n_prosthetic_joints=1,
        emg_side_slices=(slice(9, None),),
        tibial=True
    ),
    'transtibial_right': AmputationConfig(
        name='transtibial_right',
        env_id='sconewalk_h0222_osim-v1',
        subfolder='tb_right',
        action_indices=[0],
        obs_rearrange_tag='tibial_right',
        concat_tag='tibial_right',
        bilateral=False,
        agent_obs_slices=(slice(0, 45), slice(46, None)),
        n_prosthetic_joints=1,
        emg_side_slices=(slice(0, 9),),
        tibial=True
    ),
    'transfemoral_left': AmputationConfig(
        name='transfemoral_left',
        env_id='sconewalk_h0444_osim-v1',
        subfolder='tf_left',
        action_indices=[0, 1],
        obs_rearrange_tag='left',
        concat_tag='trans_left',
        bilateral=False,
        agent_obs_slices=(slice(0, 45), slice(47, None)),
        n_prosthetic_joints=2,
        emg_side_slices=(slice(9, None),)
    ),
    'transfemoral_right': AmputationConfig(
        name='transfemoral_right',
        env_id='sconewalk_h0555_osim-v1',
        subfolder='tf_right',
        action_indices=[0, 1],
        obs_rearrange_tag='right',
        concat_tag='trans_right',
        bilateral=False,
        agent_obs_slices=(slice(0, 45), slice(47, None)),
        n_prosthetic_joints=2,
        emg_side_slices=(slice(0, 9),)
    ),
    'transfemoral_both': AmputationConfig(
        name='transfemoral_both',
        env_id='sconewalk_h0333_osim-v1',
        subfolder='tf_dual',
        action_indices=[0, 1, 2, 3],
        obs_rearrange_tag='trans_both',
        concat_tag='trans_both',
        bilateral=True,
        agent_obs_slices=(slice(0, 45), slice(49, None)),
        n_prosthetic_joints=4,
        emg_side_slices=(slice(0, 9), slice(9, None))   # (right, left)
    ),
    'transtibial_both': AmputationConfig(
        name='transtibial_both',
        env_id='sconewalk_h0888_osim-v1',
        subfolder='tb_dual',
        action_indices=[0, 1],
        obs_rearrange_tag='tibial_both',
        concat_tag='tibial_both',
        bilateral=True,
        agent_obs_slices=(slice(0, 45), slice(47, None)),
        n_prosthetic_joints=2,
        emg_side_slices=(slice(0, 9), slice(9, None)),
        tibial=True
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from original, bugs fixed)
# ──────────────────────────────────────────────────────────────────────────────

def map_excitation_window(exc_window_9ch, tibial: bool = False):
    """
    Map 9-channel sconegym muscle excitations to the 13-channel EMG index space.

    Args:
        exc_window_9ch: np.array of shape (9,) or (9, n_sim_steps).
                        Channel order (sconegym leg slice):
                          0=hamstrings, 1=bifemsh, 2=glut_max, 3=iliopsoas,
                          4=rect_fem,   5=vasti,   6=gastroc,  7=soleus, 8=tib_ant
        tibial:         If True, below-knee channels (gastroc, soleus, tib_ant)
                        are populated. Set False for transfemoral (below-knee
                        muscles not present on the prosthetic side).

    Returns:
        torch.tensor of shape (13, n_sim_steps) — padded to window if 1D input.
    """
    # ── normalise to 2D (9, T) ────────────────────────────────────────────────
    if exc_window_9ch.ndim == 1:
        exc_window_9ch = exc_window_9ch[:, np.newaxis]   # (9, 1)

    n = exc_window_9ch.shape[1]
    out = torch.zeros((13, n), dtype=torch.float32)

    # above-knee — present in all configs
    out[0]  = torch.tensor(exc_window_9ch[5])  # Vastus Lateralis  ← vasti
    out[1]  = torch.tensor(exc_window_9ch[4])  # Rectus Femoris    ← rect_fem
    out[2]  = torch.tensor(exc_window_9ch[5])  # Vastus Medialis   ← vasti (lumped)
    out[4]  = torch.tensor(exc_window_9ch[1])  # Biceps Femoris    ← bifemsh
    out[5]  = torch.tensor(exc_window_9ch[0])  # Semitendinosus    ← hamstrings
    out[12] = torch.tensor(exc_window_9ch[2])  # Gluteus Maximus   ← glut_max

    # below-knee — only populated for transtibial (sound-side limb has these)
    if tibial:
        out[3]  = torch.tensor(exc_window_9ch[8])  # Tibialis Anterior ← tib_ant
        out[6]  = torch.tensor(exc_window_9ch[6])  # Gastroc Medialis  ← gastroc
        out[8]  = torch.tensor(exc_window_9ch[7])  # Soleus            ← soleus
    # indices 7 (Gastroc Lateralis), 9-11 (Peroneus L/B, Glut Med) stay 0

    return out  # (13, n_sim_steps)


def get_sagittal(impedance_values):
    sagittal_impedances = np.zeros(9,)
    counter = 0                             # BUG FIX: was `counter ==0`
    for i in range(impedance_values.shape[-1]):
        if (i + 1) % 3 == 0:
            sagittal_impedances[counter] = impedance_values[i]
            counter += 1
    return sagittal_impedances


def init_loss_dict():
    return {
        'actor_loss': [], 'q1_loss': [], 'q2_loss': [],
        'alpha_loss': [], 'log_probs': [], 'q1_mean': [], 'q2_mean': []
    }


def build_padded_emg_window(seed_emg, steps, device):
    """Replicate-pad seed_emg (13, T) to a (13, 100) window at the start of an episode."""
    pad_size = max(0, 100 - steps)
    window = F.pad(seed_emg[:, 0:steps], (pad_size, 0), mode='replicate')
    return window.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# rearrange_obs  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def rearrange_obs(obs: torch.Tensor, direction_of_control='left'):
    def expand_to_plane(joints):
        out = []
        for v in joints:
            out.extend([0.0, 0.0, v.item()])
        return out

    if direction_of_control.lower() == 'right':
        pos, vel, acc, leg = obs[3:6], obs[12:15], obs[21:24], obs[27:36]
    elif direction_of_control.lower() == 'left':
        pos, vel, acc, leg = obs[6:9], obs[15:18], obs[24:27], obs[36:45]

    elif 'tibial' in direction_of_control.lower() or 'trans' in direction_of_control.lower():
        pos_r, vel_r, acc_r, leg_r = obs[3:6],  obs[12:15], obs[21:24], obs[27:36]
        pos_l, vel_l, acc_l, leg_l = obs[6:9],  obs[15:18], obs[24:27], obs[36:45]

        dof_r = torch.tensor(expand_to_plane(pos_r) + expand_to_plane(vel_r) + expand_to_plane(acc_r), dtype=torch.float32)
        dof_l = torch.tensor(expand_to_plane(pos_l) + expand_to_plane(vel_l) + expand_to_plane(acc_l), dtype=torch.float32)

        tibial = 'tibial' in direction_of_control.lower()

        def make_emg(leg):
            below_knee = [leg[6].item(), 0.0, leg[7].item()] if tibial else [0.0, 0.0, 0.0]
            return torch.tensor([
                leg[5].item(), leg[4].item(), leg[5].item(),
                leg[8].item() if tibial else 0.0,
                leg[1].item(), leg[0].item(),
                below_knee[0], below_knee[1], below_knee[2],
                0.0, 0.0, 0.0,
                leg[2].item(),
            ], dtype=torch.float32)

        tag = direction_of_control.lower()
        if tag.endswith('_both'):
            return (dof_r, make_emg(leg_r)), (dof_l, make_emg(leg_l))
        elif tag.endswith('_left'):
            return dof_l, make_emg(leg_l)
        elif tag.endswith('_right'):
            return dof_r, make_emg(leg_r)

    dof_tensor = torch.tensor(expand_to_plane(pos) + expand_to_plane(vel) + expand_to_plane(acc), dtype=torch.float32)
    emg_tensor = torch.tensor([
        leg[5].item(), leg[4].item(), leg[5].item(), leg[8].item(),
        leg[1].item(), leg[0].item(), leg[6].item(), 0.0, leg[7].item(),
        0.0, 0.0, 0.0, leg[2].item(),
    ], dtype=torch.float32)
    return dof_tensor, emg_tensor


# ──────────────────────────────────────────────────────────────────────────────
# concatenate_actions  (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def concatenate_actions(pred_torque, muscle_action, direction):
    curr_ptr = 0

    if direction == 'right' or direction == 'left':
        full_action = np.zeros((21,))
        for i in range(pred_torque.shape[-1]):
            if (i + 1) % 3 == 0:
                full_action[curr_ptr] = pred_torque[:, i]
                curr_ptr += 1
        if direction == 'left':
            full_action[(curr_ptr + 9):] = muscle_action[9:]
        else:
            full_action[(curr_ptr):(curr_ptr + 9)] = muscle_action[:9]
        return full_action

    elif direction in ('trans_right', 'trans_left'):
        full_action = np.zeros((20,))
        for i in range(pred_torque.shape[-1]):
            if (i + 1) % 3 == 0 and i > 2:
                full_action[curr_ptr] = pred_torque[:, i]
                curr_ptr += 1
        if direction == 'trans_left':
            full_action[2:11] = muscle_action[9:]
            ZERO = [17, 18, 19]
        else:
            full_action[11:] = muscle_action[:9]
            ZERO = [8, 9, 10]
        full_action[ZERO] = 0.0
        return full_action

    elif direction.lower() in ('tibial_right', 'tibial_left'):
        full_action = np.zeros((19,))
        full_action[0] = pred_torque[:, -1]
        ZERO = [8, 9] if direction == 'tibial_left' else [17, 18]
        full_action[1:] = muscle_action
        full_action[ZERO] = 0.0
        return full_action

    elif direction.lower() == 'trans_both':
        full_action = np.zeros((22,))
        for j in range(2):
            for i in range(pred_torque[j].shape[-1]):
                if (i + 1) % 3 == 0 and i > 2:
                    full_action[curr_ptr] = pred_torque[j][:, i]
                    curr_ptr += 1
        full_action[4:] = muscle_action
        full_action[[10, 11, 12, 19, 20, 21]] = 0.0
        return full_action

    elif direction.lower() == 'tibial_both':
        full_action = np.zeros((20,))
        for j in range(2):
            for i in range(pred_torque[j].shape[-1]):
                if (i + 1) % 3 == 0 and i > 5:
                    full_action[curr_ptr] = pred_torque[j][:, i]
                    curr_ptr += 1
        full_action[2:] = muscle_action
        full_action[[8, 9, 10, 17, 18, 19]] = 0.0
        return full_action


# ──────────────────────────────────────────────────────────────────────────────
# Env + Agent setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_env_and_agent(cfg: AmputationConfig, deprl_checkpoint: str):
    env = gym.make(cfg.env_id, clip_actions=True)
    env.action_indices = cfg.action_indices

    n_skip = len(cfg.action_indices)

    trimmed_action_space = gym.spaces.Box(
        low=env.action_space.low[:-n_skip],
        high=env.action_space.high[:-n_skip],
        dtype=env.action_space.dtype
    )
    trimmed_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:-n_skip],
        high=env.observation_space.high[:-n_skip],
        dtype=env.observation_space.dtype
    )

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )
    agent.initialize(trimmed_obs_space, trimmed_action_space, seed=0)
    agent.load(deprl_checkpoint)
    print('agent loaded')

    return env, agent


# ──────────────────────────────────────────────────────────────────────────────
# Unified training loop
# ──────────────────────────────────────────────────────────────────────────────

def rl_train(
    cfg: AmputationConfig,
    prosthetic_controller,
    replay_buffer,
    Q1_b, Q2_b, Q1_m, Q2_m,
    args,
    critic_config,
    optimizers_and_schedulers,
    max_training_steps=100_000,
    max_env_steps=10_000,
    noise_cfg: Optional[NoiseConfig] = None,
):
    device = prosthetic_controller.device
    env, agent = setup_env_and_agent(cfg, args.deprl_checkpoint)
    print(f'body mass: {env.unwrapped.model.mass():.2f} | config: {cfg.name}')

    viz = TrainingVisualizer(save_dir=args.save_dir, window=200)
    training_losses = init_loss_dict()

    curr_step, episode_num = 0, 0

    while curr_step < max_training_steps:

        obs = env.reset()
        #TODO do we care to implement custom obs at reset? 
        env.unwrapped.store_next_episode()
        done, steps, episode_reward = False, 1, 0

        # ── init sides ────────────────────────────────────────────────────────
        raw = rearrange_obs(obs, cfg.obs_rearrange_tag)
        # normalize to list regardless of bilateral
        sides = list(raw) if cfg.bilateral else [raw]       # list of (dof, emg) tuples

        seed_emgs = [
            torch.zeros(13, 100, device=device) for _ in sides
        ]
        for i, (_, emg) in enumerate(sides):
            seed_emgs[i][:, 0] = emg

        emg_windows = [build_padded_emg_window(s, steps, device) for s in seed_emgs]

        # ── per-side circular buffers for rollout temporal jitter ─────────────
        # EMG frame buffer: individual (13,) frames, depth = 100 + emg_jitter_max
        #   Allows genuine window reconstruction at any δ in [0, emg_jitter_max]
        #   without padding. Initialized with 100 replicated seed frames.
        # Kin buffer: individual (27,) snapshots, depth = kin_jitter_max + 1
        #   δ=0 returns current kin; δ>0 returns a stale snapshot.
        _emg_jitter = noise_cfg.emg_jitter_max if noise_cfg is not None else 0
        _kin_jitter = noise_cfg.kin_jitter_max if noise_cfg is not None else 0

        #TODO parameterize window size
        #TODO  should the jitters not be populated interpolated pad?
        emg_frame_buffers = []
        for i, (_, emg) in enumerate(sides):
            buf = deque(maxlen=100 + max(_emg_jitter, 0))
            seed_frame = emg.cpu()                          # (13,) — first obs frame
            for _ in range(buf.maxlen):                            # replicate to fill window
                buf.append(seed_frame)
            emg_frame_buffers.append(buf)
        
        #emg_frame and kin buffers are list of queues of temporal_window+jitter

        kin_buffers = [
            deque([s[0].to(device).detach().cpu()], maxlen=max(_kin_jitter + 1, 1))
            for s in sides
        ]

        while not done and steps < max_env_steps:

            # ── update EMG windows ────────────────────────────────────────────
            if steps > 1:
                emg_windows = [
                    map_excitation_window(excitation_buffer[sl], tibial=cfg.tibial).to(device)
                    for sl in cfg.emg_side_slices
                ]
            #TODO what are emg_side_slices?

            kinematics = [s[0].to(device) for s in sides]

            # ── push CLEAN frames into circular buffers (always) ──────────────
            # EMG: push newest single frame (13,) — newest column of rolling window
            # Kin: push current snapshot (27,) — used for stale lookback if jitter>0
            for i in range(len(sides)):
                emg_frame_buffers[i].append(emg_windows[i][:, -1].detach().cpu())
                kin_buffers[i].append(kinematics[i].detach().cpu())

            # ── build noisy obs for forward pass; store CLEAN obs in replay ────
            #TODO the applying of this noise is wrong, at least for the emg
            if noise_cfg is not None and noise_cfg.noise_on_rollout:
                eff_emg_jitter, eff_kin_jitter = noise_cfg.effective_jitter(curr_step)

                emg_windows_fwd = [
                    _rollout_emg_noise(emg_frame_buffers[i], noise_cfg, eff_emg_jitter, device)
                    for i in range(len(sides))
                ]
                kinematics_fwd = [
                    _rollout_kin_noise(kinematics[i], noise_cfg, kin_buffers[i], eff_kin_jitter)
                    for i in range(len(sides))
                ]
            else:
                emg_windows_fwd = emg_windows
                kinematics_fwd  = kinematics

            # ── prosthetic forward passes (on noisy obs) ──────────────────────

            pros_actions = [
                prosthetic_controller(w, k)
                for w, k in zip(emg_windows_fwd, kinematics_fwd)
            ]

            torques = [
                compute_impedance_torque(
                    input_kin_state=k.unsqueeze(0),
                    pred_kin_state=pa['pred_kin_state'],
                    pred_impedance=pa['pred_impedance']
                )
                for k, pa in zip(kinematics_fwd, pros_actions)
            ]

            # ── agent muscle action ───────────────────────────────────────────
            agent_obs = np.concatenate([obs[cfg.agent_obs_slices[0]], obs[cfg.agent_obs_slices[1]]])
            muscle_action = agent.test_step(agent_obs, steps)

            torque_arg = torques if cfg.bilateral else torques[0]
            full_action = concatenate_actions(torque_arg, muscle_action, cfg.concat_tag)

            # ── build curr_state for replay ───────────────────────────────────
            curr_states = [
                np.concatenate([w.detach().cpu().numpy().flatten(), k.detach().cpu().numpy().flatten()])
                for w, k in zip(emg_windows, kinematics)
            ]

            # ── env step ──────────────────────────────────────────────────────
            obs, reward, done, excitation_buffer = env.step(full_action)

            # ── update sides & emg for next_state ─────────────────────────────
            raw_next = rearrange_obs(obs, cfg.obs_rearrange_tag)
            sides = list(raw_next) if cfg.bilateral else [raw_next]

            next_emg_windows = [
                map_excitation_window(excitation_buffer[sl], tibial=cfg.tibial).to(device)
                for sl in cfg.emg_side_slices
            ]

            next_states = [
                np.concatenate([w.detach().cpu().numpy().flatten(), s[0].detach().cpu().numpy().flatten()])
                for w, s in zip(next_emg_windows, sides)
            ]

            action_bufs = [
                np.concatenate([
                    pa['pred_kin_state'].detach().cpu().numpy().flatten(),
                    pa['pred_impedance'].detach().cpu().numpy().flatten()
                ])
                for pa in pros_actions
            ]

            replay_buffer.store_transition(
                state=np.concatenate(curr_states),
                action=np.concatenate(action_bufs),
                reward=float(reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else reward),
                state_=np.concatenate(next_states),
                done=bool(done)
            )

            # ── SAC update ────────────────────────────────────────────────────
            if replay_buffer.size >= args.min_replay_size:
                train_sac(
                    optimizers_and_schedulers,
                    policy_args=args,
                    critic_args=critic_config,
                    Policy=prosthetic_controller,
                    QNetwork_base1=Q1_b,
                    QNetwork_base2=Q2_b,
                    QNetwork_target1=Q1_m,
                    QNetwork_target2=Q2_m,
                    replay_buff=replay_buffer,
                    training_epochs=1,
                    training_losses=training_losses,
                    bilateral=cfg.bilateral,
                    noise_cfg=noise_cfg,
                    curr_step=curr_step,
                )

            _r = float(reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else reward)
            episode_reward += _r
            viz.log_step(_r)
            viz.log_losses(training_losses)

            emg_windows = next_emg_windows
            steps += 1

        curr_step += steps
        viz.log_episode()
        print(
            f'episode {episode_num} | steps {steps} | '
            f'total reward {episode_reward:.3f} | avg {episode_reward/steps:.4f}'
        )
        episode_num += 1
        viz.save(tag=f'episode{episode_num}_end')

        if not done:
            env.unwrapped.model.write_results(
                env.unwrapped.output_dir,
                f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
            )

        env.close()

    viz.close()
    print('training complete.')


# ──────────────────────────────────────────────────────────────────────────────
# Network / optimizer builders
# ──────────────────────────────────────────────────────────────────────────────

def build_q_network(config, device, lr, epochs):
    net = QNetwork(**config).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
    return net, opt, sch


def build_networks_and_optimizers(args, prosthetic_controller, Q_config, from_checkpoint=False):
    lr, epochs = args.lr, args.epochs
    device = args.device

    if from_checkpoint:
        path = args.sac_checkpoint_path

        policy_ckpt = torch.load(f'{path}/best_RL_transformer_model.pth')
        mc = policy_ckpt['model_config']

        policy = EMGTransformer(
            emg_channels=13, emg_window_size=100, kin_state_dim=27,
            d_model=mc['d_model'], nhead=mc['nhead'],
            num_encoder_layers=mc['num_layers'], num_decoder_layers=mc['num_layers'],
            predict_impedance=True,
            emg_mask=args.emg_mask, kinematic_mask=args.kinematic_mask
        ).to(device)
        policy.log_alpha = policy_ckpt['log_alpha'].to(device).requires_grad_(True)

        p_opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        p_opt.load_state_dict(policy_ckpt['policy_optimizer_state_dict'])
        p_sch = torch.optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=epochs, eta_min=lr / 100)
        p_sch.load_state_dict(policy_ckpt['policy_scheduler_state_dict'])

        a_opt = torch.optim.AdamW([policy.log_alpha], lr=lr, weight_decay=0.01, eps=1e-8)
        a_opt.load_state_dict(policy_ckpt['log_alpha_optimizer'])
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=epochs, eta_min=lr / 100)
        a_sch.load_state_dict(policy_ckpt['log_alpha_scheduler'])

        q_nets, q_opts, q_schs = [], [], []
        for tag in ['Q1B', 'Q2B', 'Q1T', 'Q2T']:
            ckpt = torch.load(f'{path}/{tag}')
            net = QNetwork(**ckpt['config']).to(device)
            net.load_checkpoint(f'{path}/{tag}')
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
            opt.load_state_dict(ckpt['optimizer'])
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
            sch.load_state_dict(ckpt['scheduler'])
            q_nets.append(net); q_opts.append(opt); q_schs.append(sch)

        replay_buffer = ReplayBuffer(max_size=int(1e5), input_shape=int(13 * 100 + 27), n_actions=27 * 2)
        replay_tag = (args.replay_buffer_tag if args.replay_buffer_tag is not None
                      else f'tf_{args.amputation_type}')
        replay_buffer.load(replay_tag)
        print(f'loaded replay buffer from tag: {replay_tag}')
        print(f'Loaded {len(q_nets)} Q networks')

    else:
        policy = prosthetic_controller

        q_nets, q_opts, q_schs = [], [], []
        for _ in range(4):
            net, opt, sch = build_q_network(Q_config, device, lr, epochs)
            q_nets.append(net); q_opts.append(opt); q_schs.append(sch)

        p_opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        p_sch = torch.optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=epochs, eta_min=lr / 100)
        a_opt = torch.optim.Adam([policy.log_alpha], lr=3e-4)
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=epochs, eta_min=lr / 100)

        n_sides = 2 if args.bilateral else 1
        replay_buffer = ReplayBuffer(
            max_size=int(1e5),
            input_shape=int(n_sides * (13 * 100 + 27)),
            n_actions=int(n_sides * 54),
            checkpoint_dir=os.path.join('C:/EMG/software/models/SAC', args.cfg_subfolder),
            save_name=args.amputation_type
        )

    # ── point every network at the correct subfolder ─────────────────────────
    save_dir = os.path.join('C:/EMG/software/models/SAC', args.cfg_subfolder)
    os.makedirs(save_dir, exist_ok=True)
    policy.checkpoint_dir = save_dir
    for net in q_nets:
        net.checkpoint_dir = save_dir
    replay_buffer.checkpoint_dir = save_dir

    optimizers_and_schedulers = {
        'policy':           {'optimizer': p_opt,        'scheduler': p_sch},
        'policy_log_alpha': {'optimizer': a_opt,        'scheduler': a_sch},
        'q1b':              {'optimizer': q_opts[0],    'scheduler': q_schs[0]},
        'q2b':              {'optimizer': q_opts[1],    'scheduler': q_schs[1]},
        'q1t':              {'optimizer': q_opts[2],    'scheduler': q_schs[2]},
        'q2t':              {'optimizer': q_opts[3],    'scheduler': q_schs[3]},
    }

    return policy, q_nets, replay_buffer, optimizers_and_schedulers

def build_networks_and_optimizers(args, prosthetic_controller, Q_config, from_checkpoint=False):
    lr, epochs = args.lr, args.epochs
    device = args.device
    n_sides = 2 if args.bilateral else 1

    # single source of truth for where everything lives
    save_dir = os.path.join(args.checkpoint_dir, args.amputation_type)
    os.makedirs(save_dir, exist_ok=True)

    if from_checkpoint:
        policy_ckpt = torch.load(os.path.join(save_dir, 'best_RL_transformer_model.pth'))
        mc = policy_ckpt['model_config']

        policy = EMGTransformer(
            emg_channels=13, emg_window_size=100, kin_state_dim=27,
            d_model=mc['d_model'], nhead=mc['nhead'],
            num_encoder_layers=mc['num_layers'], num_decoder_layers=mc['num_layers'],
            predict_impedance=True,
            emg_mask=args.emg_mask, kinematic_mask=args.kinematic_mask
        ).to(device)
        policy.log_alpha = policy_ckpt['log_alpha'].to(device).requires_grad_(True)

        p_opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        p_opt.load_state_dict(policy_ckpt['policy_optimizer_state_dict'])
        p_sch = torch.optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=epochs, eta_min=lr / 100)
        p_sch.load_state_dict(policy_ckpt['policy_scheduler_state_dict'])

        a_opt = torch.optim.AdamW([policy.log_alpha], lr=lr, weight_decay=0.01, eps=1e-8)
        a_opt.load_state_dict(policy_ckpt['log_alpha_optimizer'])
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=epochs, eta_min=lr / 100)
        a_sch.load_state_dict(policy_ckpt['log_alpha_scheduler'])
        print('policy + log_alpha loaded')

        q_nets, q_opts, q_schs = [], [], []
        for tag in ['Q1B', 'Q2B', 'Q1T', 'Q2T']:
            ckpt = torch.load(os.path.join(save_dir, tag))
            net = QNetwork(**ckpt['config']).to(device)
            net.load_checkpoint(os.path.join(save_dir, tag))
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
            opt.load_state_dict(ckpt['optimizer'])
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 100)
            sch.load_state_dict(ckpt['scheduler'])
            q_nets.append(net); q_opts.append(opt); q_schs.append(sch)
        print(f'loaded {len(q_nets)} Q networks')

        replay_buffer = ReplayBuffer(
            max_size=int(1e5),
            input_shape=int(n_sides * (13 * 100 + 27)),
            n_actions=int(n_sides * 54),
            checkpoint_dir=save_dir,
            save_name=args.amputation_type
        )
        replay_buffer.load()
        print(f'loaded replay buffer from: {save_dir}')

    else:
        policy = prosthetic_controller

        q_nets, q_opts, q_schs = [], [], []
        for _ in range(4):
            net, opt, sch = build_q_network(Q_config, device, lr, epochs)
            q_nets.append(net); q_opts.append(opt); q_schs.append(sch)

        p_opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        p_sch = torch.optim.lr_scheduler.CosineAnnealingLR(p_opt, T_max=epochs, eta_min=lr / 100)
        a_opt = torch.optim.Adam([policy.log_alpha], lr=3e-4)
        a_sch = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=epochs, eta_min=lr / 100)

        replay_buffer = ReplayBuffer(
            max_size=int(1e5),
            input_shape=int(n_sides * (13 * 100 + 27)),
            n_actions=int(n_sides * 54),
            checkpoint_dir=save_dir,
            save_name=args.amputation_type
        )

    # ── point every network at save_dir ──────────────────────────────────────
    policy.checkpoint_dir = save_dir
    for net in q_nets:
        net.checkpoint_dir = save_dir
    replay_buffer.checkpoint_dir = save_dir

    optimizers_and_schedulers = {
        'policy':           {'optimizer': p_opt,     'scheduler': p_sch},
        'policy_log_alpha': {'optimizer': a_opt,     'scheduler': a_sch},
        'q1b':              {'optimizer': q_opts[0], 'scheduler': q_schs[0]},
        'q2b':              {'optimizer': q_opts[1], 'scheduler': q_schs[1]},
        'q1t':              {'optimizer': q_opts[2], 'scheduler': q_schs[2]},
        'q2t':              {'optimizer': q_opts[3], 'scheduler': q_schs[3]},
    }

    return policy, q_nets, replay_buffer, optimizers_and_schedulers

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()

    # ── data / model paths ────────────────────────────────────────────────────
    parser.add_argument('--pkl_dir', type=str, default='D:/EMG/postprocessed_datasets')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default='C:/EMG/software/models/SAC',
                        help='Base directory for all checkpoints — subfoldered by amputation_type automatically')

    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume from existing checkpoint in checkpoint_dir/amputation_type')

    parser.add_argument('--replay_buffer_tag', type=str, default=None,
                        help='Tag passed to ReplayBuffer.load() when resuming from a SAC checkpoint. '
                             'Defaults to amputation_type when sac_checkpoint_path is set.')
    parser.add_argument('--deprl_checkpoint', type=str,
                        default='C:/Users/vijay/OneDrive/Documents/SCONE/results/'
                                'sconewalk_h0918_osimv1/260220.191743.H0918v2/checkpoints/step_12000000')
    parser.add_argument('--save_dir', type=str, default='C:/EMG/software/plots/SAC')

    # ── environment ───────────────────────────────────────────────────────────
    parser.add_argument('--amputation_type', type=str,
                        choices=list(AMPUTATION_CONFIGS.keys()),
                        default='transfemoral_both',
                        help='Amputation type and side to train')

    # ── training hyperparams ──────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_training_steps', type=int, default=10000)
    parser.add_argument('--max_env_steps', type=int, default=1000)
    parser.add_argument('--min_replay_size', type=int, default=200,
                        help='Minimum buffer size before SAC updates begin')

    # ── model architecture ────────────────────────────────────────────────────
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # ── sim-to-real noise ─────────────────────────────────────────────────────
    # All magnitude params are UPPER BOUNDS — actual noise is domain-randomized
    # per draw so the policy learns robustness across the full [0, max] range.
    # Per-channel/dim sampling: each EMG channel and each kin dim independently
    # draws its own (μ, std) each forward pass / batch element.
    parser.add_argument('--emg_noise_std_max', type=float, default=1.0,
                        help='Per-channel EMG std upper bound: std_c ~ U[0, std_max]')
    parser.add_argument('--emg_noise_mean_max', type=float, default=0.0,
                        help='Per-channel EMG μ upper bound: μ_c ~ U[-mean_max, mean_max]')
    parser.add_argument('--kin_noise_std_max', type=float, default=1.0,
                        help='Per-dim kin std upper bound: std_d ~ U[0, std_max]')
    parser.add_argument('--kin_noise_mean_max', type=float, default=0.0,
                        help='Per-dim kin μ upper bound: μ_d ~ U[-mean_max, mean_max]')
    parser.add_argument('--emg_jitter_max', type=int, default=200,
                        help='EMG temporal jitter upper bound in env steps '
                             '(δ ~ U[0, max]; window slid back, never forward)')
    parser.add_argument('--kin_jitter_max', type=int, default=5,
                        help='Kin temporal jitter upper bound in env steps '
                             '(δ ~ U[0, max]; sampled independently from EMG jitter)')
    parser.add_argument('--jitter_warmup_steps', type=int, default=0,
                        help='Linearly ramp both jitter_max values 0 → max over N steps (0 = off)')
    parser.add_argument('--noise_on_rollout', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply signal noise + temporal jitter at rollout act time. '
                             'Clean observations are always stored regardless.')
    parser.add_argument('--noise_on_replay', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply signal noise + temporal jitter at SAC sample time. '
                             'Requires NoisyReplayBuffer (automatic when any noise param > 0).')

    args = parser.parse_args()

    cfg = AMPUTATION_CONFIGS[args.amputation_type]
    args.bilateral    = cfg.bilateral   # expose to build_networks_and_optimizers
    args.cfg_subfolder = cfg.subfolder  # expose subfolder to build_networks_and_optimizers

    # ── masks (keyed by amputation type) ─────────────────────────────────────
    emg_masks = {
        'transtibial_left':   np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]),
        'transtibial_right':  np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]),
        'transfemoral_left':  np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
        'transfemoral_right': np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
        'transfemoral_both':  np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1]),
        'transtibial_both':   np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]),
    }
    kin_masks = {
        'transtibial_left':   np.array([[0,0,0],[0,0,0],[0,0,1]]),
        'transtibial_right':  np.array([[0,0,0],[0,0,0],[0,0,1]]),
        'transfemoral_left':  np.array([[0,0,0],[0,0,1],[0,0,1]]),
        'transfemoral_right': np.array([[0,0,0],[0,0,1],[0,0,1]]),
        'transfemoral_both':  np.array([[0,0,0],[0,0,1],[0,0,1]]),
        'transtibial_both':   np.array([[0,0,0],[0,0,0],[0,0,1]]),
    }

    args.emg_mask = emg_masks[args.amputation_type]
    args.kinematic_mask = kin_masks[args.amputation_type]

    # ── build controller ──────────────────────────────────────────────────────
    # If a checkpoint is provided, read its stored model_config so the
    # architecture is guaranteed to match — args d_model/nhead/num_layers are
    # only used as fallback when starting from scratch.
    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location=args.device)
        mc   = ckpt.get('model_config', {})
        d_model    = mc.get('d_model',     args.d_model)
        nhead      = mc.get('nhead',       args.nhead)
        num_layers = mc.get('num_layers',  args.num_layers)
        if mc:
            print(f'checkpoint model_config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}')
        else:
            print('WARNING: checkpoint has no model_config — falling back to CLI args. '
                  'Architecture mismatch will cause a RuntimeError.')
    else:
        ckpt       = None
        d_model    = args.d_model
        nhead      = args.nhead
        num_layers = args.num_layers

    prosthetic_controller = EMGTransformer(
        emg_channels=13, emg_window_size=100, kin_state_dim=27,
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_layers, num_decoder_layers=num_layers,
        predict_impedance=True,
        emg_mask=args.emg_mask, kinematic_mask=args.kinematic_mask
    ).to(args.device)

    if ckpt is not None:
        missing, unexpected = prosthetic_controller.load_state_dict(
            ckpt['model_state_dict'], strict=False
        )
        new_heads = [k for k in missing if 'log_std' in k]
        real_missing = [k for k in missing if 'log_std' not in k]
        if new_heads:
            print(f'log_std heads not in checkpoint ({len(new_heads)} keys) — '
                  f'initialising fresh. This is expected when loading a pre-SAC checkpoint.')
        if real_missing:
            print(f'WARNING: unexpected missing keys: {real_missing}')
        if unexpected:
            print(f'WARNING: {len(unexpected)} unexpected keys in checkpoint '
                  f'(architecture larger than current model?) — first few: {unexpected[:3]}')
        print(f'loaded controller from {args.checkpoint_path}')

    #prosthetic_controller.eval()

    Q_config = {
        'h_dim': 512, 'num_bins': 54,
        'emg_channels': 13, 'emg_window_size': 100,
        'kin_state_dim': 27, 'action_dim': 54,
        'd_model': 50, 'nhead': 2,
        'num_encoder_layers': 1, 'num_decoder_layers': 1,
        'dim_feedforward': 1024, 'dropout': 0.1
    }

    policy, q_nets, replay_buffer, optimizers_and_schedulers = build_networks_and_optimizers(
        args, prosthetic_controller, Q_config,
        from_checkpoint=args.resume
    )

    noise_cfg = NoiseConfig(
        emg_noise_std_max    = args.emg_noise_std_max,
        emg_noise_mean_max   = args.emg_noise_mean_max,
        kin_noise_std_max    = args.kin_noise_std_max,
        kin_noise_mean_max   = args.kin_noise_mean_max,
        emg_jitter_max       = args.emg_jitter_max,
        kin_jitter_max       = args.kin_jitter_max,
        jitter_warmup_steps  = args.jitter_warmup_steps,
        noise_on_rollout     = args.noise_on_rollout,
        noise_on_replay      = args.noise_on_replay,
    )
    print(f'noise config: {noise_cfg}')

    # Upgrade replay buffer to NoisyReplayBuffer if replay noise is requested
    # so that sample_with_jitter is available in train_sac.
    if noise_cfg.noise_on_replay and not isinstance(replay_buffer, NoisyReplayBuffer):
        replay_buffer.__class__ = NoisyReplayBuffer
        print('replay buffer upgraded to NoisyReplayBuffer')

    rl_train(
        cfg=cfg,
        prosthetic_controller=policy,
        replay_buffer=replay_buffer,
        Q1_b=q_nets[0], Q2_b=q_nets[1],
        Q1_m=q_nets[2], Q2_m=q_nets[3],
        args=args,
        critic_config=Q_config,
        optimizers_and_schedulers=optimizers_and_schedulers,
        max_training_steps=args.max_training_steps,
        max_env_steps=args.max_env_steps,
        noise_cfg=noise_cfg,
    )


if __name__ == '__main__':
    main()