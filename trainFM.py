import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
# from diffusers import DDPMScheduler
# from diffusers.models import UNet1DModel
# from diffusers.optimization import get_cosine_schedule_with_warmup
# from scipy.signal import welch
import math
from tqdm import tqdm
import time
from convert2DL import SplitDataset, TemperatureScaledStreamer, _configure_noise, safe_collate
import gc
import math
import random
import os
import logging
from datetime import datetime
import re
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import contextlib
from visualizer import create_plots,plot_test_data

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def soft_update(source, target, tau):
    """
    Soft update of the target network parameters.
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

# ── Noise Configuration ────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
class ValueNetwork(nn.Module):
    def __init__(self,
                 h_dim,
                 emg_channels=13,
                 emg_window_size=100,
                 kin_state_dim=27,
                 d_model=50,
                 nhead=2,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=1024,
                 dropout=0.1,
                 device='cuda'):
        super().__init__()

        self.device = device
        self.emg_channels = emg_channels
        self.emg_window_size = emg_window_size
        self.kin_state_dim = kin_state_dim
        self.d_model = d_model
        self.emg_conv_ip_channels = 16
        self.emg_conv_hidden_channels = 32

        # ── EMG Encoder ───────────────────────────────────────────────────────
        # Input:  [B, 13, 100]
        # Output: [B, 13, d_model]  ← encoder src tokens
        self.state_conv = nn.Sequential(
            nn.Conv1d(emg_channels, self.emg_conv_ip_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_ip_channels, emg_window_size]),
            nn.LeakyReLU(),
            nn.Conv1d(self.emg_conv_ip_channels, self.emg_conv_hidden_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_hidden_channels, emg_window_size]),
            nn.LeakyReLU(),
            nn.Conv1d(self.emg_conv_hidden_channels, emg_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Linear(emg_window_size, d_model),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # ── Kinematic Query ───────────────────────────────────────────────────
        # Projects [B, 27] → [B, 1, d_model] to use as the decoder tgt.
        # Kin cross-attends into EMG context, analogous to action in QNetwork.
        self.kin_embedding = nn.Sequential(
            nn.Linear(kin_state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        # ── Positional Encoding ───────────────────────────────────────────────
        # Encoder src is 13 EMG tokens only (kin is now the decoder query)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=emg_channels)

        # ── Transformer ───────────────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # ── Output Head ───────────────────────────────────────────────────────
        # [B, 1, d_model] → squeeze → [B, d_model] → scalar V(s)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 1)
        )

        self.replay_buffer = ReplayBuffer(
            max_size=int(1e6),
            input_shape=int(13 * 100 + 27),
            n_actions=0
        )

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, emg, kin):
        """
        Args:
            emg: [B, 1300]  — flattened 13×100 EMG window  (encoder src)
            kin: [B, 27]    — kinematic state               (decoder tgt/query)

        Returns:
            value: [B, 1]   — scalar state value V(s)
        """
        emg = emg.to(self.device)
        kin = kin.to(self.device)
        B = emg.shape[0]

        # ── 1. EMG tokens → encoder src ───────────────────────────────────────
        emg = emg.view(B, self.emg_channels, self.emg_window_size)  # [B, 13, 100]
        emg_tokens = self.state_conv(emg)                            # [B, 13, d_model]
        emg_tokens = self.pos_encoder(emg_tokens)

        # ── 2. Kin token → decoder query ──────────────────────────────────────
        kin_query = self.kin_embedding(kin).unsqueeze(1)             # [B,  1, d_model]

        # ── 3. Transformer: kin cross-attends into EMG context ────────────────
        out = self.transformer(
            src=emg_tokens,   # [B, 13, d_model]
            tgt=kin_query     # [B,  1, d_model]
        )                     # [B,  1, d_model]

        # ── 4. Value scalar ───────────────────────────────────────────────────
        value = self.output_head(out.squeeze(1))                     # [B, 1]
        return value

    def save_checkpoint(self, name, arg, optimizer, scheduler):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = f'{self.checkpoint_dir}/{name}'
        torch.save({
            'state_dict': self.state_dict(),
            'config': arg,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, self.checkpoint_file)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path)['state_dict'])


class QNetwork(nn.Module):
    def __init__(self,
                 h_dim,
                 num_bins=1,              # two-hot support bins
                 emg_channels=13,
                 emg_window_size=100,
                 kin_state_dim=27,
                 action_dim=54,
                 d_model=50,
                 nhead=2,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=1024,
                 dropout=0.1,
                 device='cuda'):
        super().__init__()

        self.device = device
        self.emg_channels = emg_channels
        self.emg_window_size = emg_window_size
        self.kin_state_dim = kin_state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.num_bins = num_bins
        self.emg_conv_ip_channels = 16
        self.emg_conv_hidden_channels = 32

        # ── EMG Encoder ───────────────────────────────────────────────────────
        # Input:  [B, 13, 100]
        # Output: [B, 13, d_model]  ← 13 tokens, one per channel
        self.state_conv = nn.Sequential(
            nn.Conv1d(emg_channels, self.emg_conv_ip_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_ip_channels, emg_window_size]),
            nn.LeakyReLU(),
            nn.Conv1d(self.emg_conv_ip_channels, self.emg_conv_hidden_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_hidden_channels, emg_window_size]),
            nn.LeakyReLU(),
            nn.Conv1d(self.emg_conv_hidden_channels, emg_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Linear(emg_window_size, d_model),  # project time → d_model
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # ── Kinematic Token ───────────────────────────────────────────────────
        # Projects [B, 27] → [B, 1, d_model] to prepend as 14th encoder token
        self.kin_embedding = nn.Sequential(
            nn.Linear(kin_state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        # ── Action Query ──────────────────────────────────────────────────────
        # Projects [B, 54] → [B, 1, d_model] to use as decoder query
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

        # ── Positional Encoding ───────────────────────────────────────────────
        # Encoder seq_len = 13 EMG tokens + 1 kin token = 14
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=emg_channels + 1)

        # ── Transformer ───────────────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # ── Output Head ───────────────────────────────────────────────────────
        # [B, 1, d_model] → [B, num_bins]  for two-hot CE loss
        self.output_head = nn.Sequential(
            nn.Linear(d_model, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, 1)
        )

        self.replay_buffer = ReplayBuffer(
            max_size=int(1e6),
            input_shape=int(13 * 100 + 27),
            n_actions=27 * 2
        )

        self._init_weights()
        self.to(device)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, emg,kin,action):
        action = action.to(self.device)
        emg=emg.to(self.device)
        kin=kin.to(self.device)
        """
        Args:
            state:  [B, 1327]  — flattened (EMG: 13×100=1300) + (kin: 27)
            action: [B, 54]    — action vector to query Q-value for

        Returns:
            logits: [B, num_bins] — two-hot distribution over symlog support
        """
        B = action.shape[0]

        emg = emg.view(B, self.emg_channels, self.emg_window_size)  # [B, 13, 100]
        emg_tokens = self.state_conv(emg)                  # [B, 13, d_model]
        kin_token = self.kin_embedding(kin).unsqueeze(1)  # [B, 1, d_model]

        encoder_input = torch.cat([kin_token, emg_tokens], dim=1)  # [B, 14, d_model]
        encoder_input = self.pos_encoder(encoder_input)

        action_query = self.action_embedding(action).unsqueeze(1)  # [B, 1, d_model]

        out = self.transformer(
            src=encoder_input,   # [B, 14, d_model]
            tgt=action_query     # [B,  1, d_model]
        )                        # [B,  1, d_model]

        # ── 7. Two-hot logits ─────────────────────────────────────────────────
        logits = self.output_head(out.squeeze(1))          # [B, num_bins]
        return logits
        
    def save_checkpoint(self,name,arg,optimizer,scheduler):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file=f'{self.checkpoint_dir}/{name}'
        torch.save({'state_dict':self.state_dict(),
                    'config': arg,
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler.state_dict()},self.checkpoint_file)

    def load_checkpoint(self,path):
        self.load_state_dict(torch.load(path)['state_dict'])

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions,checkpoint_dir='C:/EMG/software/models/SAC',save_name='default',num_workers:int=1):
        self.mem_size = max_size
        self.ptr = 0  # Current position to write
        self.size = 0  # Current buffer size
        self.num_workers = num_workers

        # Pre-allocate memory with float32 for efficiency
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_name = save_name
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)

        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        if(np.isnan(state).any() or np.isnan(action).any() or
        np.isnan(reward) or np.isnan(state_).any()):
            print("nan detected, outputting none")

            return 

        index = self.ptr
        self.state_memory[index] = state
        self.new_state_memory[index] = state_  
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.ptr = (self.ptr + 1) % self.mem_size
        self.size = min(self.size + 1, self.mem_size)

    def sample_buffer(self, batch_size):
        # Handle edge case where buffer has fewer samples than batch_size
        max_mem = min(self.size, self.mem_size)
        assert max_mem > 0, "Buffer is empty!"
        batch_size = min(batch_size, max_mem)  # Ensure we don't over-sample
        batch = np.random.choice(max_mem, batch_size, replace=(max_mem < batch_size))
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, states_, actions, rewards, dones
    
    def save(self, save_name=None):
        name = save_name or self.save_name
        os.makedirs(self.checkpoint_dir, exist_ok=True)          # fix 1: valid call + won't crash if dir exists
        path = os.path.join(self.checkpoint_dir, f'{name}_replay_buffer.npz')
        tmp  = path + '.tmp'                                     # fix 2: tmp = ...replay_buffer.npz.tmp
        np.savez_compressed(tmp,                                 # numpy writes ...replay_buffer.npz.tmp.npz
            state_memory     = self.state_memory,
            action_memory    = self.action_memory,
            reward_memory    = self.reward_memory,
            new_state_memory = self.new_state_memory,
            terminal_memory  = self.terminal_memory,
            ptr              = self.ptr,
            size             = self.size,
        )
        os.replace(tmp + '.npz', path)                           # matches what numpy actually wrote

    def load(self, save_name=None):
        name = save_name or self.save_name
        path = os.path.join(self.checkpoint_dir, f'{name}_replay_buffer.npz')  # .npy → .npz
        tmp  = path + '.tmp'                                                     # match new tmp pattern
        if os.path.exists(tmp + '.npz'):
            os.remove(tmp + '.npz')                                              # clean up orphaned tmp if present
        data = np.load(path, allow_pickle=True)

        if not hasattr(data, 'files'):
            data = data.item()

        self.state_memory     = data['state_memory']
        self.action_memory    = data['action_memory']
        self.reward_memory    = data['reward_memory']
        self.new_state_memory = data['new_state_memory']
        self.terminal_memory  = data['terminal_memory']
        self.ptr              = int(data['ptr'])
        self.size             = int(data['size'])
    
class EMGTransformer(nn.Module):
    """
    Transformer model for EMG-based gait prediction.
    Processes EMG windows + kinematic state to predict next kinematic state.
    """
    
    def __init__(self, 
                 emg_channels=13,
                 emg_window_size=100,
                 kin_state_dim=27,  # 9 angles + 9 omega + 9 alpha
                 d_model=50,
                 nhead=2,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=1024,
                 dropout=0.1,
                 predict_impedance=True,
                 kinematic_mask=np.zeros((3,3)),
                 kinetic_mask=None,
                 emg_mask=np.zeros(13,),
                 device='cuda',
                 save_dir=None):
        super().__init__()

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.emg_channels = emg_channels
        self.emg_window_size = emg_window_size
        self.kin_state_dim = kin_state_dim
        self.d_model = d_model
        self.emg_conv_ip_channels = 16
        self.emg_conv_hidden_channels = 32
        self.device = device
        self.predict_impedance = predict_impedance

        # Convert masks to tensors
        self.emg_mask = torch.Tensor(emg_mask).float().to(device)
        self.kinematic_mask = torch.Tensor(np.tile(kinematic_mask.flatten(), 3)).float().to(device)
        self.log_mask = self.kinematic_mask
        if kinetic_mask is not None and kinetic_mask.any():
            self.kinetic_mask = torch.Tensor(kinetic_mask.flatten()).float().to(device)
        else:
            self.kinetic_mask = torch.Tensor(np.zeros((9))).float().to(device)
        
        self.emg_conv = nn.Sequential(
            nn.Conv1d(self.emg_channels, self.emg_conv_ip_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_ip_channels, emg_window_size]),  # Add normalization
            nn.LeakyReLU(),
            nn.Conv1d(self.emg_conv_ip_channels, self.emg_conv_hidden_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_hidden_channels, emg_window_size]),  # Add normalization
            nn.LeakyReLU(),
            nn.Conv1d(self.emg_conv_hidden_channels, self.emg_channels, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            nn.Linear(self.emg_window_size, d_model),
            nn.Tanh(),  # Changed from Sigmoid to Tanh for better gradients
            nn.Dropout(dropout)
        )
        
        # Calculate sequence length after convolutions
        self.emg_seq_len = emg_window_size // 4
        
        # Kinematic state embedding
        self.kin_embedding = nn.Sequential(
            nn.Linear(kin_state_dim, d_model),
            nn.LayerNorm(d_model),  # Add normalization
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Gait percentage embedding
        self.gait_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),  # Add normalization
            nn.LeakyReLU(),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.emg_seq_len + 2)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads with normalization
        self.kin_output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, kin_state_dim)
        )

        self.kin_output_log_std = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, kin_state_dim)
        )

        self.kin_min = torch.full((27,), -2.0 * math.pi, device=self.device)
        self.kin_max = torch.full((27,),  2.0 * math.pi, device=self.device)
        self.kin_range = self.kin_max - self.kin_min

        self.log_alpha = torch.tensor(-4.0, requires_grad=True, device=device)
        active_kin_dims = self.kinematic_mask.sum()
        total_active_dims = int(active_kin_dims * 2) # * 2 for kinematics + impedance
        self.target_entropy = -float(total_active_dims) * 4
        print(f"Dynamic Target Entropy set to: {self.target_entropy}")

        self.gait_output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            #nn.Sigmoid()  # Gait percentage should be 0-1
        )
        
        if predict_impedance:

            self.max_stiffness = 600.0   # The hardest the human body can push
            self.max_damping = 10.0      # The thickest shock absorber needed
            self.max_inertia = 0.2       # The max virtual mass

            # INDEX ORDER (x9): 
            # [Hip_roll, Hip_yaw, Hip_pitch, Knee_roll, Knee_yaw, Knee_pitch, Ankle_roll, Ankle_yaw, Ankle_pitch]

            # 1. STIFFNESS (K) BOUNDS - Nm/rad
            # Roll and Yaw are mathematically executed (Max = 0.0). Pitch gets realistic sagittal walking limits.
            k_min = [0.0] * 9
            k_max = [
                0.0,   0.0, 250.0,  # Hip: Pitch only
                0.0,   0.0, 150.0,  # Knee: Pitch only
                0.0,   0.0, 300.0   # Ankle: Pitch only
            ]

            # 2. DAMPING (D) BOUNDS - Nms/rad
            # Same here. Kill the damping on Roll/Yaw so the network doesn't waste time tuning it.
            d_min = [
                0.0,   0.0,   0.1,
                0.0,   0.0,   0.1,
                0.0,   0.0,   0.1
            ]
            d_max = [
                0.0,   0.0,   5.0,
                0.0,   0.0,   5.0,
                0.0,   0.0,   5.0
            ]

            # 3. VIRTUAL MASS (M) BOUNDS - kg
            # Keep the 0.2 cap, but only allow it on the active sagittal joints.
            m_min = [
                0.0,   0.0, 0.001,
                0.0,   0.0, 0.001,
                0.0,   0.0, 0.001
            ]
            m_max = [
                0.0,   0.0,   0.2,
                0.0,   0.0,   0.2,
                0.0,   0.0,   0.2
            ]

            self.imp_min = torch.tensor(k_min + d_min + m_min, device=self.device)
            self.imp_max = torch.tensor(k_max + d_max + m_max, device=self.device)
            self.imp_range = self.imp_max - self.imp_min

            self.impedance_output = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 27),
            )

            self.impedance_output_log_std = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 27),
            )
   
        # self.checkpoint_dir = 'C:/EMG/software/models/SAC'
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def save_checkpoint(self,optimizer,scheduler,args,best_val_loss,epoch,alpha_optimizer=None,alpha_scheduler=None):

        save_dict = {
            'model_config': {'num_layers':args.num_layers,'d_model':args.d_model,'nhead':args.nhead},
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'policy_optimizer_state_dict': optimizer.state_dict(),
            'policy_scheduler_state_dict':scheduler.state_dict(),
            'val_loss': best_val_loss,
        }
        if alpha_optimizer != None and alpha_scheduler != None:
            save_dict['log_alpha_scheduler']=alpha_scheduler.state_dict()
            save_dict['log_alpha']=self.log_alpha.detach().cpu()
            save_dict['log_alpha_optimizer']=alpha_optimizer.state_dict()
        torch.save(save_dict, os.path.join(self.checkpoint_dir, 'best_RL_transformer_model.pth'))

    def check_and_save_checkpoints(self,model, optimizer, scheduler, args,
                                    curr_eval_dataset_losses,
                                    overall_eval_dataset_losses,
                                    overall_best_ceiling_losses,
                                    outer_epoch, logger):

        torque_datasets = ['k2muse', 'moreira','lencioni','k2muse','moghadam','siat']

        # compute current ceilings
        curr_kin_ceiling = max(
            curr_eval_dataset_losses[d]['dataset_total_kinematic_loss']
            for d in curr_eval_dataset_losses
        )
        curr_avg_ceiling = max(
            curr_eval_dataset_losses[d]['dataset_total_avg_loss']
            for d in curr_eval_dataset_losses
        )

        # torque ceiling only computed if any torque datasets were evaluated
        torque_vals = [
            curr_eval_dataset_losses[d]['dataset_total_torque_loss']
            for d in torque_datasets
            if d in curr_eval_dataset_losses
            and curr_eval_dataset_losses[d]['dataset_total_torque_loss'] is not None
        ]
        curr_torque_ceiling = max(torque_vals) if torque_vals else None

        base_save = {
            'model_config': {
                'num_layers': args.num_layers,
                'd_model': args.d_model,
                'nhead': args.nhead
            },
            'outer_epoch': outer_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'overall_eval_dataset_losses': overall_eval_dataset_losses,
            'overall_best_ceiling_losses': overall_best_ceiling_losses,
        }

        if curr_kin_ceiling < overall_best_ceiling_losses['kinematic']:
            overall_best_ceiling_losses['kinematic'] = curr_kin_ceiling
            torch.save(base_save, os.path.join(self.save_dir,'best_kinematic1.pth'))
            logger.info(f'Saved best_kinematic.pth | ceiling: {curr_kin_ceiling:.4f}')

        if curr_torque_ceiling is not None and curr_torque_ceiling < overall_best_ceiling_losses['torque']:
            overall_best_ceiling_losses['torque'] = curr_torque_ceiling
            torch.save(base_save, os.path.join(self.save_dir,'best_torque1.pth'))
            logger.info(f'Saved best_torque.pth | ceiling: {curr_torque_ceiling:.4f}')

        if curr_avg_ceiling < overall_best_ceiling_losses['avg']:
            overall_best_ceiling_losses['avg'] = curr_avg_ceiling
            torch.save(base_save, os.path.join(self.save_dir,'best_avg1.pth'))
            logger.info(f'Saved best_avg.pth | ceiling: {curr_avg_ceiling:.4f}')

        # update overall history with current epoch results
        for d in curr_eval_dataset_losses:
            for metric in curr_eval_dataset_losses[d]:
                overall_eval_dataset_losses[d][metric] = curr_eval_dataset_losses[d][metric]

        return overall_eval_dataset_losses, overall_best_ceiling_losses
        
    def forward(self, emg, input_kin_state, input_gait_pct=None, 
                emg_mask=None, kinematic_mask=None,sample=False,rparam=False):

        if emg_mask == None:
            emg_mask = self.emg_mask
        if kinematic_mask == None:
            kinematic_mask = self.kinematic_mask
        
        outputs = {}
        
        """
        Args:
            emg: (batch, emg_channels, emg_window_size)
            input_kin_state: (batch, 27) - current angles, omega, alpha
            input_gait_pct: (batch, 1) - current gait percentage
        
        Returns:
            Dictionary with predictions
        """
        if emg_mask is not None:
            emg_masked = emg * emg_mask.view(-1, self.emg_channels, 1)
        
        # Process EMG        
        emg_features = self.emg_conv(emg_masked) * emg_mask.view(-1, self.emg_channels, 1)


        # Process kinematic state and gait
        if kinematic_mask.shape[-1] == 3: 
            kinematic_mask = torch.Tensor(np.tile(kinematic_mask.flatten(), 3)).float().to(self.device)
            
        kin_masked = input_kin_state * kinematic_mask

        kin_features = self.kin_embedding(kin_masked.unsqueeze(1))  # (batch, 1, d_model)
        
        if input_gait_pct is not None:
            gait_features = self.gait_embedding(input_gait_pct.unsqueeze(1))  # (batch, 1, d_model)

        # Combine into encoder input sequence
        encoder_input = emg_features
        encoder_input = self.pos_encoder(encoder_input)
        
        # Create decoder input
        if input_gait_pct is not None:
            decoder_input = torch.cat([kin_features, gait_features], dim=1)
        else: 
            decoder_input = kin_features
        
        # Transformer
        transformer_output = self.transformer(encoder_input, decoder_input)
        
        # ─── KINEMATIC PREDICTIONS ──────────────────────────────────────────────
        pred_kin_state = self.kin_output(transformer_output[:, 0, :])
        if sample:
            pred_kin_state_log_std = self.kin_output_log_std(transformer_output[:, 0, :])
            clamped_log_std = torch.clamp(pred_kin_state_log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(clamped_log_std)
            kin_dist = torch.distributions.Normal(pred_kin_state, std)
            
            # 1. Get raw sample
            if rparam:
                kin_sample_raw = kin_dist.rsample()
            else: 
                kin_sample_raw=kin_dist.sample()
            # 2. Squash it (MUST use tanh to prevent NaN in Jacobian)
            kin_sample = torch.tanh(kin_sample_raw)
            kin_normalized_mean = (kin_sample + 1.0) / 2.0
            outputs['pred_kin_state'] = (kin_normalized_mean * self.kin_range) + self.kin_min
            
            # 3. Calculate log_prob on the RAW sample
            kin_log_prob_raw = kin_dist.log_prob(kin_sample_raw)
            
            # 4. Apply Jacobian correction
            kin_correction = torch.log(self.kin_range / 2.0) + torch.log(1.0 - kin_sample.pow(2) + 1e-6)            
            # 5. Mask and sum
            kin_log_prob = ((kin_log_prob_raw - kin_correction) * self.log_mask.unsqueeze(dim=0)).sum(dim=-1, keepdim=True)

            outputs['pred_kin_log_pdf'] = kin_log_prob
        else:
             # Deterministic squash
            kin_normalized_mean = (torch.tanh(pred_kin_state) + 1.0) / 2.0
            outputs['pred_kin_state'] = (kin_normalized_mean * self.kin_range) + self.kin_min

        if input_gait_pct is not None:
            outputs['pred_gait_pct'] = self.gait_output(transformer_output[:, 1, :])

        # ─── IMPEDANCE PREDICTIONS ──────────────────────────────────────────────
        if self.predict_impedance:
            # 1. Define the 27D Biometric Bounds (9 K, 9 B, 9 I)
            # You can move this to __init__ to save a tiny bit of compute!
            
            pred_impedance = self.impedance_output(transformer_output[:, 0, :])

            if sample: 
                pred_impedance_log_std = self.impedance_output_log_std(transformer_output[:, 0, :])
                clamped_log_std = torch.clamp(pred_impedance_log_std, LOG_STD_MIN, LOG_STD_MAX)
                std = torch.exp(clamped_log_std)
                pred_impedance_dist = torch.distributions.Normal(pred_impedance, std)
                
                # 1. Get the raw sample
                pred_impedance_sample_raw = pred_impedance_dist.rsample()
                
                # 2. Squash to [-1, 1]
                imp_squashed = torch.tanh(pred_impedance_sample_raw)
                
                # 3. Shift to [0, 1] and scale to [Min, Max] boundaries
                imp_normalized = (imp_squashed + 1.0) / 2.0
                pred_impedance_sample = (imp_normalized * self.imp_range) + self.imp_min
                
                # 4. Calculate log_prob on the RAW sample
                imp_log_prob_raw = pred_impedance_dist.log_prob(pred_impedance_sample_raw)
                
                # 5. Apply the Jacobian correction (accounting for shift AND scale)
                #NOTE scaling for impedance dist was taken away due for normal representation about a_t
                scale_factor = 1 / 2.0
                imp_correction = torch.log(scale_factor * (1.0 - imp_squashed.pow(2)) + 1e-6)
                
                # 6. Mask and sum
                pred_imp_log_pdf = ((imp_log_prob_raw - imp_correction) * self.log_mask.unsqueeze(dim=0)).sum(dim=-1, keepdim=True)
                
                outputs['pred_impedance_nm'] = pred_impedance_sample
                outputs['pred_impedance_unscaled'] = imp_normalized
                outputs['pred_impedance_log_pdf'] = pred_imp_log_pdf
            else: 
                # Deterministic path: Squash, shift, and scale the mean
                imp_squashed_mean = torch.tanh(pred_impedance)
                imp_normalized_mean = (imp_squashed_mean + 1.0) / 2.0
                outputs['pred_impedance_nm'] = (imp_normalized_mean * self.imp_range) + self.imp_min

        return outputs

    def masked_mse_loss(self,pred, target, mask):
        """
        Compute MSE loss only for masked (available) dimensions.
        
        Args:
            pred: (batch, dim) - predictions
            target: (batch, dim) - ground truth
            mask: (dim,) - binary mask indicating available dimensions
        
        Returns:
            loss: scalar - mean squared error over available dimensions only
        """
        # Apply mask to both prediction and target
        if mask.shape[-1] == 3:
            mask = mask.view(mask.size(0), -1)

        try:
            pred_masked = pred * mask.unsqueeze(0)
        except Exception as e:
            
            print('shape',mask.shape, pred.shape)

        target_masked = target * mask.unsqueeze(0)
        
        # Compute squared error
        squared_error = (pred_masked - target_masked) ** 2
        
        # Sum over available dimensions and average over batch
        # Only count non-zero mask elements in the denominator
        n_available = mask.sum()
        if n_available == 0:
            return torch.tensor(0.0, device=pred.device)
        
        loss = squared_error.sum() / (pred.size(0) * n_available)
        return loss


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
def parse_state(states,actions=None):

    B = states.shape[0]
    emg_L = states[:, :1300].reshape(B, 13, 100)
    kin_L = states[:, 1300:1327]

    if actions is not None:
        action_R = actions[:, :54]
        action_L = actions[:, 54:]

        return emg_L, kin_L, action_L, action_R

    return emg_L, kin_L

def parse_bilateral_state(states, actions=None):
    """
    Split flattened bilateral state and optionally actions.
    
    Args:
        states:  [B, 2654] — [emg_L(1300) | kin_L(27) | emg_R(1300) | kin_R(27)]
        actions: [B, 108]  — [action_R(54) | action_L(54)] optional

    Returns:
        emg_L, kin_L, emg_R, kin_R, (action_L, action_R if actions provided)
    """
    B = states.shape[0]
    emg_L = states[:, :1300].reshape(B, 13, 100)
    kin_L = states[:, 1300:1327]
    emg_R = states[:, 1327:2627].reshape(B, 13, 100)
    kin_R = states[:, 2627:2654]


    if actions is not None:
        action_R = actions[:, :54]
        action_L = actions[:, 54:]

        return emg_L, kin_L, emg_R, kin_R, action_L, action_R

    return emg_L, kin_L, emg_R, kin_R

def compute_impedance_torque(input_kin_state, pred_kin_state, pred_impedance):
    """Compute predicted torque using impedance control formula."""
    theta_curr = input_kin_state[:, :9]
    omega_curr = input_kin_state[:, 9:18]
    alpha_curr = input_kin_state[:, 18:27]
    
    theta_des = pred_kin_state[:, :9]
    omega_des = pred_kin_state[:, 9:18]
    alpha_des = pred_kin_state[:, 18:27]
    
    K = pred_impedance[:, :9]
    C = pred_impedance[:, 9:18]
    M = pred_impedance[:, 18:27]
    
    pred_torque = (K * (theta_des - theta_curr) + 
                   C * (omega_des - omega_curr) + 
                   M * (alpha_des - alpha_curr))
    
    return pred_torque

def validate_batch(batch, batch_idx):
    """Validate and clean a batch of data."""
    has_issues = False
    cleaned_batch = {}
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            # Check for NaN
            if torch.isnan(value).any():
                print(f"  WARNING: NaN in batch {batch_idx}, field '{key}'")  # ← This should show the field!
                print(f"    NaN count: {torch.isnan(value).sum().item()}")
                print(f"    Shape: {value.shape}")
                has_issues = True
                value = torch.nan_to_num(value, nan=0.0)
            
            # Check for Inf
            if torch.isinf(value).any():
                print(f"  WARNING: Inf in batch {batch_idx}, field '{key}'")
                has_issues = True
                value = torch.nan_to_num(value, posinf=1e6, neginf=-1e6)
            
            cleaned_batch[key] = value
        else:
            cleaned_batch[key] = value
    
    return not has_issues, cleaned_batch
    
def train_val_test_transformer(model, split_loader, optimizer, scheduler, args,
                               n_epochs=50, device='cuda', lr=1e-4, split_type='train',
                               use_impedance=False, lambda_kin=1, lambda_gait=1,
                               lambda_torque=1.0, lambda_jerk=1, val_dict={},
                               logger=None, global_step=None,max_steps=None):
    
    prev_impedances = [None,None]
    
    if logger is None:
        logger,log_file = setup_logger(args.log_dir)

    # Determine which noise flag applies to this split
    split_noise_flag = {
        'train': getattr(args, 'train_noise', False),
        'val':   getattr(args, 'val_noise',   False),
        'test':  getattr(args, 'test_noise',  False),
    }.get(split_type, False)

    for epoch in range(n_epochs):
        
        if split_type == 'train':
            pbar_desc = f"Training"
            pbar_total = max_steps
        else:
            # For val/test, we just call it an evaluation pass
            pbar_desc = f"{split_type.capitalize()} Eval"
            pbar_total = max_steps
                    
        split_pbar = tqdm(
            split_loader, 
            desc=pbar_desc, 
            total=pbar_total, 
        )

        # Validation
        if split_type == 'val' or split_type == 'test':
            model.eval()
        elif split_type=='train':
            model.train()

        with torch.no_grad() if split_type != 'train' else contextlib.nullcontext():

            #split_jerk_loss = 0
            n_split_batches = 0

            total_active_eval_loss = 0
            kinematic_active_eval_loss = 0
            torque_active_eval_loss = 0
            gait_active_eval_loss = 0

            n_active_terms = 0
            kinematic_active_terms = 0
            torque_active_terms = 0
            gait_active_terms = 0
            #jerk_active_terms = 0

            pred_kinematic_range = [float('inf'), float('-inf')]
            gt_kinematic_range = [float('inf'), float('-inf')]
            pred_torque_range = [float('inf'), float('-inf')]
            gt_torque_range = [float('inf'), float('-inf')]
            pred_impedance_range = [float('inf'), float('-inf')]
            
            for batch in split_pbar:

                if max_steps is not None and n_split_batches >= max_steps:
                    break

                emg             = batch['emg'].to(device)
                input_kin_state = batch['input_kin_state'].to(device)
                input_gait_pct  = batch['input_gait_pct'].to(device)
                target_kin_state = batch['target_kin_state'].to(device)
                target_gait_pct  = batch['target_gait_pct'].to(device)
                target_torque    = batch['target_torque'].to(device)
                has_torque       = batch['has_torque']
                
                emg_mask       = batch['emg_mask'].to(device)

                forward_emg_mask = batch['forward_emg_mask'].to(device)
                forward_kinematic_mask = batch['forward_kin_mask'].to(device)
                
                # 2. Pull True Masks (for the loss functions)

                try:
                    true_kinematic_mask = batch['kinematic_mask'].to(device)
                    true_kinetic_mask   = batch['kinetic_mask'].to(device)

                except Exception as e:
                    print('for real this time',e)

                # ── Gaussian signal noise ─────────────────────────────────────
                # Applied post-collation on device; targets are never noised.
                # Each batch draws fresh per-channel/per-dim distributions so
                # the model sees a different noise realization every forward pass.
                if split_noise_flag:
                    B = emg.shape[0]
 
                    # EMG: [B, C, T] — each sample gets its own per-channel mean and std
                    C = emg.shape[1]
                    emg_std  = torch.rand(B, C, 1, device=device) * args.emg_noise_std_max
                    emg_mean = (torch.rand(B, C, 1, device=device) * 2.0 - 1.0) * args.emg_noise_mean_max
                    emg = emg + torch.randn_like(emg) * emg_std + emg_mean
 
                    # Kin state: [B, D] — each sample gets its own per-dim mean and std
                    D = input_kin_state.shape[1]
                    kin_std  = torch.rand(B, D, device=device) * args.kin_noise_std_max
                    kin_mean = (torch.rand(B, D, device=device) * 2.0 - 1.0) * args.kin_noise_mean_max
                    input_kin_state = input_kin_state + torch.randn_like(input_kin_state) * kin_std + kin_mean
 
                    # Gait pct: [B, 1] — each sample gets its own scalar mean and std
                    gait_std  = torch.rand(B, 1, device=device) * args.gait_noise_std_max
                    gait_mean = (torch.rand(B, 1, device=device) * 2.0 - 1.0) * args.gait_noise_mean_max
                    input_gait_pct = input_gait_pct + torch.randn_like(input_gait_pct) * gait_std + gait_mean

                outputs = model(emg, input_kin_state, input_gait_pct, 
                                emg_mask=forward_emg_mask, kinematic_mask=forward_kinematic_mask, sample=False)

                pred_kin_state = outputs['pred_kin_state']
                pred_gait_pct  = outputs['pred_gait_pct']
                
                loss_kin  = model.masked_mse_loss(pred_kin_state, target_kin_state, true_kinematic_mask)

                loss_gait = nn.functional.mse_loss(pred_gait_pct, target_gait_pct.unsqueeze(dim=-1))
                loss = lambda_kin * loss_kin + lambda_gait * loss_gait

                total_active_eval_loss      += loss.item()
                kinematic_active_eval_loss  += loss.item()
                gait_active_eval_loss       += (lambda_gait * loss_gait).item()
                kinematic_active_terms      += 2
                gait_active_terms           += 1
                n_active_terms              += 2

                pred_kinematic_range[0] = min(pred_kin_state.min().item(), pred_kinematic_range[0])
                pred_kinematic_range[1] = max(pred_kin_state.max().item(), pred_kinematic_range[1])

                gt_kinematic_range[0] = min(gt_kinematic_range[0], target_kin_state.min().item())
                gt_kinematic_range[1] = max(gt_kinematic_range[1], target_kin_state.max().item())

                
                if use_impedance and 'pred_impedance_nm' in outputs:
                    pred_impedance = outputs['pred_impedance_nm']
                    pred_torque = compute_impedance_torque(
                        input_kin_state, pred_kin_state, pred_impedance
                    )
                    
                    if has_torque.any():
                        # FIXED: Use masked loss for torque as well if needed
                        if model.kinetic_mask.sum() > 0:
                            loss_torque = model.masked_mse_loss(
                                pred_torque, 
                                target_torque, 
                                true_kinetic_mask  # Only first 9 dimensions for torque
                            )
                        #NOTE biometric 2nd order temporal loss, penalize great changes

                        pred_impedance_range[0] = min(pred_impedance.min().item(), pred_impedance_range[0])
                        pred_impedance_range[1] = max(pred_impedance.max().item(), pred_impedance_range[1])
                        pred_torque_range[0] = min(pred_torque.min().item(), pred_torque_range[0])
                        pred_torque_range[1] = max(pred_torque.max().item(), pred_torque_range[1])

                        gt_torque_range[0] = min(gt_torque_range[0], target_torque.min().item())
                        gt_torque_range[1] = max(gt_torque_range[1], target_torque.max().item())

                        loss = loss + lambda_torque * loss_torque

                        total_active_eval_loss  += lambda_torque * loss_torque.item()
                        torque_active_eval_loss += lambda_torque * loss_torque.item()
                        torque_active_terms     += 1
                        n_active_terms          += 1
                    
                        #NOTE due to shuffling, jerk loss is not possible in these batch inferences
                        # if prev_impedances[0] is not None and prev_impedances[1] is not None:

                        #     loss_temporal_impedance_jerk = ((
                        #         pred_impedance - 2 * prev_impedances[0] + prev_impedances[1]
                        #     ) ** 2).mean()

                        #     loss = loss + loss_temporal_impedance_jerk
                            
                        #     total_active_eval_loss += loss_temporal_impedance_jerk.item()
                        #     n_active_terms         += 1
                        #     torque_active_terms    += 1
                        #     jerk_active_terms      += 1
                            
                        #     split_jerk_loss += loss_temporal_impedance_jerk.item()
                
                        # prev_impedances[1] = prev_impedances[0]
                        # prev_impedances[0] = pred_impedance.detach()

                n_split_batches += 1

                if split_type == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # ── Global step counter (drives jitter warmup ramp) ───────
                    if global_step is not None:
                        global_step[0] += 1

            if split_type == 'train':
                scheduler.step()
        
            # print('angle prediction ranges',pred_kinematic_range)
            # print('angle gt ranges',gt_kinematic_range)
            # print('impedance param prediction ranges',pred_impedance_range)
            # print('torque prediction ranges',pred_torque_range)
            # print('torque gt ranges',gt_torque_range)


        # Print statistics
        #NOTE losses are calculated about the avg of the total loss, normalized by the amount of individual active terms and batch
        #NOTE those that are backpropped are batch normalized
        avg_dataset_loss = total_active_eval_loss / (n_active_terms * max(n_split_batches, 1))

        if torque_active_terms != 0:
            #avg_dataset_jerk_loss  = split_jerk_loss / (jerk_active_terms * max(n_split_batches, 1))
            avg_dataset_torque_loss = torque_active_eval_loss / (torque_active_terms * max(n_split_batches, 1))
        avg_dataset_kinematic_loss = kinematic_active_eval_loss / (kinematic_active_terms * max(n_split_batches, 1))
        avg_dataset_gait_loss      = gait_active_eval_loss / (gait_active_terms * max(n_split_batches, 1))

        split_log = (f'{split_type} Loss: {avg_dataset_loss:.4f} | '
                   f'Avg Kin: {avg_dataset_kinematic_loss:.4f} | '
                   f'Avg Gait: {avg_dataset_gait_loss:.4f} | '
                   )
                    
        if torque_active_terms != 0:
            split_log += f' | Avg Torque: {avg_dataset_torque_loss:.4f}'
            #split_log += f' | Avg Jerk: {avg_dataset_jerk_loss:.4f}'
        logger.info(split_log)
            

    loss_dict = {'avg_total_loss': avg_dataset_loss,
            'avg_torque_loss' : None,
            'avg_kinematic_loss': avg_dataset_kinematic_loss
    }
    if torque_active_terms != 0: 
        loss_dict['avg_torque_loss'] = avg_dataset_torque_loss

    return loss_dict

def check_load_time(args, dataset_path='D:/EMG/ML_datasets/run1'):
    
    for i, curr_dataset in enumerate(os.listdir(dataset_path)):
        print('loading ', curr_dataset)
        
        for j, activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
            
            trainData = []
            valData = []
            testData = []
            
            total_load_time = 0
            total_dataloader_time = 0
            
            for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                # Time the torch.load operation
                load_start = time.time()
                train_path = dataset_path + '/' + curr_dataset + '/' + activity + '/' + chunk + '/' + 'train.pt'
                train_data = torch.load(train_path)
                load_end = time.time()
                load_time = load_end - load_start
                total_load_time += load_time
                
                # Time the DataLoader creation
                dataloader_start = time.time()
                train_obj = SplitDataset(split='train',use_noise=args.train_noise, args=args)
                train_obj.data = {'train': train_data} 
                
                train_loader = DataLoader(
                    train_obj, 
                    batch_size=args.batch_size,
                    shuffle=True, 
                    num_workers=args.num_workers,
                    pin_memory=True,
                    prefetch_factor=2,
                    drop_last=True
                )
                dataloader_end = time.time()
                dataloader_time = dataloader_end - dataloader_start
                total_dataloader_time += dataloader_time
                
                trainData.append(train_loader)
                
                print(f'  Chunk {k}: Load time = {load_time:.3f}s, DataLoader creation = {dataloader_time:.3f}s')
            
            print(f'\nActivity {activity} Summary:')
            print(f'  Total chunks: {k+1}')
            print(f'  Total load time: {total_load_time:.2f}s (avg {total_load_time/(k+1):.3f}s per chunk)')
            print(f'  Total DataLoader creation: {total_dataloader_time:.2f}s')
            print(f'  Combined overhead: {total_load_time + total_dataloader_time:.2f}s\n')

def meta_train_temperature_loop(args, dataset_path='D:/EMG/ML_datasets/debug', outer_epochs=3,
                                steps_per_epoch=512, checkpoint_path=None):

    load = False
    global_step = [0]

    overall_eval_dataset_losses = {
        'bacek':    {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'gait120':  {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'k2muse':   {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'lencioni': {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'moghadam': {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'moreira':  {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'siat':     {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        'hu':       {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
    }

    overall_best_ceiling_losses = {
        'kinematic': float('inf'),
        'torque':    float('inf'),
        'avg':       float('inf'),
    }

    datasets = {
        # 'bacek': 258418, 
        #'macaluso': 66035, 
        #'camargo': 53713, 
        # 'k2muse': 40612,
        #'angelidou': 40204, 
        #'embry': 26846, 
        #'grimmer': 10772, 
        # 'hu': 6365,
        # 'gait120': 6310, 
        # 'moreira': 2613, 
        #'criekinge': 2102, 
        'lencioni': 1159,
        #'siat': 441, 
        # 'moghadam': 290
    }

    logger, log_file = setup_logger(args.log_dir)


    logger.info("Pre-loading Validation DataLoaders into memory...")
    val_loaders = {}

    for curr_dataset in os.listdir(dataset_path):   
        for j, activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
            for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):

                val_path = f'{dataset_path}/{curr_dataset}/{activity}/{chunk}/val.pt'
                if not os.path.exists(val_path): continue
                
                # Load from disk ONCE
                val_data = torch.load(val_path, weights_only=False)
                val_obj = SplitDataset(split='val', use_noise=args.val_noise, args=args)
                val_obj.data = {'val': val_data}
                if len(val_obj) ==0:
                    continue
                
                # Create DataLoader ONCE
                loader = DataLoader(val_obj, batch_size=args.batch_size, shuffle=False, drop_last=True, collate_fn=safe_collate)
                
                # Store it
                key = f"{curr_dataset}_{activity}_{chunk}"

                val_loaders[key] = {
                    'loader': loader,
                    'dataset_name': curr_dataset,
                    'masks': val_data['masks']
                }

    logger.info("Validation caching complete.") 

    # 1. Init Streamer and Loader
    # Using alpha=0.3 as discussed for polynomial smoothing
    train_streamer = TemperatureScaledStreamer(
        dataset_path=dataset_path,
        dataset_sizes=datasets,
        args=args,
        split='train',
        alpha=0.3, 
        buffer_capacity=50000, 
        refill_threshold=10000,
        global_step=global_step
    )
    
    train_loader = DataLoader(
        train_streamer,
        batch_size=args.batch_size,
        num_workers=args.num_workers, 
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=safe_collate
    )

    # 2. Init Model
    model = EMGTransformer(
        emg_channels=13, emg_window_size=100, kin_state_dim=27,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
        predict_impedance=args.use_impedance,
        emg_mask=np.ones(13),             # Pass dummies here, actual masks loaded dynamically
        kinematic_mask=np.ones((3,3)), 
        kinetic_mask=np.ones(9),
        device=args.device,
        save_dir=args.save_model_path
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=outer_epochs, eta_min=args.lr/100)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'overall_best_ceiling_losses' in checkpoint.keys():
            overall_best_ceiling_losses = checkpoint['overall_best_ceiling_losses']

    # Because train_loader is now infinite, we turn it into an iterator
    train_iterator = iter(train_loader)

    # ────────────────────────────────────────────────────────────────────────
    # 3. THE TRAINING LOOP (Step-Based)
    # ────────────────────────────────────────────────────────────────────────
    for outer_epoch in range(outer_epochs):
        logger.info(f'================ OUTER EPOCH {outer_epoch+1}/{outer_epochs} ================')
        
        # -- TRAIN PHASE --
        logger.info(f'Training for {steps_per_epoch} mixed batches...')
        
        # We pass the infinite iterator to the trainer, and tell it to stop after `steps_per_epoch`
        loss_dict = train_val_test_transformer(
            model, 
            train_iterator, # Note: Pass the iterator, not the loader
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            split_type='train',
            n_epochs=1, 
            device=args.device,
            lr=args.lr,
            use_impedance=args.use_impedance,
            logger=logger,
            global_step=global_step,
            max_steps=steps_per_epoch
        )

        # -- VALIDATION PHASE --
        curr_eval_dataset_losses = {k: {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': None, 'dataset_total_kinematic_loss': 0.0} for k in datasets.keys()}

        # Just ONE loop over the flat dictionary
        for key, chunk_data in val_loaders.items():
            
            curr_dataset = chunk_data['dataset_name'].lower()
            if curr_dataset not in curr_eval_dataset_losses: 
                continue
                
            logger.info(f"Validating {key}")
            
            val_loader = chunk_data['loader']
            masks = chunk_data['masks']

            # Dynamically update the model masks for validation
            model.emg_mask = torch.Tensor(masks['emg']).float().to(model.device)
            model.kinematic_mask = torch.Tensor(np.tile(masks['kinematic'].flatten(), 3)).float().to(model.device)
            if masks['kinetic'] is not None:
                model.kinetic_mask = torch.Tensor(masks['kinetic'].flatten()).float().to(model.device)
            else:
                model.kinetic_mask = torch.zeros(9).float().to(model.device)
                                                
            loss_dict = train_val_test_transformer(
                model, val_loader, optimizer=optimizer, scheduler=scheduler,
                split_type='val', args=args, n_epochs=1, device=args.device,
                lr=args.lr, use_impedance=args.use_impedance, logger=logger,
                max_steps=100 
            )

            # Accumulate losses
            curr_eval_dataset_losses[curr_dataset]['dataset_total_avg_loss'] += loss_dict['avg_total_loss']
            curr_eval_dataset_losses[curr_dataset]['dataset_total_kinematic_loss'] += loss_dict['avg_kinematic_loss']
            
            # Safely accumulate torque loss
            if loss_dict['avg_torque_loss'] is not None: 
                if curr_eval_dataset_losses[curr_dataset]['dataset_total_torque_loss'] is None:
                    curr_eval_dataset_losses[curr_dataset]['dataset_total_torque_loss'] = 0.0
                curr_eval_dataset_losses[curr_dataset]['dataset_total_torque_loss'] += loss_dict['avg_torque_loss']

        # Check and save checkpoints
        overall_eval_dataset_losses, overall_best_ceiling_losses = model.check_and_save_checkpoints(
            model, optimizer, scheduler, args, curr_eval_dataset_losses,
            overall_eval_dataset_losses, overall_best_ceiling_losses, outer_epoch, logger
        )
    
    create_plots(log_file, args.save_plot_dir)
    print("\nDone!")
           
def setup_logger(log_dir='/gpfs/data/s001/vwulfek1/software/logs'):
    """
    Set up logging to both file and console.
    Creates a timestamped log file in the specified directory.
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Path(log_dir) / f'training_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file created: {log_file}")
    return logger, log_file

def main():

    parser = argparse.ArgumentParser()
    #TODO CLI loading + num_workers 

    # ── Data / training ───────────────────────────────────────────────────────
    parser.add_argument('--dataset_path', type=str, default='D:/EMG/ML_datasets',
                       help='Directory containing pickle files')
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Directory containing pickle files')         
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--save_plot_dir', type=str, default='/gpfs/data/s001/vwulfek1/software/plots/', 
                       help='Save path of model checkpoints')
    parser.add_argument('--save_model_path', type=str, default='C:/EMG/software/models/IM', 
                       help='Save path of model checkpoints')
    parser.add_argument('--load_path', type=str, default=None, 
                       help='load path of model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_impedance', action='store_true',
                       help='Use impedance control with torque prediction', default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)

    parser.add_argument('--artificial_emg_mask_prob', type=float, default=0.15,
                        help='Probability of completely zeroing out an available EMG channel during training')
    parser.add_argument('--artificial_kin_mask_prob', type=float, default=0.15,
                        help='Probability of completely zeroing out an available kinematic dimension during training')

    # ── Noise augmentation — split flags ─────────────────────────────────────
    parser.add_argument('--train_noise', action='store_true', default=False,
                        help='Enable signal + temporal jitter noise on train split')
    parser.add_argument('--val_noise', action='store_true', default=False,
                        help='Enable signal + temporal jitter noise on val split')
    parser.add_argument('--test_noise', action='store_true', default=False,
                        help='Enable signal + temporal jitter noise on test split')

    # ── Gaussian signal noise ─────────────────────────────────────────────────
    parser.add_argument('--emg_noise_std_max', type=float, default=1.0,
                        help='Per-channel EMG std upper bound: std_c ~ U[0, std_max]')
    parser.add_argument('--emg_noise_mean_max', type=float, default=0.0,
                        help='Per-channel EMG mean upper bound: mean_c ~ U[-mean_max, mean_max]')
    parser.add_argument('--kin_noise_std_max', type=float, default=1.0,
                        help='Per-dim kin std upper bound: std_d ~ U[0, std_max]')
    parser.add_argument('--kin_noise_mean_max', type=float, default=0.0,
                        help='Per-dim kin mean upper bound: mean_d ~ U[-mean_max, mean_max]')
    parser.add_argument('--gait_noise_std_max', type=float, default=0.05,
                        help='Gait pct std upper bound: std ~ U[0, std_max]')
    parser.add_argument('--gait_noise_mean_max', type=float, default=0.0,
                        help='Gait pct mean upper bound: mean ~ U[-mean_max, mean_max]')

    # ── Temporal jitter ───────────────────────────────────────────────────────
    parser.add_argument('--emg_jitter_max', type=int, default=5,
                        help='EMG temporal jitter upper bound in window steps '
                             '(delta ~ U[0, max]; window slid back within same stride)')
    parser.add_argument('--kin_jitter_max', type=int, default=5,
                        help='Kin temporal jitter upper bound in window steps '
                             '(sampled independently from EMG jitter)')
    parser.add_argument('--jitter_warmup_steps', type=int, default=0,
                        help='Linearly ramp both jitter_max values 0 → max over N global '
                             'train steps (0 = off; only applied to train split)')
    parser.add_argument('--jitter_retries', type=int, default=5,
                        help='Max attempts to find a valid jittered index before '
                             'falling back to the original index (no jitter)')

    args = parser.parse_args()
    
    print("Loading and parsing datasets...")

    #STEPS PER EPOCH IS HARD CODED HERE ALONG WITH THE EVALUATION STEPS PER EPOCH
    meta_train_temperature_loop(args=args, dataset_path=args.dataset_path, steps_per_epoch =100,checkpoint_path=args.load_path,outer_epochs=args.epochs)
    
    print("\nTraining complete!")


if __name__ == '__main__':

    main()