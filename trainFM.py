import numpy as np
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
from convert2DL import WindowedGaitDataParser, SplitDataset
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
#TODO biometric loss

def soft_update(source, target, tau):
    """
    Soft update of the target network parameters.
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

class QNetwork(nn.Module):
    def __init__(self,input_size,h_dim,output_size,dropout=0.1,lr=1e-8,checkpoint_dir='C:/EMG/models/SAC'):
        super().__init__()
        self.device='cuda'
        self.input_size = input_size
        self.h_dim = h_dim
        self.output_size = output_size
        self.dropout = dropout
        self.q_network = nn.Sequential(nn.Linear(self.input_size,self.h_dim),
                                       nn.LayerNorm(self.h_dim),
                                       nn.Tanh(),
                                       nn.Linear(self.h_dim,self.h_dim*2),
                                       nn.LayerNorm(self.h_dim*2),
                                       nn.Tanh(),
                                       nn.Linear(self.h_dim*2,self.output_size),
                                       nn.Dropout(self.dropout)
                                       ).to(self.device)
        self._init_weights()
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
        self.checkpoint_dir = checkpoint_dir

    def _init_weights(self):
        """Initialize weights to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self,state,action):
        return self.q_network(torch.cat([state,action],dim=-1))
        
    def save_checkpoint(self,name,arg):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({'state_dict':self.state_dict(),
                    'config':arg}, f'{self.checkpoint_dir}/{name}')
        self.checkpoint_file=f'{self.checkpoint_dir}/{name}'

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.ptr = 0  # Current position to write
        self.size = 0  # Current buffer size

        # Pre-allocate memory with float32 for efficiency
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
    

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'state_memoryTryNow31.npy'), self.state_memory)
        np.save(os.path.join(save_dir, 'action_memoryTryNow31.npy'), self.action_memory)
        np.save(os.path.join(save_dir, 'reward_memoryTryNow31.npy'), self.reward_memory)
        np.save(os.path.join(save_dir, 'new_state_memoryTryNow31.npy'), self.new_state_memory)
        np.save(os.path.join(save_dir, 'terminal_memoryTryNow31.npy'), self.terminal_memory)
        np.save(os.path.join(save_dir, 'ptrTryNow31.npy'), self.ptr)
        np.save(os.path.join(save_dir, 'sizeTryNow31.npy'), self.size)

    def load(self, load_dir):
        self.state_memory = np.load(os.path.join(load_dir, 'state_memoryTryNow31.npy'))
        self.action_memory = np.load(os.path.join(load_dir, 'action_memoryTryNow31.npy'))
        self.reward_memory = np.load(os.path.join(load_dir, 'reward_memoryTryNow31.npy'))
        self.new_state_memory = np.load(os.path.join(load_dir, 'new_state_memoryTryNow31.npy'))
        self.terminal_memory = np.load(os.path.join(load_dir, 'terminal_memoryTryNow31.npy'))
        self.ptr = np.load(os.path.join(load_dir, 'ptrTryNow31.npy'))
        self.size = np.load(os.path.join(load_dir, 'sizeTryNow31.npy'))

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
                 device='cuda'):
        super().__init__()
        
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
        if kinetic_mask is not None and kinetic_mask.any():
            self.kinetic_mask = torch.Tensor(kinetic_mask.flatten()).float().to(device)
        else:
            self.predict_impedance = False
            self.kinetic_mask = torch.Tensor(np.zeros((9))).float().to(device)
        
        self.emg_conv = nn.Sequential(
            nn.Conv1d(self.emg_channels, self.emg_conv_ip_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_ip_channels, emg_window_size]),  # Add normalization
            nn.ReLU(),
            nn.Conv1d(self.emg_conv_ip_channels, self.emg_conv_hidden_channels, kernel_size=5, padding=2),
            nn.LayerNorm([self.emg_conv_hidden_channels, emg_window_size]),  # Add normalization
            nn.ReLU(),
            nn.Conv1d(self.emg_conv_hidden_channels, self.emg_channels, kernel_size=5, padding=2),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gait percentage embedding
        self.gait_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model),  # Add normalization
            nn.ReLU(),
        )
        self.replay_buffer = ReplayBuffer(max_size=int(1e6),input_shape=int(13*100+27),n_actions=27*2)
        
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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, kin_state_dim)
        )

        self.kin_output_log_std = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, kin_state_dim)
        )

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -27

        self.gait_output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            #nn.Sigmoid()  # Gait percentage should be 0-1
        )
        
        if predict_impedance:
            self.impedance_output = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 27),
                nn.Softplus()
            )

            self.impedance_output_log_std = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.LayerNorm(dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 27),
                nn.Softplus()
            )

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def save_checkpoint(self,optimizer,scheduler,args,best_val_loss,epoch):
        torch.save({
            'model_config': {'num_layers':args.num_layers,'d_model':args.d_model,'nhead':args.nhead},
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict':scheduler.state_dict(),
            'val_loss': best_val_loss,
        }, 'C:/EMG/models/SAC/best_RL_transformer_model.pth')

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
            torch.save(base_save, 'C:/EMG/models/best_kinematic.pth')
            logger.info(f'Saved best_kinematic.pth | ceiling: {curr_kin_ceiling:.4f}')

        if curr_torque_ceiling is not None and curr_torque_ceiling < overall_best_ceiling_losses['torque']:
            overall_best_ceiling_losses['torque'] = curr_torque_ceiling
            torch.save(base_save, 'C:/EMG/models/best_torque.pth')
            logger.info(f'Saved best_torque.pth | ceiling: {curr_torque_ceiling:.4f}')

        if curr_avg_ceiling < overall_best_ceiling_losses['avg']:
            overall_best_ceiling_losses['avg'] = curr_avg_ceiling
            torch.save(base_save, 'C:/EMG/models/best_avg.pth')
            logger.info(f'Saved best_avg.pth | ceiling: {curr_avg_ceiling:.4f}')

        # update overall history with current epoch results
        for d in curr_eval_dataset_losses:
            for metric in curr_eval_dataset_losses[d]:
                overall_eval_dataset_losses[d][metric] = curr_eval_dataset_losses[d][metric]

        return overall_eval_dataset_losses, overall_best_ceiling_losses
        
    def forward(self, emg, input_kin_state, input_gait_pct=None,sample=False):
        outputs = {}
        
        """
        Args:
            emg: (batch, emg_channels, emg_window_size)
            input_kin_state: (batch, 27) - current angles, omega, alpha
            input_gait_pct: (batch, 1) - current gait percentage
        
        Returns:
            Dictionary with predictions
        """
        self.emg_mask = self.emg_mask.to(self.device)
        # Apply masks properly (element-wise multiplication)
        emg_masked = emg * self.emg_mask.view(1, -1, 1)
        
        # Process EMG
        emg_features = self.emg_conv(emg_masked)  # (batch, d_model, emg_seq_len)
        
        # Process kinematic state and gait
        if self.kinematic_mask.shape[-1]==3: self.kinematic_mask = torch.Tensor(np.tile(self.kinematic_mask.flatten(), 3)).float().to(self.device)

        kin_masked = input_kin_state * self.kinematic_mask.view(1, -1)
        kin_features = self.kin_embedding(kin_masked.unsqueeze(1))  # (batch, 1, d_model)
        if input_gait_pct.any():
            gait_features = self.gait_embedding(input_gait_pct.unsqueeze(1))  # (batch, 1, d_model)

        # Combine into encoder input sequence
        encoder_input = emg_features
        encoder_input = self.pos_encoder(encoder_input)
        
        # Create decoder input
        if input_gait_pct.any():
            decoder_input = torch.cat([kin_features, gait_features], dim=1)
        else: 
            decoder_input = kin_features
        
        # Transformer
        transformer_output = self.transformer(encoder_input, decoder_input)
        
        # Predictions
        pred_kin_state = self.kin_output(transformer_output[:, 0, :])
        if sample:
            pred_kin_state_log_std = self.kin_output_log_std(transformer_output[:, 0, :])
            clamped_log_std = torch.clamp(pred_kin_state_log_std, LOG_STD_MIN, LOG_STD_MAX)            #exponentiate it
            std = torch.exp(clamped_log_std)
            kin_dist = torch.distributions.Normal(pred_kin_state, std)
            kin_sample=kin_dist.rsample()
            kin_log_prob = kin_dist.log_prob(kin_sample).sum(dim=-1, keepdim=True)

            outputs['pred_kin_state'] = kin_sample
            outputs['pred_kin_log_pdf'] = kin_log_prob
        else:
            outputs[ 'pred_kin_state'] = pred_kin_state

            #rparam sample
            #log_pdf of sample and distribution
            
        if input_gait_pct.any():
            pred_gait_pct = self.gait_output(transformer_output[:, 1, :])


        if input_gait_pct.any():
            outputs['pred_gait_pct'] = pred_gait_pct
        
        
        if self.predict_impedance:
            pred_impedance = self.impedance_output(transformer_output[:, 0, :])

            if sample: 
                pred_impedance_log_std=self.impedance_output_log_std(transformer_output[:, 0, :])
                clamped_log_std = torch.clamp(pred_impedance_log_std, LOG_STD_MIN, LOG_STD_MAX)            #exponentiate it
                std = torch.exp(clamped_log_std)
                pred_impedance_dist = torch.distributions.Normal(pred_impedance, std)
                pred_impedance_sample=pred_impedance_dist.rsample()
                pred_imp_log_pdf = pred_impedance_dist.log_prob(pred_impedance_sample).sum(dim=-1, keepdim=True)
                outputs['pred_impedance'] = pred_impedance_sample
                outputs['pred_impedance_log_pdf'] = pred_imp_log_pdf
            else: 
                outputs['pred_impedance'] = pred_impedance


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
        pred_masked = pred * mask.unsqueeze(0)
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

def train_sac(policy_args,critic_args,Policy,QNetwork_base1,QNetwork_base2,QNetwork_target1,QNetwork_target2,
              replay_buff,training_epochs,training_losses,sample_batch_size=256):
    
    gamma = 0.99
    tau = 0.05  # Soft update coefficient
    training_iterations = 0
    q1_loss = 0
    q2_loss = 0
    actor_loss=0

    if replay_buff.size < sample_batch_size: 
        return 
    
    while training_iterations < training_epochs:
        # Ensure enough samples in replay buffer


    # Sample from replay buffer
        states, states_, actions, rewards, dones = replay_buff.sample_buffer(sample_batch_size)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to('cuda')
        states_ = torch.tensor(states_, dtype=torch.float32).to('cuda')
        actions = torch.tensor(actions, dtype=torch.float32).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float32).to('cuda').unsqueeze(dim=-1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(dim=-1).to('cuda')
    
        emg_state = states[:,:13*100].reshape(states.shape[0],13, 100)
        kinematic_state = states[:,13*100:].reshape(states.shape[0],27)

        emg_next_state = states_[:,:13*100].reshape(states_.shape[0],13, 100)
        kinematic_next_state = states_[:,13*100:].reshape(states_.shape[0],27)

        # Critic (Q-network) update
        #the torch.no_grads dont contribute to the gradient computation
        with torch.no_grad():
            # Sample next actions
            outputs_= Policy(emg_next_state.to(Policy.device),kinematic_next_state.to(Policy.device),sample=True)
            next_actions = torch.cat([outputs_['pred_kin_state'], outputs_['pred_impedance']],dim=-1)
            # Compute target Q-values
            target1_q = QNetwork_target1(states_.to(QNetwork_target1.device), next_actions.to(QNetwork_target1.device))
            target2_q = QNetwork_target2(states_.to(QNetwork_target2.device), next_actions.to(QNetwork_target2.device))
            target_q = torch.min(target1_q, target2_q)
            # next_log_probs = next_log_probs.unsqueeze(-1)
            # rewards = rewards.unsqueeze(-1)
            # dones = dones.unsqueeze(-1)        
    
            # Compute target values

            y = rewards + gamma * (1 - dones) * (target_q - Policy.log_alpha.exp().detach() * (outputs_['pred_kin_log_pdf'].detach()+outputs_['pred_kin_log_pdf'].detach()))
            #detaching irrelevant calcs in the backprop update!
        # Current Q-values

        current_q1 = QNetwork_base1.forward(states, actions)
        current_q2 = QNetwork_base2.forward(states, actions)

        training_losses['q1_mean'].append(current_q1.mean().item())
        training_losses['q2_mean'].append(current_q2.mean().item())
        # print(current_q1,current_q2,target_q)
        # Q-network losses
        q1_loss = F.mse_loss(current_q1, y)
        q2_loss = F.mse_loss(current_q2, y)

        QNetwork_base1.optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(QNetwork_base1.parameters(), max_norm=0.5)
        QNetwork_base1.optimizer.step()

        QNetwork_base2.optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(QNetwork_base2.parameters(), max_norm=0.5)
        QNetwork_base2.optimizer.step()

        # Actor loss
        outputs = Policy(emg_state,kinematic_state,sample=True)
        # print("action dim, log prob dim:", sampled_actions.shape, log_probs.shape)

        #with torch.no_grad():
        q1_vals = QNetwork_base1(states.to(QNetwork_base1.device), torch.cat([outputs['pred_kin_state'], outputs['pred_impedance']],dim=-1).to(QNetwork_base1.device))
        q2_vals = QNetwork_base2(states.to(QNetwork_base2.device), torch.cat([outputs['pred_kin_state'], outputs['pred_impedance']],dim=-1).to(QNetwork_base2.device))
        q_vals = torch.min(q1_vals, q2_vals)
            # print("q_vals shape", q_vals.shape)
        
            #detaching irrelevant calcs in the backprop update!
        for p in QNetwork_base1.parameters():
            p.requires_grad = False
        for p in QNetwork_base2.parameters():
            p.requires_grad = False
        log_pdfs=(outputs_['pred_kin_log_pdf']+outputs_['pred_impedance_log_pdf'])

        actor_loss = (Policy.log_alpha * log_pdfs - q_vals).mean()
        Policy.optimizer.zero_grad()
        actor_loss.backward()
        Policy.optimizer.step()

        # Unfreeze
        for p in QNetwork_base1.parameters():
            p.requires_grad = True
        for p in QNetwork_base2.parameters():
            p.requires_grad = True

        alpha_loss = -(Policy.log_alpha * (log_pdfs.detach() + Policy.target_entropy)).mean()

        Policy.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        Policy.alpha_optimizer.step()

        #loss_alpha = -a(log_pi(a|s)+ H)

            #detaching irrelevant calcs in the backprop update!

        #alpha_loss/=sample_batch_size

        # print("q1 and q2 loss", q1_loss.shape, q2_loss.shape)


        training_losses['actor_loss'].append(actor_loss.item())
        training_losses['q1_loss'].append(q1_loss.item())
        training_losses['q2_loss'].append(q2_loss.item())
        training_losses['alpha_loss'].append(alpha_loss.item())

        soft_update(QNetwork_base1, QNetwork_target1, tau)
    
        soft_update(QNetwork_base2, QNetwork_target2, tau)
        
        actor_loss=0
        q1_loss=0
        q2_loss=0
        alpha_loss=0
        training_iterations += 1

    # Logging

    # Print progress

    # Final training summary
    print("\n--- Training Phase Complete ---")
    print("Average Losses:")
    print(f"  Actor Loss: {np.mean(training_losses['actor_loss']):.4f}")
    print(f"  Q1 Network Loss: {np.mean(training_losses['q1_loss']):.4f}")
    print(f"  Q2 Network Loss: {np.mean(training_losses['q2_loss']):.4f}")

    #visualizer.plot()

    # Save networks
    Policy.save_checkpoint(policy_args,np.mean(training_losses['actor_loss']),training_iterations)
    QNetwork_base1.save_checkpoint('Q1B',critic_args)
    QNetwork_base2.save_checkpoint('Q2B',critic_args)
    QNetwork_target1.save_checkpoint('Q1T',critic_args)
    QNetwork_target2.save_checkpoint('Q2T',critic_args)
    replay_buff.save('/tmp1')
    print("Checkpoints saved.")

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
    
def train_val_test_transformer(model, split_loader,optimizer,scheduler,args,n_epochs=50, 
                      device='cuda', lr=1e-4, split_type='train',use_impedance=False,
                      lambda_kin=1, lambda_gait=1, lambda_torque=1.0,lambda_jerk = 1,val_dict={},logger=None):
    
    prev_impedances = [None,None]
    
    if logger is None:
        logger,log_file = setup_logger()

    if len(split_loader)==0: 
        logger.info(f'length of split data is 0, skipping..')
        return {'avg_total_loss': None,
            'avg_torque_loss' : None,
            'avg_kinematic_loss': None
            }
    
    for epoch in range(n_epochs):
        
        # Validation
        if split_type == 'val' or split_type == 'test':
            model.eval()
        elif split_type=='train':
            model.train()

        with torch.no_grad() if split_type != 'train' else contextlib.nullcontext():

            split_jerk_loss = 0
            n_split_batches = 0

            total_active_eval_loss = 0
            kinematic_active_eval_loss = 0
            torque_active_eval_loss = 0
            gait_active_eval_loss = 0

            n_active_terms = 0
            kinematic_active_terms = 0
            torque_active_terms = 0
            gait_active_terms = 0
            jerk_active_terms = 0

            pred_kinematic_range = [float('inf'), float('-inf')]
            gt_kinematic_range = [float('inf'), float('-inf')]
            pred_torque_range = [float('inf'), float('-inf')]
            gt_torque_range = [float('inf'), float('-inf')]
            pred_impedance_range = [float('inf'), float('-inf')]

            
            split_pbar = tqdm(split_loader, desc=f'Epoch {epoch+1}/{n_epochs} [split]')

            for batch in split_pbar:
                emg = batch['emg'].to(device)
                input_kin_state = batch['input_kin_state'].to(device)
                input_gait_pct = batch['input_gait_pct'].to(device)
                target_kin_state = batch['target_kin_state'].to(device)
                target_gait_pct = batch['target_gait_pct'].to(device)
                target_torque = batch['target_torque'].to(device)
                has_torque = batch['has_torque']
                
                outputs = model(emg, input_kin_state, input_gait_pct,sample=False)
                pred_kin_state = outputs['pred_kin_state']
                pred_gait_pct = outputs['pred_gait_pct']
                
                loss_kin = model.masked_mse_loss(pred_kin_state, target_kin_state, model.kinematic_mask)
                loss_gait = nn.functional.mse_loss(pred_gait_pct, target_gait_pct)
                loss = lambda_kin * loss_kin + lambda_gait * loss_gait

                total_active_eval_loss+=loss.item()
                kinematic_active_eval_loss+=loss.item()
                gait_active_eval_loss+=(lambda_gait * loss_gait).item()
                kinematic_active_terms+=2
                gait_active_terms+=1
                n_active_terms +=2

                pred_kinematic_range[0] = min(pred_kin_state.min().item(),pred_kinematic_range[0])
                pred_kinematic_range[1] = max(pred_kin_state.max().item(),pred_kinematic_range[1])

                gt_kinematic_range[0] = min(gt_kinematic_range[0],target_kin_state.min().item())
                gt_kinematic_range[1] = max(gt_kinematic_range[1],target_kin_state.max().item())

                
                if use_impedance and 'pred_impedance' in outputs:
                    pred_impedance = outputs['pred_impedance']
                    pred_torque = compute_impedance_torque(
                        input_kin_state, pred_kin_state, pred_impedance
                    )
                    
                    if has_torque.any():
                        # FIXED: Use masked loss for torque as well if needed
                        if model.kinetic_mask.sum() > 0:
                            loss_torque = model.masked_mse_loss(
                                pred_torque, 
                                target_torque, 
                                model.kinetic_mask  # Only first 9 dimensions for torque
                            )
                        #NOTE biometric 2nd order temporal loss, penalize great changes

                        pred_impedance_range[0] = min(pred_impedance.min().item(),pred_impedance_range[0])
                        pred_impedance_range[1] = max(pred_impedance.max().item(),pred_impedance_range[1])
                        pred_torque_range[0]=min(pred_torque.min().item(),pred_torque_range[0])
                        pred_torque_range[1]=max(pred_torque.max().item(),pred_torque_range[1])

                        gt_torque_range[0] = min(gt_torque_range[0],target_torque.min().item())
                        gt_torque_range[1] = max(gt_torque_range[1],target_torque.max().item())

                        loss = loss + lambda_torque * loss_torque

                        total_active_eval_loss +=lambda_torque * loss_torque.item()
                        torque_active_eval_loss+=lambda_torque * loss_torque.item()
                        torque_active_terms+=1
                        n_active_terms+=1
                    
                        if prev_impedances[0] is not None and prev_impedances[1] is not None:

                            loss_temporal_impedance_jerk = ((
                                pred_impedance - 2 * prev_impedances[0] + prev_impedances[1]
                            ) ** 2).mean()

                            loss = loss + loss_temporal_impedance_jerk
                            
                            total_active_eval_loss += loss_temporal_impedance_jerk.item()
                            n_active_terms+=1
                            torque_active_terms+=1
                            jerk_active_terms+=1
                            
                            split_jerk_loss+=loss_temporal_impedance_jerk.item()
                
                        prev_impedances[1] = prev_impedances[0]
                        prev_impedances[0] = pred_impedance.detach()

                n_split_batches += 1

                if split_type=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if split_type=='train':
                scheduler.step()
        
            # print('angle prediction ranges',pred_kinematic_range)
            # print('angle gt ranges',gt_kinematic_range)
            # print('impedance param prediction ranges',pred_impedance_range)
            # print('torque prediction ranges',pred_torque_range)
            # print('torque gt ranges',gt_torque_range)

            # input('waiting for input')

        # Print statistics
        #NOTE losses are calculated about the avg of the total loss, normalized by the amount of individual active terms and batch
        #NOTE those that are backpropped are batch normalized
        avg_dataset_loss = total_active_eval_loss / (n_active_terms * max(n_split_batches, 1))
        if torque_active_terms!=0:
            avg_dataset_jerk_loss = split_jerk_loss / (jerk_active_terms * max(n_split_batches, 1))
            avg_dataset_torque_loss = torque_active_eval_loss / (torque_active_terms * max(n_split_batches, 1))
        avg_dataset_kinematic_loss = kinematic_active_eval_loss / (kinematic_active_terms * max(n_split_batches, 1))
        avg_dataset_gait_loss = gait_active_eval_loss / (gait_active_terms * max(n_split_batches, 1))

        
        #logger.info(f'\nEpoch {epoch+1}/{n_epochs}')

        split_log = (f'{split_type} Loss: {avg_dataset_loss:.4f} | '
                   f'Avg Kin: {avg_dataset_kinematic_loss:.4f} | '
                   f'Avg Gait: {avg_dataset_gait_loss:.4f} | '

                   )
                    
        if use_impedance:
            split_log += f' | Avg Torque: {avg_dataset_torque_loss:.4f}'
            split_log += f' | Avg Jerk: {avg_dataset_jerk_loss:.4f}'
        logger.info(split_log)
            

    loss_dict = {'avg_total_loss': avg_dataset_loss,
            'avg_torque_loss' : None,
            'avg_kinematic_loss': avg_dataset_kinematic_loss
    }
    if torque_active_terms!=0: 
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
                train_obj = SplitDataset(split='train')
                train_obj.data = {'train': train_data} 
                
                train_loader = DataLoader(
                    train_obj, 
                    batch_size=args.batch_size,
                    shuffle=True, 
                    num_workers=2,
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

def meta_train_transformer_loop(args,dataset_path = 'D:/EMG/ML_datasets/debug',outer_epochs=2,checkpoint_path='C:/EMG/models/best_transformer_model100m.pth'):
    load = False

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
        'bacek': 258418,
        'macaluso': 66035,
        'camargo': 53713,
        'k2muse': 40612,
        'angelidou': 40204,
        'embry': 26846,
        'grimmer': 10772,
        'hu': 6365,
        'gait120': 6310,
        'moreira': 2613,
        'criekinge': 2102,
        'lencioni': 1159,
        'siat': 441,
        'moghadam': 290
    }


    inverse_values = {k: 1/v for k, v in datasets.items()}
    total_inverse = sum(inverse_values.values())

    logger,log_file = setup_logger()


    # Normalize to percentages and scales to number of epochs
    #sum of data will get args.epochs with each dataset getting their inverse normalized proportion
    inverse_proportions = {k: math.ceil((v/total_inverse) * args.epochs) for k, v in inverse_values.items()}
    print(inverse_proportions)

    dataset_list = os.listdir(dataset_path)
    random.shuffle(dataset_list)

    for outer_epoch in range(outer_epochs):
        logger.info(f'OUTER EPOCH {outer_epoch}/{outer_epochs}')

        for i,curr_dataset in enumerate(dataset_list):
            print('loading ',curr_dataset)
            for curr_epoch_iter in range(inverse_proportions[curr_dataset.lower()]):
                
                logger.info(f'EPOCH {curr_epoch_iter}/{inverse_proportions[curr_dataset.lower()]} DATASET {curr_dataset}')

                for j,activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
                    for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                        train_path =dataset_path + '/'+ curr_dataset + '/' + activity + '/' + chunk + '/' + 'train.pt'
                        #  = dataset_path + '/' + curr_dataset + '/' + activity + '/' + chunk + '/' + 'val.pt'
                        train_data = torch.load(train_path,weights_only=False)

                        train_obj = SplitDataset(split='train')

                        train_obj.data = {'train':train_data}

                        train_loader = DataLoader(
                            train_obj, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=2,
                            pin_memory=True,
                            prefetch_factor=2,
                            drop_last=True
                        )
                        
                        print('loaded data')

                        if load==False:

                            model = EMGTransformer(
                                emg_channels=13,
                                emg_window_size=100,
                                kin_state_dim=27,
                                d_model=args.d_model,
                                nhead=args.nhead,
                                num_encoder_layers=args.num_layers,
                                num_decoder_layers=args.num_layers,
                                predict_impedance=args.use_impedance,
                                emg_mask=train_data['masks']['emg'],
                                kinematic_mask=train_data['masks']['kinematic'],
                                kinetic_mask=train_data['masks']['kinetic'],
                                device=args.device
                            ).to(args.device)


                            print(f"{'Layer':<50} {'Shape':<20} {'Params':>15}")
                            print("-" * 85)
                            total = 0
                            for name, param in model.named_parameters():
                                params = param.numel()
                                total += params
                                print(f"{name:<50} {str(param.shape):<20} {params:>15,}")
                            print("-" * 85)
                            logger.info(f"{'Total':<50} {'':<20} {total:>15,}")
                            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
            
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer, T_max=args.epochs, eta_min=args.lr/100
                            )
                            load = True

                        else: 
                            model.emg_mask = torch.Tensor(train_data['masks']['emg']).float().to(model.device)
                            model.kinematic_mask = torch.Tensor(np.tile(train_data['masks']['kinematic'].flatten(), 3)).float().to(model.device)
                            if model.kinetic_mask is not None and train_data['masks']['kinetic'].any():
                                model.kinetic_mask = torch.Tensor(train_data['masks']['kinetic'].flatten()).float().to(model.device)
                                

                        if checkpoint_path != None:
                            checkpoint = torch.load(checkpoint_path)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                            if ['overall_eval_dataset_losses'] in checkpoint.keys():
                                overall_eval_dataset_losses = checkpoint['overall_best_ceiling_losses']
                            if ['overall_best_ceiling_losses'] in checkpoint.keys():
                                overall_best_ceiling_losses = checkpoint['overall_best_ceiling_losses']
                            
                        logger.info(
                            "INFO - TRAINING ON %s | activity=%s | chunk=%s",
                            curr_dataset,
                            activity,
                            chunk
                        )

                        loss_dict=train_val_test_transformer(
                            model, 
                            train_loader, 
                            optimizer = optimizer,
                            scheduler = scheduler,
                            args=args,
                            split_type='train',
                            n_epochs=1,
                            device=args.device,
                            lr=args.lr,
                            use_impedance=args.use_impedance,

                            logger=logger
                        )

            train_data = None

        curr_eval_dataset_losses = {
            'bacek':    {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'gait120':  {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'k2muse':   {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'lencioni': {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'moghadam': {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'moreira':  {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'siat':     {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
            'hu':       {'dataset_total_avg_loss': 0.0, 'dataset_total_torque_loss': 0.0, 'dataset_total_kinematic_loss': 0.0},
        }

        for i,curr_dataset in enumerate(os.listdir((dataset_path))):

            if curr_dataset.lower() not in curr_eval_dataset_losses:
                continue

            print('loading ',curr_dataset)
            dataset_total_avg_loss = 0
            dataset_total_kinematic_loss = 0
            dataset_total_torque_loss = 0
            for j,activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
                for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                    val_path =dataset_path + '/'+ curr_dataset + '/' + activity + '/' + chunk + '/' + 'val.pt'
                    val_data = torch.load(val_path,weights_only=False)

                    val_obj = SplitDataset(split='val')

                    val_obj.data = {'val':val_data}

                    val_loader = DataLoader(
                        val_obj, 
                        batch_size=args.batch_size,
                        shuffle=True, 
                        num_workers=2,
                        pin_memory=True,
                        prefetch_factor=2,
                        drop_last=True
                    )
                    
                    print('loaded data')

                    model.emg_mask = torch.Tensor(val_data['masks']['emg']).float().to(model.device)
                    model.kinematic_mask = torch.Tensor(np.tile(val_data['masks']['kinematic'].flatten(), 3)).float().to(model.device)
                    if model.kinetic_mask is not None and val_data['masks']['kinetic'].any():
                        model.kinetic_mask = torch.Tensor(val_data['masks']['kinetic'].flatten()).float().to(model.device)
                                                        
                    logger.info(
                        "INFO - VALIDATING ON %s | activity=%s | chunk=%s",
                        curr_dataset,
                        activity,
                        chunk
                    )

                    loss_dict=train_val_test_transformer(
                        model, 
                        val_loader, 
                        optimizer = optimizer,
                        scheduler = scheduler,
                        split_type='val',
                        args=args,
                        n_epochs=1,
                        device=args.device,
                        lr=args.lr,
                        use_impedance=args.use_impedance,
                        logger=logger
                    )

                    dataset_total_avg_loss+=loss_dict['avg_total_loss']
                    dataset_total_torque_loss+=loss_dict['avg_torque_loss']
                    dataset_total_kinematic_loss+=loss_dict['avg_kinematic_loss']

            curr_eval_dataset_losses[curr_dataset]['dataset_avg_total_loss'] = dataset_total_avg_loss
            curr_eval_dataset_losses[curr_dataset]['dataset_avg_torque_loss'] = dataset_total_torque_loss
            curr_eval_dataset_losses[curr_dataset]['dataset_avg_kinematic_loss'] = dataset_total_kinematic_loss

        overall_eval_dataset_losses, overall_best_ceiling_losses = model.check_and_save_checkpoints(
            model, optimizer, scheduler, args,
            curr_eval_dataset_losses,
            overall_eval_dataset_losses,
            overall_best_ceiling_losses,
            outer_epoch, logger
        )
        val_data = None


    for i,curr_dataset in enumerate(os.listdir((dataset_path))):
        print('loading ',curr_dataset)
        for j,activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
            for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                test_path =dataset_path + '/'+ curr_dataset + '/' + activity + '/' + chunk + '/' + 'test.pt'
                #  = dataset_path + '/' + curr_dataset + '/' + activity + '/' + chunk + '/' + 'testorch.pt'
                test_data = torch.load(test_path,weights_only=False)

                test_obj = SplitDataset(split='test')

                test_obj.data = {'test':test_data}

                test_loader = DataLoader(
                    test_obj, 
                    batch_size=args.batch_size,
                    shuffle=True, 
                    num_workers=2,
                    pin_memory=True,
                    prefetch_factor=2,
                    drop_last=True
                )
                
                print('loaded data')

                model.emg_mask = torch.Tensor(test_data['masks']['emg']).float().to(model.device)
                model.kinematic_mask = torch.Tensor(np.tile(test_data['masks']['kinematic'].flatten(), 3)).float().to(model.device)
                if model.kinetic_mask is not None and test_data['masks']['kinetic'].any():
                    model.kinetic_mask = torch.Tensor(test_data['masks']['kinetic'].flatten()).float().to(model.device)
                        
                logger.info(
                    "INFO - TESTING ON %s | activity=%s | chunk=%s",
                    curr_dataset,
                    activity,
                    chunk
                )

                train_val_test_transformer(
                    model, 
                    test_loader, 
                    optimizer = optimizer,
                    scheduler = scheduler,
                    args=args,
                    split_type='test',
                    n_epochs=1,
                    device=args.device,
                    lr=args.lr,
                    use_impedance=args.use_impedance,
                    logger=logger
                )
    
    # Create plots
    create_plots(log_file)
    plot_test_data(model=model,test_obj=test_obj)
    
    print("\nDone!")
             
def setup_logger(log_dir='logs'):
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
    create_plots('C:/EMG/logs/training_20260305_183145.log')
    input()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='D:/EMG/postprocessed_datasets',
                       help='Directory containing pickle files')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_impedance', action='store_true',
                       help='Use impedance control with torque prediction',default=True)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=8)
    args = parser.parse_args()
    
    print("Loading and parsing datasets...")

    #create_plots('C:/EMG/logs/training_20260304_191426.log')

    meta_train_transformer_loop(args=args,checkpoint_path=None)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()