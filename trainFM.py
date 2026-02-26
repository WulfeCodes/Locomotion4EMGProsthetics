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
from visualizer import create_plots,plot_test_data

def soft_update(source, target, tau):
    """
    Soft update of the target network parameters.
    θ_target = τ * θ_source + (1 - τ) * θ_target
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

class QNetwork(nn.Module):
    def __init__(self,input_size,h_dim,output_size,dropout=0.1):
        super().__init__()
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
                                       )
        
        def forward(input):
            return self.q_network(input)
        
    def save_checkpoint(self):
        os.makedirs(self.checkpoint_directory, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

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
        self.replay_buffer = ReplayBuffer(max_size=int(1e6),input_shape=int(13*100+27),n_actions=int(9))
        
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
        
        # FIX 2: Initialize weights properly

        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.initorch.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.initorch.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.initorch.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.initorch.constant_(m.bias, 0)
    
    def forward(self, emg, input_kin_state, input_gait_pct=None):
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
        if input_gait_pct:
            gait_features = self.gait_embedding(input_gait_pctorch.unsqueeze(1))  # (batch, 1, d_model)

        # Combine into encoder input sequence
        encoder_input = emg_features
        encoder_input = self.pos_encoder(encoder_input)
        
        # Create decoder input
        if input_gait_pct:
            decoder_input = torch.cat([kin_features, gait_features], dim=1)
        else: 
            decoder_input = kin_features
        
        # Transformer
        transformer_output = self.transformer(encoder_input, decoder_input)
        
        # Predictions
        pred_kin_state = self.kin_output(transformer_output[:, 0, :])

        if input_gait_pct:
            pred_gait_pct = self.gait_output(transformer_output[:, 1, :])

        outputs = {
            'pred_kin_state': pred_kin_state,
        }

        if input_gait_pct:
            outputs['pred_gait_pct'] = pred_gait_pct
        
        if self.predict_impedance:
            pred_impedance = self.impedance_output(transformer_output[:, 0, :])
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

def train_sac(Policy,QNetwork_base1,QNetwork_base2,QNetwork_target1,QNetwork_target2,
              replay_buff,training_batch_size,training_losses,sample_batch_size=256):
    
    gamma = 0.99
    tau = 0.05  # Soft update coefficient
    
    while training_iterations < training_batch_size:
        # Ensure enough samples in replay buffer

        batch_count = 0

        while batch_count<sample_batch_size:
        # Sample from replay buffer
            states, states_, actions, rewards, dones = replay_buff.sample_buffer(sample_batch_size)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float32)
            states_ = torch.tensor(states_, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)

            # Critic (Q-network) update
            #the torch.no_grads dont contribute to the gradient computation
            with torch.no_grad():
                # Sample next actions
                next_actions, next_log_probs = Policy.sampleAction(states_)
                
                # Compute target Q-values
                target1_q = QNetwork_target1.forward(states_, next_actions)
                target2_q = QNetwork_target2.forward(states_, next_actions)
                target_q = torch.min(target1_q, target2_q)
                # next_log_probs = next_log_probs.unsqueeze(-1)
                # rewards = rewards.unsqueeze(-1)
                # dones = dones.unsqueeze(-1)        
        
                # Compute target values
                y = rewards + gamma * (1 - dones) * (target_q - Policy.alpha * next_log_probs.detach())
                #detaching irrelevant calcs in the backprop update!
            # Current Q-values

                #PRINT DEBUG STATEMENTS

                # print(states.shape, states_.shape, rewards.shape, dones.shape,next_actions.shape,next_log_probs.shape, target_q.shape, y.shape)

            current_q1 = QNetwork_base1.forward(states, actions)
            current_q2 = QNetwork_base2.forward(states, actions)
            # print(current_q1,current_q2,target_q)
            # Q-network losses
            q1_loss += pow((y-current_q1),2)
            q2_loss += pow((y-current_q2),2)

            # Actor loss
            sampled_actions, log_probs = Policy.sampleAction(states)
            # print("action dim, log prob dim:", sampled_actions.shape, log_probs.shape)

            with torch.no_grad():
                q1_vals = QNetwork_base1.forward(states, sampled_actions)
                q2_vals = QNetwork_base2.forward(states, sampled_actions)
                q_vals = torch.min(q1_vals, q2_vals)
                # print("q_vals shape", q_vals.shape)
            
                #detaching irrelevant calcs in the backprop update!
            actor_loss += ((Policy.log_alpha.unsqueeze(0)).detach() * log_probs - q_vals)

            #loss_alpha = -a(log_pi(a|s)+ H)

                #detaching irrelevant calcs in the backprop update!

            alpha_loss += -((Policy.log_alpha.unsqueeze(0)) * (log_probs.detach() + Policy.target_entropy))
            # print("target entropy squozed shaped:", Policy.target_entropy.shape)
            # print("y output shape:",y.shape)
            # print("targ q shape", target_q.shape)
            # print("q base shape:", q_vals.shape)
            # print("actor log prob and q base shape",log_probs.shape,q_vals.shape)
            # print("actor loss shape",actor_loss.shape)
            # print("alpha shapes: ", (Policy.log_alpha.unsqueeze(0)).shape, (Policy.alpha.unsqueeze(0)).shape, alpha_loss.shape)
            batch_count+=1


        # if training_iterations % 10 == 0:
        #     print(f"Training Iteration: {training_iterations}")
        actor_loss/=256.0
        q2_loss/=256.0
        q1_loss/=256.0
        alpha_loss/=256.0

        # print("q1 and q2 loss", q1_loss.shape, q2_loss.shape)


        training_losses['actor_loss'].append(actor_loss.item())
        training_losses['q1_loss'].append(q1_loss.item())
        training_losses['q2_loss'].append(q2_loss.item())
        training_losses['log_probs'].append(log_probs.item())

        QNetwork_base1.optimizer.zero_grad()
        q1_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(QNetwork_base1.parameters(), max_norm=0.5)
        QNetwork_base1.optimizer.step()


        QNetwork_base2.optimizer.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(QNetwork_base2.parameters(), max_norm=0.5)
        QNetwork_base2.optimizer.step()
        # Backward pass
        #clipping the gradients helps prevent exploding or diminished grad updates


        # Then policy
        Policy.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(Policy.parameters(), max_norm=1.0)
        Policy.optimizer.step()

        # Finally alpha
        Policy.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_norm_([Policy.log_alpha], max_norm=0.5)  
        Policy.alpha_optimizer.step()

        if target2_q<target1_q:
            soft_update(QNetwork_base2, QNetwork_target1, tau)
            soft_update(QNetwork_base2, QNetwork_target2, tau)
        
        else:
            soft_update(QNetwork_base1, QNetwork_target1, tau)
            soft_update(QNetwork_base1, QNetwork_target2, tau)
        
        Policy.alpha = Policy.log_alpha.exp()

        actor_loss=0
        q1_loss=0
        q2_loss=0
        alpha_loss=0
        training_iterations += 1

        # if np.mean(training_losses['actor_loss']) < 2 and np.mean(training_losses['q1_loss'])<20 and np.mean(training_losses['actor_loss']) <20:
        #     print("break!")
        #     print(np.mean(training_losses['actor_loss']), np.mean(training_losses['q1_loss']), np.mean(training_losses['q2_loss']) )
        #     break
    # Soft update of target networks

    # Logging

    # Print progress

    # Final training summary
    print("\n--- Training Phase Complete ---")
    print("Average Losses:")
    print(f"  Actor Loss: {np.mean(training_losses['actor_loss']):.4f}")
    print(f"  Q1 Network Loss: {np.mean(training_losses['q1_loss']):.4f}")
    print(f"  Q2 Network Loss: {np.mean(training_losses['q2_loss']):.4f}")
    print(f"Alpha: {Policy.alpha.item():.3f}, Entropy: {-log_probs.mean().item():.3f}")
    visualizer.plot()
    # Save networks
    Policy.save_checkpoint()
    QNetwork_base1.save_checkpoint()
    QNetwork_base2.save_checkpoint()
    QNetwork_target1.save_checkpoint()
    QNetwork_target2.save_checkpoint()
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
                      lambda_kin=1.2, lambda_gait=0.5, lambda_torque=1.0,logger=None):
    
    if logger is None:
        logger,log_file = setup_logger()
    
    best_split_loss = float('inf')

    for epoch in range(n_epochs):
        
        # Validation
        if split_type == 'val' or split_type == 'test':
            model.eval()
        elif split_type=='train':
            model.train()
        split_loss = 0
        split_kin_loss = 0
        split_gait_loss = 0
        split_torque_loss = 0
        n_split_batches = 0

        if len(split_loader)==0: 
            logger.info(f'length of split data is 0, skipping..')
            continue
        
        split_pbar = tqdm(split_loader, desc=f'Epoch {epoch+1}/{n_epochs} [split]')

        with torch.no_grad():
            for batch in split_pbar:
                emg = batch['emg'].to(device)
                input_kin_state = batch['input_kin_state'].to(device)
                input_gait_pct = batch['input_gait_pct'].to(device)
                target_kin_state = batch['target_kin_state'].to(device)
                target_gait_pct = batch['target_gait_pct'].to(device)
                target_torque = batch['target_torque'].to(device)
                has_torque = batch['has_torque']
                
                outputs = model(emg, input_kin_state, input_gait_pct)
                pred_kin_state = outputs['pred_kin_state']
                pred_gait_pct = outputs['pred_gait_pct']
                
                loss_kin = model.masked_mse_loss(pred_kin_state, target_kin_state, model.kinematic_mask)
                loss_gait = nn.functional.mse_loss(pred_gait_pct, target_gait_pct)
                loss = lambda_kin * loss_kin + lambda_gait * loss_gait
                
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

                        loss = loss + lambda_torque * loss_torque
                        train_torque_loss += loss_torque.item()
                
                split_loss += loss.item()
                split_kin_loss += loss_kin.item()
                split_gait_loss += loss_gaitorch.item()
                n_split_batches += 1
        
        scheduler.step()
        
        # Print statistics
        avg_split_loss = split_loss / max(n_split_batches, 1)
        
        logger.info(f'\nEpoch {epoch+1}/{n_epochs}')

        split_log = (f'{split_type} Loss: {avg_split_loss:.4f} | '
                   f'Kin: {split_kin_loss/max(n_split_batches,1):.4f} | '
                   f'Gait: {split_gait_loss/max(n_split_batches,1):.4f}')
        if use_impedance:
            split_log += f' | Torque: {split_torque_loss/max(n_split_batches,1):.4f}'
        logger.info(split_log)
        
        # Save best model
        if split_type=='val':
            if avg_split_loss < best_val_loss:
                best_val_loss = avg_split_loss

                torch.save({
                    'model_config': {'num_layers':args.num_layers,'d_model':args.d_model,'nhead':args.nhead},
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, 'best_transformer_model.pth')
                logger.info('Saved best model')
    
    return model


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

def meta_train_transformer_loop(args,dataset_path = 'D:/EMG/ML_datasets/run1',outer_epochs=2,checkpoint_path=None):
    load = False

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

    proportion_mapping = {}

    inverse_values = {k: 1/v for k, v in datasets.items()}
    total_inverse = sum(inverse_values.values())

    # Normalize to percentages and scales to number of epochs
    #sum of data will get args.epochs with each dataset getting their inverse normalized proportion
    inverse_proportions = {k: math.ceil((v/total_inverse) * args.epochs) for k, v in inverse_values.items()}

    for outer_epoch in range(outer_epochs):

        for i,curr_dataset in enumerate(os.listdir((dataset_path))):
            print('loading ',curr_dataset)

            for curr_epoch_iter in range(inverse_proportions[curr_datasetorch.lower()]):
                print('EPOCH ',curr_epoch_iter, 'DATASET ',curr_dataset)

                for j,activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
                    for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                        train_path =dataset_path + '/'+ curr_dataset + '/' + activity + '/' + chunk + '/' + 'train.pt'
                        #  = dataset_path + '/' + curr_dataset + '/' + activity + '/' + chunk + '/' + 'val.pt'
                        train_data = torch.load(train_path)

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


                            logger,log_file = setup_logger()

                            print(f"{'Layer':<50} {'Shape':<20} {'Params':>15}")
                            print("-" * 85)
                            total = 0
                            for name, param in model.named_parameters():
                                params = param.numel()
                                total += params
                                print(f"{name:<50} {str(param.shape):<20} {params:>15,}")
                            print("-" * 85)
                            logger.info(f"{'Total':<50} {'':<20} {total:>15,}")
                            load = True

                        else: 
                            model.emg_mask = torch.Tensor(train_data['masks']['emg']).float().to(model.device)
                            model.kinematic_mask = torch.Tensor(np.tile(train_data['masks']['kinematic'].flatten(), 3)).float().to(model.device)
                            if model.kinetic_mask is not None and train_data['masks']['kinetic'].any():
                                model.kinetic_mask = torch.Tensor(train_data['masks']['kinetic'].flatten()).float().to(model.device)
                                
                        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
        
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=args.epochs, eta_min=args.lr/100
                        )

                        if checkpoint_path != None:
                            checkpoint = torch.load(checkpoint_path)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            
                        logger.info(
                            "INFO - TRAINING ON %s | activity=%s | chunk=%s",
                            curr_dataset,
                            activity,
                            chunk
                        )

                        train_val_test_transformer(
                            model, 
                            train_loader, 
                            optimizer = optimizer,
                            scheduler = scheduler,
                            split_type='train',
                            n_epochs=1,
                            device=args.device,
                            lr=args.lr,
                            use_impedance=args.use_impedance,
                            logger=logger
                        )

            train_data = None

        for i,curr_dataset in enumerate(os.listdir((dataset_path))):
            print('loading ',curr_dataset)
            for j,activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
                for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                    val_path =dataset_path + '/'+ curr_dataset + '/' + activity + '/' + chunk + '/' + 'val.pt'
                    #  = dataset_path + '/' + curr_dataset + '/' + activity + '/' + chunk + '/' + 'val.pt'
                    val_data = torch.load(val_path)


                    val_obj = SplitDataset(split='val')

                    val_obj.data = {'val':val_data}

                    val_loader = DataLoader(
                        val_obj, 
                        batch_size=args.batch_size,
                        shuffle=True, 
                        split_type='val',
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
                                
                    if checkpoint_path != None:
                        checkpoint = torch.load(checkpoint_path)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                    logger.info(
                        "INFO - VALIDATING ON %s | activity=%s | chunk=%s",
                        curr_dataset,
                        activity,
                        chunk
                    )

                    train_val_test_transformer(
                        model, 
                        val_loader, 
                        optimizer = optimizer,
                        scheduler = scheduler,
                        args=args,
                        n_epochs=1,
                        device=args.device,
                        lr=args.lr,
                        use_impedance=args.use_impedance,
                        logger=logger
                    )
        val_data = None

    for i,curr_dataset in enumerate(os.listdir((dataset_path))):
        print('loading ',curr_dataset)
        for j,activity in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}')):
            for k, chunk in enumerate(os.listdir(f'{dataset_path}/{curr_dataset}/{activity}')):
                test_path =dataset_path + '/'+ curr_dataset + '/' + activity + '/' + chunk + '/' + 'testorch.pt'
                #  = dataset_path + '/' + curr_dataset + '/' + activity + '/' + chunk + '/' + 'testorch.pt'
                test_data = torch.load(test_path)

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
                        
                if checkpoint_path != None:
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
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

    #create_plots('C:/EMG/logs/training_20260208_004649.log')

    meta_train_transformer_loop(args=args,checkpoint_path='C:/EMG/best_transformer_model.pth')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()