import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tqdm import tqdm
from convert2DL import WindowedGaitDataParser, SplitDataset
from datetime import datetime
import re
import matplotlib.pyplot as plt
from  matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def parse_training_logs(log_file_path: str) -> List[Dict]:
    """
    Parse training log file and extract metrics for each training run.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        List of dictionaries, each containing metrics for one training run
    """
    training_runs = []
    current_run = None
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # Check for new training run marker
            if 'INFO - TRAINING ON' in line:
                # Save previous run if it exists
                if current_run is not None and current_run['epochs']:
                    training_runs.append(current_run)
                
                # Extract dataset, activity, chunk info
                match = re.search(r'TRAINING ON\s+(.+?)\s+(.+?)\s+(.+?)(?:\s|$)', line)
                if match:
                    dataset, activity, chunk = match.groups()
                else:
                    # Fallback: extract everything after "TRAINING ON"
                    parts = line.split('TRAINING ON')[-1].strip().split()
                    dataset = parts[0] if len(parts) > 0 else 'unknown'
                    activity = parts[1] if len(parts) > 1 else 'unknown'
                    chunk = parts[2] if len(parts) > 2 else 'unknown'
                
                current_run = {
                    'dataset': dataset,
                    'activity': activity,
                    'chunk': chunk,
                    'epochs': [],
                    'train_loss': [],
                    'train_kin': [],
                    'train_gait': [],
                    'train_torque': [],
                    'val_loss': [],
                    'val_kin': [],
                    'val_gait': [],
                    'val_torque': []
                }
            
            # Check for epoch number
            elif 'Epoch' in line and '/' in line:
                epoch_match = re.search(r'Epoch (\d+)/(\d+)', line)
                if epoch_match and current_run is not None:
                    epoch_num = int(epoch_match.group(1))
                    current_run['epochs'].append(epoch_num)
            
            # Parse training metrics
            elif 'Train Loss:' in line and current_run is not None:
                metrics = re.findall(r'(\d+\.\d+)', line)
                if len(metrics) >= 4:
                    current_run['train_loss'].append(float(metrics[0]))
                    current_run['train_kin'].append(float(metrics[1]))
                    current_run['train_gait'].append(float(metrics[2]))
                    current_run['train_torque'].append(float(metrics[3]))
            
            # Parse validation metrics
            elif 'Val Loss:' in line and current_run is not None:
                metrics = re.findall(r'(\d+\.\d+)', line)
                if len(metrics) >= 4:
                    current_run['val_loss'].append(float(metrics[0]))
                    current_run['val_kin'].append(float(metrics[1]))
                    current_run['val_gait'].append(float(metrics[2]))
                    current_run['val_torque'].append(float(metrics[3]))
    
    # Add the last run
    if current_run is not None and current_run['epochs']:
        training_runs.append(current_run)
    
    return training_runs

def plot_training_runs(training_runs: List[Dict], save_dir: str = 'plots'):
    """
    Create plots for each training run showing loss components over epochs.
    
    Args:
        training_runs: List of training run dictionaries from parse_training_logs
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    for i, run in enumerate(training_runs):
        if not run['epochs']:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Training Run {i+1}: {run['dataset']} - {run['activity']} - {run['chunk']}", 
                     fontsize=14, fontweight='bold')
        
        epochs = run['epochs']
        
        # Plot 1: Total Loss
        axes[0, 0].plot(epochs, run['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, run['val_loss'], 'r-s', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Kinematic Loss
        axes[0, 1].plot(epochs, run['train_kin'], 'b-o', label='Train Kin', linewidth=2)
        axes[0, 1].plot(epochs, run['val_kin'], 'r-s', label='Val Kin', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Kinematic Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gait Loss
        axes[1, 0].plot(epochs, run['train_gait'], 'b-o', label='Train Gait', linewidth=2)
        axes[1, 0].plot(epochs, run['val_gait'], 'r-s', label='Val Gait', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Gait Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Torque Loss
        axes[1, 1].plot(epochs, run['train_torque'], 'b-o', label='Train Torque', linewidth=2)
        axes[1, 1].plot(epochs, run['val_torque'], 'r-s', label='Val Torque', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].set_title('Torque Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{save_dir}/run_{i+1}_{run['dataset']}_{run['activity']}_{run['chunk']}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filename}")

def plot_all_runs_combined(training_runs: List[Dict], save_dir: str = 'plots'):
    """
    Create a combined plot showing all training runs together.
    
    Args:
        training_runs: List of training run dictionaries from parse_training_logs
        save_dir: Directory to save plots
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("All Training Runs Combined", fontsize=14, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_runs)))
    
    for i, run in enumerate(training_runs):
        if not run['epochs']:
            continue
            
        epochs = run['epochs']
        label = f"{run['dataset']}-{run['activity']}-{run['chunk']}"
        
        # Total Loss
        axes[0, 0].plot(epochs, run['val_loss'], '-o', color=colors[i], 
                       label=label, linewidth=2, markersize=4)
        
        # Kinematic Loss
        axes[0, 1].plot(epochs, run['val_kin'], '-o', color=colors[i], 
                       label=label, linewidth=2, markersize=4)
        
        # Gait Loss
        axes[1, 0].plot(epochs, run['val_gait'], '-o', color=colors[i], 
                       label=label, linewidth=2, markersize=4)
        
        # Torque Loss
        axes[1, 1].plot(epochs, run['val_torque'], '-o', color=colors[i], 
                       label=label, linewidth=2, markersize=4)
    
    axes[0, 0].set_title('Total Validation Loss')
    axes[0, 1].set_title('Kinematic Validation Loss')
    axes[1, 0].set_title('Gait Validation Loss')
    axes[1, 1].set_title('Torque Validation Loss')
    
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{save_dir}/all_runs_combined.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined plot: {filename}")

def plot_test_data(model, test_obj):
    print('key test', test_obj.data['test'].keys())
    emg_mask = test_obj.data['test']['masks']['emg']  # shape = 13
    kinetic_mask = test_obj.data['test']['masks']['kinetic']
    kinematic_mask = test_obj.data['test']['masks']['kinematic']

    stride_sequences = []
    current_stride_start = 0

    curr_min = 1e9
    curr_max = 1e-9
    
    for i in range(len(test_obj)):
        data_t = test_obj[i]  # Use __getitem__
        timestep = data_t['metadata']['window_idx']
        curr_min = min(timestep,curr_min)
        curr_max = max(timestep,curr_max)
        
        # When we hit window_idx=0 and we're not at the first sample
        if timestep == 0 and i != 0:
            # Previous stride ended at i-1
            data_prev = test_obj[i-1]
            timestep_prev = data_prev['metadata']['window_idx']
            
            stride_info = {
                'start_idx': current_stride_start,
                'end_idx': i - 1,
                'length': i - current_stride_start,
                'end_window_idx': timestep_prev,
                'metadata': data_prev['metadata']  # Capture patient/activity info
            }

            stride_sequences.append(stride_info)
            
            # New stride starts here
            current_stride_start = i
    
    # Don't forget the last stride!
    if len(test_obj) > 0:
        data_last = test_obj[len(test_obj) - 1]
        timestep_last = data_last['metadata']['window_idx']
        
        stride_info = {
            'start_idx': current_stride_start,
            'end_idx': len(test_obj) - 1,
            'length': len(test_obj) - current_stride_start,
            'end_window_idx': timestep_last,
            'metadata': data_last['metadata']
        }
        stride_sequences.append(stride_info)
    
    if len(stride_sequences) == 0:
        print('No strides found!')
        return
    
    # Find complete strides (window_idx goes 0 to 199, so length=200)
    complete_strides = []
    for stride in stride_sequences:
        if stride['length'] >=198 and stride['end_window_idx'] >=197:
            complete_strides.append(stride)
    
    print(f'Total test strides found: {len(stride_sequences)}')
    print(f'Complete test strides (length=200): {len(complete_strides)}')
    print(f'Max test stride idx found {curr_max}, \n Min test stride idx found {curr_min}')
    
    for i, stride in enumerate(complete_strides):
        print(f'\nComplete stride {i}:')
        print(f"  Indices: {stride['start_idx']} to {stride['end_idx']}")
        print(f"  Patient: {stride['metadata']['patient_id']}")
        print(f"  Activity: {stride['metadata']['activity']}")
        print(f"  Direction: {stride['metadata']['direction']}")
        print(f"  Dataset: {stride['metadata']['dataset']}")
        print(f"  Has torque: {stride['metadata']['has_torque']}")
    
    # TODO: Now visualize one of these complete strides
    if len(complete_strides) > 0:
        # Pick the first complete stride for now
        selected_stride = complete_strides[0]
        visualize_stride(model, test_obj, selected_stride, emg_mask, kinetic_mask, kinematic_mask)
    
    return complete_strides


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D
import torch

def visualize_stride(model, test_obj, stride_info, emg_mask, kinematic_mask, kinetic_mask, 
                     save_path=None, fps=30):
    """
    Visualize a complete stride with EMG, kinematics, and kinetics in 3D
    
    Args:
        model: Your trained model
        test_obj: SplitDataset object
        stride_info: Dict with 'start_idx', 'end_idx', 'metadata'
        emg_mask: Boolean mask for EMG channels (13,)
        kinematic_mask: Boolean mask for kinematic features (27,)
        kinetic_mask: Boolean mask for kinetic features (9,)
        save_path: Optional path to save video
        fps: Frames per second for animation
    """
    
    # Extract the full stride data
    start_idx = stride_info['start_idx']
    end_idx = stride_info['end_idx']
    n_frames = end_idx - start_idx + 1
    
    print(f"Loading stride from indices {start_idx} to {end_idx} ({n_frames} frames)")
    
    # Collect all data for this stride
    emg_data = []
    gt_kin_states = []
    gt_torques = []
    gait_pcts = []
    input_kin_states = []
    input_gait_pcts = []
    
    for i in range(start_idx, end_idx + 1):
        sample = test_obj[i]
        emg_data.append(sample['emg'])
        gt_kin_states.append(sample['target_kin_state'])
        gt_torques.append(sample['target_torque'])
        gait_pcts.append(sample['target_gait_pct'])
        input_kin_states.append(sample['input_kin_state'])
        input_gait_pcts.append(sample['input_gait_pct'])
    
    # Stack into tensors
    emg_data = torch.stack(emg_data).to(model.device)  # (200, 13, 100)
    gt_kin_states = torch.stack(gt_kin_states).to(model.device)  # (200, 27)
    gt_torques = torch.stack(gt_torques).to(model.device)  # (200, 9)
    gait_pcts = torch.stack(gait_pcts).to(model.device)  # (200,)
    input_kin_states = torch.stack(input_kin_states).to(model.device)  # (200, 27)
    input_gait_pcts = torch.stack(input_gait_pcts).to(model.device)  # (200,)
    
    # Run model inference
    print("Running model inference...")
    model.eval()
    with torch.no_grad():
        # Run inference for each timepoint
        pred_kin_states = []
        pred_gait_pcts = []
        pred_impedances = []
        
        for i in range(n_frames):
            outputs = model(
                emg_data[i:i+1], 
                input_kin_states[i:i+1], 
                input_gait_pcts[i:i+1]
            )
            pred_kin_states.append(outputs['pred_kin_state'])
            pred_gait_pcts.append(outputs['pred_gait_pct'])
            if 'pred_impedance' in outputs:
                pred_impedances.append(outputs['pred_impedance'])
        
        pred_kin_states = torch.cat(pred_kin_states, dim=0)  # (200, 27)
        pred_gait_pcts = torch.cat(pred_gait_pcts, dim=0)  # (200,)
        if len(pred_impedances) > 0:
            pred_impedances = torch.cat(pred_impedances, dim=0)  # (200, 9)
        else:
            pred_impedances = None
    
    # Convert to numpy
    emg_np = emg_data.cpu().numpy()
    gt_kin_np = gt_kin_states.cpu().numpy()
    pred_kin_np = pred_kin_states.cpu().numpy()
    gt_torque_np = gt_torques.cpu().numpy()
    pred_impedance_np = pred_impedances.cpu().numpy() if pred_impedances is not None else None
    gait_pct_np = gait_pcts.cpu().numpy()
    
    # Parse kinematic data into angles, velocities, accelerations
    def parse_kinematics(kin_state):
        """
        Parse 27D kinematic state into structured dict
        Layout: [angles(9), omega(9), alpha(9)]
        Each group of 9: [hip_roll, hip_yaw, hip_pitch, knee_roll, knee_yaw, knee_pitch, ankle_roll, ankle_yaw, ankle_pitch]
        """
        angles = kin_state[:, 0:9]
        omega = kin_state[:, 9:18]
        alpha = kin_state[:, 18:27]
        
        return {
            'hip': {
                'angles': angles[:, 0:3],    # roll, yaw, pitch
                'omega': omega[:, 0:3],
                'alpha': alpha[:, 0:3]
            },
            'knee': {
                'angles': angles[:, 3:6],    # roll, yaw, pitch
                'omega': omega[:, 3:6],
                'alpha': alpha[:, 3:6]
            },
            'ankle': {
                'angles': angles[:, 6:9],    # roll, yaw, pitch
                'omega': omega[:, 6:9],
                'alpha': alpha[:, 6:9]
            }
        }
    
    gt_parsed = parse_kinematics(gt_kin_np)
    pred_parsed = parse_kinematics(pred_kin_np)
    
    # Select active EMG channels
    active_emg_channels = np.where(emg_mask)[0]
    print(f"Active EMG channels: {active_emg_channels}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3, 
                          height_ratios=[1.8, 1, 1, 1, 1],
                          left=0.08, right=0.95, top=0.93, bottom=0.12)
    
    # 3D stick figure subplots (top row)
    ax_stick_pred = fig.add_subplot(gs[0, 0], projection='3d')
    ax_stick_gt = fig.add_subplot(gs[0, 1], projection='3d')
    
    # Joint angles (all three axes)
    ax_angles = fig.add_subplot(gs[1, :])
    
    # Torques/Impedance
    ax_torques = fig.add_subplot(gs[2, :])
    
    # EMG (show up to 6 most active channels)
    ax_emg = fig.add_subplot(gs[3, :])
    
    # Gait percentage
    ax_gait = fig.add_subplot(gs[4, :])
    
    # Setup 3D stick figures
    def setup_stick_figure_3d(ax, title):
        ax.set_xlim(-0.3, 0.3)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(0, 1.2)
        ax.set_xlabel('X (m)', fontsize=9)
        ax.set_ylabel('Y (m)', fontsize=9)
        ax.set_zlabel('Z (m)', fontsize=9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.view_init(elev=15, azim=45)  # Good viewing angle
        ax.grid(True, alpha=0.3)
        return ax
    
    setup_stick_figure_3d(ax_stick_pred, 'Predicted')
    setup_stick_figure_3d(ax_stick_gt, 'Ground Truth')
    
    # 3D stick figure helper using rotation matrices
    def rotation_matrix_x(angle):
        """Rotation matrix around X-axis (roll)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0],
                        [0, c, -s],
                        [0, s, c]])
    
    def rotation_matrix_y(angle):
        """Rotation matrix around Y-axis (yaw)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, 0, s],
                        [0, 1, 0],
                        [-s, 0, c]])
    
    def rotation_matrix_z(angle):
        """Rotation matrix around Z-axis (pitch)"""
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0],
                        [s, c, 0],
                        [0, 0, 1]])
    
    def apply_rotation(roll, yaw, pitch):
        """Apply roll-yaw-pitch rotation"""
        R = rotation_matrix_z(pitch) @ rotation_matrix_y(yaw) @ rotation_matrix_x(roll)
        return R
    
    def draw_stick_figure_3d(ax, hip_angles, knee_angles, ankle_angles, color='blue'):
        """
        Draw a 3D stick figure
        hip_angles, knee_angles, ankle_angles: [roll, yaw, pitch] in radians
        """
        # Segment lengths
        thigh_length = 0.4
        shank_length = 0.4
        foot_length = 0.15
        
        # Start at hip (origin)
        hip_pos = np.array([0.0, 0.0, 1.0])
        
        # Initial thigh direction (pointing down)
        thigh_dir = np.array([0.0, 0.0, -thigh_length])
        
        # Apply hip rotation
        R_hip = apply_rotation(hip_angles[0], hip_angles[1], hip_angles[2])
        thigh_rotated = R_hip @ thigh_dir
        knee_pos = hip_pos + thigh_rotated
        
        # Initial shank direction (continuing down)
        shank_dir = np.array([0.0, 0.0, -shank_length])
        
        # Apply knee rotation (relative to thigh orientation)
        R_knee = apply_rotation(knee_angles[0], knee_angles[1], knee_angles[2])
        shank_rotated = R_hip @ R_knee @ shank_dir
        ankle_pos = knee_pos + shank_rotated
        
        # Initial foot direction (forward)
        foot_dir = np.array([foot_length, 0.0, 0.0])
        
        # Apply ankle rotation (relative to shank orientation)
        R_ankle = apply_rotation(ankle_angles[0], ankle_angles[1], ankle_angles[2])
        foot_rotated = R_hip @ R_knee @ R_ankle @ foot_dir
        toe_pos = ankle_pos + foot_rotated
        
        # Draw segments
        lines = []
        # Thigh
        line1, = ax.plot([hip_pos[0], knee_pos[0]], 
                         [hip_pos[1], knee_pos[1]], 
                         [hip_pos[2], knee_pos[2]], 
                         'o-', color=color, linewidth=4, markersize=10)
        # Shank
        line2, = ax.plot([knee_pos[0], ankle_pos[0]], 
                         [knee_pos[1], ankle_pos[1]], 
                         [knee_pos[2], ankle_pos[2]], 
                         'o-', color=color, linewidth=4, markersize=10)
        # Foot
        line3, = ax.plot([ankle_pos[0], toe_pos[0]], 
                         [ankle_pos[1], toe_pos[1]], 
                         [ankle_pos[2], toe_pos[2]], 
                         's-', color=color, linewidth=3, markersize=8)
        
        return [line1, line2, line3]
    
    # Initialize stick figures
    stick_pred_lines = draw_stick_figure_3d(ax_stick_pred, 
                                            pred_parsed['hip']['angles'][0], 
                                            pred_parsed['knee']['angles'][0],
                                            pred_parsed['ankle']['angles'][0],
                                            color='red')
    stick_gt_lines = draw_stick_figure_3d(ax_stick_gt,
                                          gt_parsed['hip']['angles'][0],
                                          gt_parsed['knee']['angles'][0],
                                          gt_parsed['ankle']['angles'][0],
                                          color='blue')
    
    # Setup joint angles plot (showing pitch primarily, but can show all)
    time_vec = np.arange(n_frames)
    ax_angles.set_xlim(0, n_frames)
    ax_angles.set_xlabel('Frame')
    ax_angles.set_ylabel('Joint Angle (rad)')
    ax_angles.set_title('Joint Angles (Roll, Yaw, Pitch)')
    ax_angles.grid(True, alpha=0.3)
    
    # Plot pitch angles (most important for gait) - index 2
    line_hip_gt, = ax_angles.plot(time_vec, gt_parsed['hip']['angles'][:, 2], 
                                   'b-', label='Hip Pitch GT', linewidth=2)
    line_hip_pred, = ax_angles.plot(time_vec, pred_parsed['hip']['angles'][:, 2], 
                                     'r--', label='Hip Pitch Pred', linewidth=2)
    
    line_knee_gt, = ax_angles.plot(time_vec, gt_parsed['knee']['angles'][:, 2], 
                                    'g-', label='Knee Pitch GT', linewidth=2)
    line_knee_pred, = ax_angles.plot(time_vec, pred_parsed['knee']['angles'][:, 2], 
                                      'orange', linestyle='--', label='Knee Pitch Pred', linewidth=2)
    
    line_ankle_gt, = ax_angles.plot(time_vec, gt_parsed['ankle']['angles'][:, 2], 
                                     'purple', label='Ankle Pitch GT', linewidth=2)
    line_ankle_pred, = ax_angles.plot(time_vec, pred_parsed['ankle']['angles'][:, 2], 
                                       'brown', linestyle='--', label='Ankle Pitch Pred', linewidth=2)
    
    vline_angles = ax_angles.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax_angles.legend(loc='upper right', ncol=3, fontsize=8)
    
    # Setup torques plot
    ax_torques.set_xlim(0, n_frames)
    ax_torques.set_xlabel('Frame')
    ax_torques.set_ylabel('Torque/Impedance (N·m)')
    ax_torques.set_title('Joint Torques (Roll, Yaw, Pitch)')
    ax_torques.grid(True, alpha=0.3)
    
    # Show pitch torques primarily (indices 2, 5, 8)
    if pred_impedance_np is not None:
        line_hip_torque_gt, = ax_torques.plot(time_vec, gt_torque_np[:, 2], 
                                              'b-', label='Hip Pitch GT', linewidth=2)
        line_hip_torque_pred, = ax_torques.plot(time_vec, pred_impedance_np[:, 2], 
                                                'r--', label='Hip Pitch Pred', linewidth=2)
        
        line_knee_torque_gt, = ax_torques.plot(time_vec, gt_torque_np[:, 5], 
                                               'g-', label='Knee Pitch GT', linewidth=2)
        line_knee_torque_pred, = ax_torques.plot(time_vec, pred_impedance_np[:, 5], 
                                                 'orange', linestyle='--', label='Knee Pitch Pred', linewidth=2)
        
        line_ankle_torque_gt, = ax_torques.plot(time_vec, gt_torque_np[:, 8], 
                                                'purple', label='Ankle Pitch GT', linewidth=2)
        line_ankle_torque_pred, = ax_torques.plot(time_vec, pred_impedance_np[:, 8], 
                                                  'brown', linestyle='--', label='Ankle Pitch Pred', linewidth=2)
    else:
        # Only ground truth available
        ax_torques.plot(time_vec, gt_torque_np[:, 2], 'b-', label='Hip Pitch', linewidth=2)
        ax_torques.plot(time_vec, gt_torque_np[:, 5], 'g-', label='Knee Pitch', linewidth=2)
        ax_torques.plot(time_vec, gt_torque_np[:, 8], 'purple', label='Ankle Pitch', linewidth=2)
    
    vline_torques = ax_torques.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax_torques.legend(loc='upper right', ncol=3, fontsize=8)
    
    # Setup EMG plot (show top 6 active channels)
    emg_rms = np.sqrt(np.mean(emg_np**2, axis=(0, 2)))  # (13,)
    active_rms = emg_rms[active_emg_channels]
    top_indices = np.argsort(active_rms)[-6:]
    top_channels = active_emg_channels[top_indices]
    
    ax_emg.set_xlim(0, n_frames)
    ax_emg.set_xlabel('Frame')
    ax_emg.set_ylabel('EMG Envelope (normalized)')
    ax_emg.set_title(f'Top 6 Active EMG Channels: {top_channels}')
    ax_emg.grid(True, alpha=0.3)
    
    # Compute EMG envelopes (RMS over the 100-sample window)
    emg_envelopes = np.sqrt(np.mean(emg_np**2, axis=2))  # (200, 13)
    
    # Normalize and plot each channel
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_channels)))
    for i, ch in enumerate(top_channels):
        envelope = emg_envelopes[:, ch]
        envelope_norm = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-8)
        offset = i * 1.2  # Stack vertically
        ax_emg.plot(time_vec, envelope_norm + offset, label=f'Ch {ch}', 
                   linewidth=1.5, color=colors[i])
    
    vline_emg = ax_emg.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
    ax_emg.legend(loc='upper right', ncol=6, fontsize=8)
    ax_emg.set_ylim(-0.5, len(top_channels) * 1.2 + 0.5)
    
    # Setup gait percentage plot
    ax_gait.set_xlim(0, n_frames)
    ax_gait.set_ylim(0, 105)
    ax_gait.set_xlabel('Frame')
    ax_gait.set_ylabel('Gait %')
    ax_gait.set_title('Gait Cycle Percentage')
    ax_gait.grid(True, alpha=0.3)
    ax_gait.plot(time_vec, gait_pct_np, 'k-', linewidth=2, label='Gait %')
    ax_gait.fill_between(time_vec, 0, gait_pct_np, alpha=0.3, color='blue')
    vline_gait = ax_gait.axvline(0, color='red', linewidth=2, linestyle='-', alpha=0.7)
    ax_gait.legend()
    
    # Add metadata text
    metadata = stride_info['metadata']
    fig.suptitle(f"Patient: {metadata['patient_id']} | Activity: {metadata['activity']} | "
                 f"Direction: {metadata['direction']} | Dataset: {metadata['dataset']}", 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Animation update function
    current_frame = [0]
    is_playing = [True]
    
    def update(frame):
        current_frame[0] = frame
        
        # Update 3D stick figures
        for line in stick_pred_lines:
            line.remove()
        for line in stick_gt_lines:
            line.remove()
        
        stick_pred_lines.clear()
        stick_gt_lines.clear()
        
        pred_lines = draw_stick_figure_3d(
            ax_stick_pred,
            pred_parsed['hip']['angles'][frame],
            pred_parsed['knee']['angles'][frame],
            pred_parsed['ankle']['angles'][frame],
            color='red'
        )
        stick_pred_lines.extend(pred_lines)
        
        gt_lines = draw_stick_figure_3d(
            ax_stick_gt,
            gt_parsed['hip']['angles'][frame],
            gt_parsed['knee']['angles'][frame],
            gt_parsed['ankle']['angles'][frame],
            color='blue'
        )
        stick_gt_lines.extend(gt_lines)
        
        # Update vertical lines
        vline_angles.set_xdata([frame, frame])
        vline_torques.set_xdata([frame, frame])
        vline_emg.set_xdata([frame, frame])
        vline_gait.set_xdata([frame, frame])
        
        return stick_pred_lines + stick_gt_lines + [vline_angles, vline_torques, vline_emg, vline_gait]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=True, repeat=True)
    
    # Add interactive controls
    ax_button = plt.axes([0.45, 0.04, 0.1, 0.025])
    btn_play_pause = Button(ax_button, 'Pause')
    
    def toggle_play_pause(event):
        if is_playing[0]:
            anim.event_source.stop()
            btn_play_pause.label.set_text('Play')
            is_playing[0] = False
        else:
            anim.event_source.start()
            btn_play_pause.label.set_text('Pause')
            is_playing[0] = True
    
    btn_play_pause.on_clicked(toggle_play_pause)
    
    # Add frame slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.015])
    slider_frame = Slider(ax_slider, 'Frame', 0, n_frames-1, valinit=0, valstep=1)
    
    def update_slider(val):
        frame = int(slider_frame.val)
        update(frame)
        fig.canvas.draw_idle()
    
    slider_frame.on_changed(update_slider)
    
    # Save if requested
    if save_path is not None:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
        print("Saved!")
    
    plt.show()
    
    return anim

def create_plots(log_file):

    print("Parsing log file...")
    training_runs = parse_training_logs(log_file)
    
    print(f"Found {len(training_runs)} training runs")

    print("\nGenerating plots...")
    plot_training_runs(training_runs)
    plot_all_runs_combined(training_runs)
