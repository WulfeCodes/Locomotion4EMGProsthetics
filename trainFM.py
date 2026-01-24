import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
from pathlib import Path
from diffusers import DDPMScheduler
from diffusers.models import UNet1DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from scipy.signal import welch
import math
from convert2DL import WindowedGaitDataParser

#TODO what is the best multi head design? 
#NOTE should there be a learned gait for masking? 

class EMGTransformer(nn.Module):
    """
    Transformer model for EMG-based gait prediction.
    Processes EMG windows + kinematic state to predict next kinematic state.
    """
    
    
    def __init__(self, 
                 emg_channels=13,
                 emg_window_size=200,
                 kin_state_dim=27,  # 9 angles + 9 omega + 9 alpha
                 d_model=50,#change with scaling
                 nhead=2,
                 num_encoder_layers=1,
                 num_decoder_layers=1,
                 dim_feedforward=1024,
                 dropout=0.1,
                 predict_impedance=True,
                 kinematic_mask=np.zeros((3,3)),
                 kinetic_mask=None,
                 emg_mask = np.zeros(13,),
                 device = 'cuda'):
        super().__init__()
        
        self.emg_channels = emg_channels
        self.emg_window_size = emg_window_size
        self.kin_state_dim = kin_state_dim
        self.d_model = d_model
        self.emg_conv_ip_channels = 16
        self.emg_conv_hidden_channels = 32
        self.device = device

        self.predict_impedance = predict_impedance

        self.emg_mask = torch.from_numpy(emg_mask).to(device)
        self.kinematic_mask = torch.from_numpy(np.tile(kinematic_mask.flatten(), 3)).to(device)
        if kinetic_mask is not None and kinetic_mask.any():
            self.kinetic_mask = torch.from_numpy(np.tile(kinetic_mask.flatten(),3)).to(device)
        else: self.kinetic_mask = torch.from_numpy(np.zeros((27))).to(device)
        
        # EMG embedding: Conv1D to extract features from EMG time series
        self.emg_conv = nn.Sequential(
            nn.Conv1d(self.emg_channels, self.emg_conv_ip_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(self.emg_conv_ip_channels, self.emg_conv_hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.MaxPool1d(2),
            nn.Conv1d(self.emg_conv_hidden_channels, self.emg_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Linear(self.emg_window_size,d_model),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        
        # Calculate sequence length after convolutions
        self.emg_seq_len = emg_window_size // 4  # After 2 maxpool layers
        
        # Kinematic state embedding
        self.kin_embedding = nn.Sequential(
            nn.Linear(kin_state_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gait percentage embedding
        self.gait_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=self.emg_seq_len + 2)
        
        # Transformer
        #dim_feedforward expands through a 2 layer NN but back to d_model for add + norm operations

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output heads
        self.kin_output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, kin_state_dim)
        )
        
        self.gait_output = nn.Sequential(
            nn.Linear(d_model, dim_feedforward //2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1)
        )
        
        if predict_impedance:
            # Impedance parameters: K, C, M for 9 joints
            self.impedance_output = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, 27),  # 9 joints * 3 parameters (K, C, M)
                nn.Softplus()  # Ensure positive impedance values
            )
    
    def forward(self, emg, input_kin_state, input_gait_pct):
        """
        Args:
            emg: (batch, emg_channels, emg_window_size)
            input_kin_state: (batch, 27) - current angles, omega, alpha
            input_gait_pct: (batch, 1) - current gait percentage
        
        Returns:
            pred_kin_state: (batch, 27) - predicted next state
            pred_gait_pct: (batch, 1) - predicted next gait %
            pred_impedance: (batch, 27) - K, C, M if predict_impedance=True
        """
        #print('shape checks', emg.shape,input_kin_state.shape,input_gait_pct.shape,"\n should be 13,200; 27, 1",self.emg_mask.shape)
        # Process EMG: (batch, channels, time) -> (batch, time', d_model)
        # print('shaper',input_kin_state.shape,(input_kin_state.reshape,(self.kin_embedding(input_kin_state.flatten().unsqueeze(dim=1) * self.kinetic_mask.bool().unsqueeze(dim=0).unsqueeze(dim=0))).shape)
        emg_features = self.emg_conv(emg * self.emg_mask.float().unsqueeze(dim=0).unsqueeze(dim=-1))  # (batch, d_model, emg_seq_len)
        # emg_features = emg_features.permute(0, 2, 1)  # (batch, emg_seq_len, d_model)
        
        #print('dungaree:',emg_features.shape,type(emg_features),'\n',input_kin_state.shape,type(input_kin_state))
        # Process kinematic state and gait

        kin_features = self.kin_embedding(input_kin_state.reshape(input_kin_state.shape[0],1,input_kin_state.shape[1]*input_kin_state.shape[-1]) * self.kinetic_mask.float().unsqueeze(dim=0).unsqueeze(dim=0)) # (batch, 1, d_model)
        gait_features = self.gait_embedding(input_gait_pct.unsqueeze(dim=-1))  # (batch, 1, d_model)

        # Combine into encoder input sequence
        encoder_input = emg_features  # (batch, emg_seq_len+2, d_model)
        encoder_input = self.pos_encoder(encoder_input)
        
        # Create decoder input (learnable query)
        decoder_input = torch.cat([kin_features, gait_features],dim=1) 
        
        # Right before transformer_output = self.transformer(encoder_input, decoder_input)

        # Transformer
        transformer_output = self.transformer(encoder_input, decoder_input)  # (batch, 1, d_model)
        output_features = transformer_output  # (batch,2,d_model)
        #dim 1 is supposed to represent the kinematic positional encoding whilst the -1 is the gait
        
        # Predictions
        pred_kin_state = self.kin_output(output_features[:,0,:])
        pred_gait_pct = self.gait_output(output_features[:,1,:])
        outputs = {
            'pred_kin_state': pred_kin_state,
            'pred_gait_pct': pred_gait_pct
        }
        
        #NOTE only passing the kinematic information, impedance may benefit from gait information
        if self.predict_impedance:
            pred_impedance = self.impedance_output(output_features[:,0,:])
            outputs['pred_impedance'] = pred_impedance
        
        return outputs


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

def compute_impedance_torque(input_kin_state, pred_kin_state, pred_impedance):
    """
    Compute predicted torque using impedance control formula.
    
    Args:
        input_kin_state: (batch, 27) - current state [angles, omega, alpha]
        pred_kin_state: (batch, 27) - desired state [angles, omega, alpha]
        pred_impedance: (batch, 27) - [K1..K9, C1..C9, M1..M9]
    
    Returns:
        pred_torque: (batch, 9) - predicted joint torques
    """
    # Split states
    theta_curr = input_kin_state[:, :9]
    omega_curr = input_kin_state[:, 9:18]
    alpha_curr = input_kin_state[:, 18:27]
    
    theta_des = pred_kin_state[:, :9]
    omega_des = pred_kin_state[:, 9:18]
    alpha_des = pred_kin_state[:, 18:27]
    
    # Split impedance parameters
    K = pred_impedance[:, :9]
    C = pred_impedance[:, 9:18]
    M = pred_impedance[:, 18:27]
    
    # Impedance control law: τ = K(θ_des - θ) + C(ω_des - ω) + M(α_des - α)
    pred_torque = (K * (theta_des - theta_curr) + 
                   C * (omega_des - omega_curr) + 
                   M * (alpha_des - alpha_curr))
    
    return pred_torque

def train_transformer(model, train_loader, val_loader, n_epochs=50, 
                      device='cuda', lr=1e-4, use_impedance=False,
                      lambda_kin=1.0, lambda_gait=0.5, lambda_torque=1.0):
    """Train the EMGTransformer model."""


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr/100
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_kin_loss = 0
        train_gait_loss = 0
        train_torque_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            # Move to device
            emg = batch['emg'].to(device)
            input_kin_state = batch['input_kin_state'].to(device)
            input_gait_pct = batch['input_gait_pct'].to(device)
            target_kin_state = batch['target_kin_state'].to(device)
            target_gait_pct = batch['target_gait_pct'].to(device)
            target_torque = batch['target_torque'].to(device)
            has_torque = batch['has_torque']
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(emg, input_kin_state, input_gait_pct)
            pred_kin_state = outputs['pred_kin_state']
            pred_gait_pct = outputs['pred_gait_pct']
            
            # Kinematic loss
            loss_kin = nn.functional.mse_loss(pred_kin_state, target_kin_state)
            
            # Gait percentage loss
            loss_gait = nn.functional.mse_loss(pred_gait_pct, target_gait_pct)
            
            # Total loss
            loss = lambda_kin * loss_kin + lambda_gait * loss_gait
            
            # Impedance/torque loss (if applicable)
            if use_impedance and 'pred_impedance' in outputs:
                pred_impedance = outputs['pred_impedance']
                pred_torque = compute_impedance_torque(
                    input_kin_state, pred_kin_state, pred_impedance
                )
                
                # Only compute torque loss for samples with ground truth torque
                if has_torque.any():
                    torque_mask = has_torque.to(device)
                    loss_torque = nn.functional.mse_loss(
                        pred_torque[torque_mask], 
                        target_torque[torque_mask]
                    )
                    loss = loss + lambda_torque * loss_torque
                    train_torque_loss += loss_torque.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print('stepped')
            
            train_loss += loss.item()
            train_kin_loss += loss_kin.item()
            train_gait_loss += loss_gait.item()
            n_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_kin_loss = 0
        val_gait_loss = 0
        val_torque_loss = 0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                emg = batch['emg'].to(device)
                
                outputs = model(emg, input_kin_state, input_gait_pct)
                pred_kin_state = outputs['pred_kin_state']
                pred_gait_pct = outputs['pred_gait_pct']
                
                loss_kin = nn.functional.mse_loss(pred_kin_state, target_kin_state)
                loss_gait = nn.functional.mse_loss(pred_gait_pct, target_gait_pct)
                loss = lambda_kin * loss_kin + lambda_gait * loss_gait
                
                if use_impedance and 'pred_impedance' in outputs:
                    pred_impedance = outputs['pred_impedance']
                    pred_torque = compute_impedance_torque(
                        input_kin_state, pred_kin_state, pred_impedance
                    )
                    if has_torque.any():
                        torque_mask = has_torque.to(device)
                        loss_torque = nn.functional.mse_loss(
                            pred_torque[torque_mask], 
                            target_torque[torque_mask]
                        )
                        loss = loss + lambda_torque * loss_torque
                        val_torque_loss += loss_torque.item()
                
                val_loss += loss.item()
                val_kin_loss += loss_kin.item()
                val_gait_loss += loss_gait.item()
                n_val_batches += 1
        
        # Update learning rate (OUTSIDE torch.no_grad, but still in epoch loop)
        scheduler.step()
        
        # Print statistics (INSIDE epoch loop)
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss/n_batches:.4f} | '
              f'Kin: {train_kin_loss/n_batches:.4f} | '
              f'Gait: {train_gait_loss/n_batches:.4f}', end='')
        if use_impedance:
            print(f' | Torque: {train_torque_loss/n_batches:.4f}', end='')
        print()
        
        print(f'Val Loss: {val_loss/n_val_batches:.4f} | '
              f'Kin: {val_kin_loss/n_val_batches:.4f} | '
              f'Gait: {val_gait_loss/n_val_batches:.4f}', end='')
        if use_impedance:
            print(f' | Torque: {val_torque_loss/n_val_batches:.4f}', end='')
        print()
        
        # Save best model (INSIDE epoch loop)
        if val_loss/n_val_batches < best_val_loss:
            best_val_loss = val_loss/n_val_batches

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_transformer_model.pth')
            print('✓ Saved best model')
    
    return model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='D:/EMG/postprocessed_datasets',
                       help='Directory containing pickle files')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_impedance', action='store_true',
                       help='Use impedance control with torque prediction')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    args = parser.parse_args()
    
    print("Loading and parsing datasets...")
    
    # Initialize parser and load all datasets
    parser_obj = WindowedGaitDataParser(
        window_size=200,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        input_dir=args.pkl_dir,
        device=args.device
    )
    
    #TODO load and train datasets one by one
    parser_obj.parse_criekinge("D:/EMG/postprocessed_datasets/criekinge.pkl")
    currMasks=parser_obj.dataset_masks['criekinge']
    #parser_obj.convert_all()
    
    # Create datasets
    # train_dataset = GaitDataset(parser_obj.data, split='train')
    # val_dataset = GaitDataset(parser_obj.data, split='val')
    # test_dataset = GaitDataset(parser_obj.data, split='test')

    # Create dataloaders
    train_loader = DataLoader(
        parser_obj.trainDataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

        
    val_loader = DataLoader(
        parser_obj.valDataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        parser_obj.testDataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(parser_obj.trainDataset)}")
    print(f"  Val: {len(parser_obj.valDataset)}")
    print(f"  Test: {len(parser_obj.testDataset)}")
    parser_obj.trainDataset.verify_lengths()
    parser_obj.testDataset.verify_lengths()
    parser_obj.valDataset.verify_lengths()

    #clearing some RAM
    del parser_obj
    torch.cuda.empty_cache()  # If using GPU
    import gc; gc.collect()
    

    # Create model
    model = EMGTransformer(
        emg_channels=13,
        emg_window_size=200,
        kin_state_dim=27,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        predict_impedance=args.use_impedance,
        emg_mask=currMasks['emg'],
        kinematic_mask=currMasks['kinematic'],
        kinetic_mask=currMasks['kinetic']
    ).to(args.device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    #define model.ex_mask before training on a given dataset
    print("\nStarting training...")
    train_transformer(
        model, 
        train_loader, 
        val_loader,
        n_epochs=args.epochs,
        device=args.device,
        lr=args.lr,
        use_impedance=args.use_impedance
    )
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()