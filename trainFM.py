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
from tqdm import tqdm
from convert2DL import WindowedGaitDataParser
import gc
import logging
from datetime import datetime

class EMGTransformer(nn.Module):
    """
    Transformer model for EMG-based gait prediction.
    Processes EMG windows + kinematic state to predict next kinematic state.
    """
    
    def __init__(self, 
                 emg_channels=13,
                 emg_window_size=200,
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
            self.kinetic_mask = torch.Tensor(np.tile(kinetic_mask.flatten(),3)).float().to(device)
        else:
            self.kinetic_mask = torch.Tensor(np.zeros((27))).float().to(device)
        
        # FIX 1: Improved EMG embedding with better initialization
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
            nn.Sigmoid()  # Gait percentage should be 0-1
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
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, emg, input_kin_state, input_gait_pct):
        """
        Args:
            emg: (batch, emg_channels, emg_window_size)
            input_kin_state: (batch, 27) - current angles, omega, alpha
            input_gait_pct: (batch, 1) - current gait percentage
        
        Returns:
            Dictionary with predictions
        """

        # Apply masks properly (element-wise multiplication)
        emg_masked = emg * self.emg_mask.view(1, -1, 1)
        
        # Process EMG
        emg_features = self.emg_conv(emg_masked)  # (batch, d_model, emg_seq_len)
        
        # Process kinematic state and gait
        kin_masked = input_kin_state * self.kinetic_mask.view(1, -1)
        kin_features = self.kin_embedding(kin_masked.unsqueeze(1))  # (batch, 1, d_model)
        gait_features = self.gait_embedding(input_gait_pct.unsqueeze(1))  # (batch, 1, d_model)

        # Combine into encoder input sequence
        encoder_input = emg_features
        encoder_input = self.pos_encoder(encoder_input)
        
        # Create decoder input
        decoder_input = torch.cat([kin_features, gait_features], dim=1)
        
        # Transformer
        transformer_output = self.transformer(encoder_input, decoder_input)
        
        # Predictions
        pred_kin_state = self.kin_output(transformer_output[:, 0, :])
        pred_gait_pct = self.gait_output(transformer_output[:, 1, :])
        
        outputs = {
            'pred_kin_state': pred_kin_state,
            'pred_gait_pct': pred_gait_pct
        }
        
        if self.predict_impedance:
            pred_impedance = self.impedance_output(transformer_output[:, 0, :])
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


def train_transformer(model, train_loader, val_loader,test_loader ,n_epochs=50, 
                      device='cuda', lr=1e-4, use_impedance=False,
                      lambda_kin=1.0, lambda_gait=0.5, lambda_torque=1.0,logger=None):
    """Train the EMGTransformer model with NaN detection."""

    if logger is None:
        logger = setup_logger()

    # FIX 4: Use gradient clipping and adjust optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr/100
    )
    
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_kin_loss = 0
        train_gait_loss = 0
        train_torque_loss = 0
        n_batches = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move to device

            emg = batch['emg'].to(device)
            input_kin_state = batch['input_kin_state'].to(device,non_blocking=True)
            input_gait_pct = batch['input_gait_pct'].to(device)
            target_kin_state = batch['target_kin_state'].to(device)
            target_gait_pct = batch['target_gait_pct'].to(device)
            target_torque = batch['target_torque'].to(device)
            has_torque = batch['has_torque']
            
            # FIX 5: Check for NaN in inputs
            validate_batch(batch=batch,batch_idx=batch_idx)
            if torch.isnan(emg).any():
                logger.warning(f"NaN in batch EMG {batch_idx}, skipping... {emg}")
                if torch.isnan(input_kin_state).any():
                    logger.warning(f"NaN in batch Kinetic {batch_idx}, skipping... {input_kin_state}")
                    continue
                continue               

            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(emg, input_kin_state, input_gait_pct)
            pred_kin_state = outputs['pred_kin_state']
            pred_gait_pct = outputs['pred_gait_pct']
            
            # FIX 6: Check for NaN in outputs
            if torch.isnan(pred_kin_state).any() or torch.isnan(pred_gait_pct).any():
                logger.warning(f"NaN in predictions at batch {batch_idx}")
                logger.warning(f"  EMG range: [{emg.min():.4f}, {emg.max():.4f}]")
                logger.warning(f"  Kin state range: [{input_kin_state.min():.4f}, {input_kin_state.max():.4f}]")
                continue
            
            # Kinematic loss
            loss_kin = nn.functional.mse_loss(pred_kin_state, target_kin_state)
            
            # Gait percentage loss
            loss_gait = nn.functional.mse_loss(pred_gait_pct, target_gait_pct)
            
            # Total loss
            loss = lambda_kin * loss_kin + lambda_gait * loss_gait
            losser = None
            # Impedance/torque loss
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
                    losser = loss_torque.item()
                    train_torque_loss += loss_torque.item()
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at batch {batch_idx}, skipping backward")
                continue
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_kin_loss += loss_kin.item()
            train_gait_loss += loss_gait.item()
            n_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'gait': f'{loss_gait.item():.4f}',
                'kin': f'{loss_kin.item():.4f}',
                'impedance' : f'{losser:.4f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0
        val_kin_loss = 0
        val_gait_loss = 0
        val_torque_loss = 0
        n_val_batches = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]')

        with torch.no_grad():
            for batch in val_pbar:
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
        
        scheduler.step()
        
        # Print statistics
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_val_loss = val_loss / max(n_val_batches, 1)
        
        logger.info(f'\nEpoch {epoch+1}/{n_epochs}')
        train_log = (f'Train Loss: {avg_train_loss:.4f} | '
                     f'Kin: {train_kin_loss/max(n_batches,1):.4f} | '
                     f'Gait: {train_gait_loss/max(n_batches,1):.4f}')
        if use_impedance:
            train_log += f' | Torque: {train_torque_loss/max(n_batches,1):.4f}'
        logger.info(train_log)
        
        val_log = (f'Val Loss: {avg_val_loss:.4f} | '
                   f'Kin: {val_kin_loss/max(n_val_batches,1):.4f} | '
                   f'Gait: {val_gait_loss/max(n_val_batches,1):.4f}')
        if use_impedance:
            val_log += f' | Torque: {val_torque_loss/max(n_val_batches,1):.4f}'
        logger.info(val_log)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, 'best_transformer_model.pth')
            logger.info('✓ Saved best model')
    
    return model

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
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='D:/EMG/postprocessed_datasets',
                       help='Directory containing pickle files')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_impedance', action='store_true',
                       help='Use impedance control with torque prediction',default=True)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=1)
    args = parser.parse_args()
    
    print("Loading and parsing datasets...")
    
    parser_obj = WindowedGaitDataParser(
        window_size=200,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        input_dir=args.pkl_dir,
        device=args.device
    )
        
    #parser_obj.parse_grimmer("D:/EMG/postprocessed_datasets/grimmer.pkl")
    #parser_obj.parse_moghadam("D:/EMG/postprocessed_datasets/moghadam.pkl")
    parser_obj.parse_moreira("D:/EMG/postprocessed_datasets/moreira.pkl")
    #parser_obj.parse_embry()
    currMasks = parser_obj.dataset_masks['moreira']

    #NOTE each worker should have its own local list of pointers that point to a single reference train, val, and test dataset within parser_obj 

    train_loader = DataLoader(
        parser_obj.trainDataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        parser_obj.valDataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )
    
    test_loader = DataLoader(
        parser_obj.testDataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(parser_obj.trainDataset)}")
    print(f"  Val: {len(parser_obj.valDataset)}")
    print(f"  Test: {len(parser_obj.testDataset)}")
    parser_obj.trainDataset.verify_lengths()
    parser_obj.testDataset.verify_lengths()
    parser_obj.valDataset.verify_lengths()
    print('Data masks:', currMasks)

    to_stack={'train':parser_obj.trainDataset.data,
     'val':parser_obj.valDataset.data,
     'test':parser_obj.testDataset.data
     }

    for curr_key in to_stack.keys():
        for data_type in to_stack[curr_key][curr_key].keys():
            if data_type!='metadata':
                to_stack[curr_key][curr_key][data_type]=torch.stack(to_stack[curr_key][curr_key][data_type])
        

    torch.cuda.empty_cache()
    gc.collect()
    
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
        kinetic_mask=currMasks['kinetic'],
        device=args.device
    ).to(args.device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nStarting training...")
    train_transformer(
        model, 
        train_loader, 
        val_loader,
        test_loader,
        n_epochs=args.epochs,
        device=args.device,
        lr=args.lr,
        use_impedance=args.use_impedance
    )

    del train_loader, test_loader, val_loader
    del parser_obj.trainDataset, parser_obj.valDataset, parser_obj.testDataset

    parser_obj.parse_lencioni("D:/EMG/postprocessed_datasets/lencioni.pkl")
    currMasks = parser_obj.dataset_masks['lencioni']

    train_loader = DataLoader(
    parser_obj.trainDataset, 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    #persistent_workers=True,
    drop_last=True
    )
    
    val_loader = DataLoader(
        parser_obj.valDataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        #persistent_workers=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        parser_obj.testDataset, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=4,
        prefetch_factor=2,
        #persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(parser_obj.trainDataset)}")
    print(f"  Val: {len(parser_obj.valDataset)}")
    print(f"  Test: {len(parser_obj.testDataset)}")


    to_stack={'train':parser_obj.trainDataset.data,
     'val':parser_obj.valDataset.data,
     'test':parser_obj.testDataset.data
     }

    for curr_key in to_stack.keys():
        for data_type in to_stack[curr_key][curr_key].keys():
            if data_type!='metadata':
                to_stack[curr_key][curr_key][data_type]=torch.stack(to_stack[curr_key][curr_key][data_type])
        

    parser_obj.trainDataset.verify_lengths()
    parser_obj.testDataset.verify_lengths()
    parser_obj.valDataset.verify_lengths()
    print('Data masks:', currMasks)

    print("\nStarting training...")
    train_transformer(
        model, 
        train_loader, 
        val_loader,
        test_loader,
        n_epochs=args.epochs,
        device=args.device,
        lr=args.lr,
        use_impedance=args.use_impedance
    )

    del train_loader, test_loader, val_loader
    del parser_obj.trainDataset, parser_obj.valDataset, parser_obj.testDataset

    print("\nTraining complete!")


if __name__ == '__main__':
    main()