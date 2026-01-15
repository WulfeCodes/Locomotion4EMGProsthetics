import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import argparse
from patlib import Path
import json
import torch.nn as nn
from diffuser import DDPMScheduler, DDIMSCheduler
from diffusers.models import UNet1DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from scipy.signal import welch

class EMGDiffusionModel(nn.Module):
    """
    Conditional diffusion model for EMG generation.
    Conditions on activity labels and/or kinematics.
    """
    #NOTE prev CNN encoder?
    
    def __init__(self, 
                 sample_size=1000,      # EMG sequence length
                 in_channels=13,         # Number of EMG channels
                 out_channels=9,
                 num_train_timesteps=1000,
                 block_out_channels=(32, 64, 128, 256),
                 num_class_embeds=10):   # Number of activity classes
        super().__init__()
        
        # UNet for 1D time series
        self.unet = UNet1DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock1D",
                "DownBlock1D",
                "DownBlock1D",
                "AttnDownBlock1D",
            ),
            up_block_types=(
                "AttnUpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
                "UpBlock1D",
            ),
            num_class_embeds=num_class_embeds,  # For conditioning
        )
        
        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",  # or "sample" or "v_prediction"
        )
        
        self.num_train_timesteps = num_train_timesteps
    
    def forward(self, x, class_labels=None):
        """
        Forward pass during training.
        
        Args:
            x: (batch, channels, time) - clean EMG data
            class_labels: (batch,) - activity labels for conditioning
        """
        # Sample random timesteps
        batch_size = x.shape[0]
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,),
            device=x.device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # Add noise to clean data
        noisy_x = self.scheduler.add_noise(x, noise, timesteps)
        
        # Predict the noise
        noise_pred = self.unet(
            noisy_x, 
            timesteps, 
            class_labels=class_labels
        ).sample
        
        return noise_pred, noise
    
    @torch.no_grad()
    def generate(self, batch_size=1, class_labels=None, device='cuda'):
        """
        Generate new EMG samples.
        
        Args:
            batch_size: Number of samples to generate
            class_labels: Activity labels for conditioning
            device: Device to generate on
        """
        # Start from pure noise
        shape = (batch_size, self.unet.config.out_channels, 
                self.unet.config.sample_size)
        x = torch.randn(shape, device=device)
        
        # Iterative denoising
        self.scheduler.set_timesteps(50)  # Number of inference steps
        
        for t in self.scheduler.timesteps:
            # Predict noise
            timestep = torch.tensor([t] * batch_size, device=device)
            noise_pred = self.unet(
                x, 
                timestep, 
                class_labels=class_labels
            ).sample
            
            # Denoise step
            x = self.scheduler.step(noise_pred, t, x).prev_sample
        
        return x

class EMGClassifier(nn.Module):
    """CNN-LSTM for EMG-based activity classification."""
    
    def __init__(self, n_channels=13, n_classes=10, hidden_size=128):
        super().__init__()
        
        # CNN for spatial features across channels
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(128, hidden_size, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, time, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        
        # CNN
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, time, features)
        x, _ = self.lstm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classify
        x = self.fc(x)
        return x

class GaitEMGDataset(Dataset):
    """Dataset for gait EMG and kinematics with train/val/test splits."""
    
    def __init__(self, h5_path, mode='emg', transform=None, 
                 subjects=None, activities=None, augment=False):
        """
        Args:
            h5_path: Path to HDF5 file
            mode: 'emg', 'kinematics', 'kinetics', or 'all'
            transform: Optional transform
            subjects: List of subject IDs to include
            activities: List of activities to include
            augment: Whether to apply data augmentation (for training)
        """
        self.h5_path = h5_path
        self.mode = mode
        self.transform = transform
        self.augment = augment
        
        # Load metadata (small, keep in memory)
        with h5py.File(h5_path, 'r') as f:
            self.labels = f['activity_labels'][:]
            self.subjects = f['subject_ids'][:]
            self.datasets = f['dataset_names'][:]
            
            # Get data shapes
            self.emg_shape = f['emg'].shape
            self.kin_shape = f['kinematics'].shape
            
            # Load activity map
            self.activity_map = json.loads(f.attrs['activity_map'])
        
        # Filter by subjects/activities
        self.indices = self._filter_indices(subjects, activities)
        
        print(f"Dataset created with {len(self.indices)} samples")
        self._print_dataset_stats()
    
    def _filter_indices(self, subjects, activities):
        """Filter which samples to include."""
        mask = np.ones(len(self.labels), dtype=bool)
        
        if subjects is not None:
            mask &= np.isin(self.subjects, subjects)
        
        if activities is not None:
            mask &= np.isin(self.labels, activities)
        
        return np.where(mask)[0]
    
    def _print_dataset_stats(self):
        """Print dataset statistics."""
        filtered_labels = self.labels[self.indices]
        filtered_subjects = self.subjects[self.indices]
        filtered_datasets = self.datasets[self.indices]
        
        unique_labels, label_counts = np.unique(filtered_labels, return_counts=True)
        unique_subjects = np.unique(filtered_subjects)
        unique_datasets, dataset_counts = np.unique(filtered_datasets, return_counts=True)
        
        print(f"  Unique subjects: {len(unique_subjects)}")
        print(f"  Activity distribution:")
        for label, count in zip(unique_labels, label_counts):
            activity_name = [k for k, v in self.activity_map.items() if v == label][0]
            print(f"    {activity_name}: {count}")
        print(f"  Dataset distribution:")
        for dataset, count in zip(unique_datasets, dataset_counts):
            print(f"    {dataset}: {count}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Load data on-the-fly to avoid memory issues."""
        actual_idx = self.indices[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            if self.mode == 'emg':
                data = f['emg'][actual_idx]
            elif self.mode == 'kinematics':
                data = f['kinematics'][actual_idx]
            elif self.mode == 'kinetics':
                data = f['kinetics'][actual_idx]
            elif self.mode == 'all':
                emg = f['emg'][actual_idx]
                kin = f['kinematics'][actual_idx]
                knt = f['kinetics'][actual_idx]
                data = {'emg': emg, 'kinematics': kin, 'kinetics': knt}
            
            label = f['activity_labels'][actual_idx]
        
        # Apply augmentation if training
        if self.augment:
            data = self._augment(data)
        
        # Convert to torch tensors
        if isinstance(data, dict):
            data = {k: torch.FloatTensor(v) for k, v in data.items()}
        else:
            data = torch.FloatTensor(data)
        
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def _augment(self, data):
        """Apply data augmentation for training."""
        if isinstance(data, dict):
            # Augment each modality separately
            return {k: self._augment_array(v) for k, v in data.items()}
        else:
            return self._augment_array(data)
    
    def _augment_array(self, arr):
        """Augment a single array."""
        # Random time shift
        if np.random.rand() < 0.3:
            shift = np.random.randint(-10, 10)
            arr = np.roll(arr, shift, axis=0)
        
        # Random amplitude scaling
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            arr = arr * scale
        
        # Random noise
        if np.random.rand() < 0.2:
            noise = np.random.normal(0, 0.01, arr.shape)
            arr = arr + noise
        
        return arr

def create_subject_splits(h5_path, train_ratio=0.7, val_ratio=0.15, 
                         test_ratio=0.15, random_seed=42, 
                         stratify_by_dataset=True):
    """
    Split data by subjects to avoid data leakage.
    
    Args:
        h5_path: Path to HDF5 file
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        random_seed: Random seed for reproducibility
        stratify_by_dataset: If True, ensure each dataset is represented 
                            in train/val/test
    
    Returns:
        train_subjects, val_subjects, test_subjects
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    with h5py.File(h5_path, 'r') as f:
        subjects = f['subject_ids'][:]
        datasets = f['dataset_names'][:]
    
    np.random.seed(random_seed)
    
    if stratify_by_dataset:
        # Split by dataset to ensure representation
        train_subjects = []
        val_subjects = []
        test_subjects = []
        
        unique_datasets = np.unique(datasets)
        
        for dataset_name in unique_datasets:
            # Get subjects from this dataset
            dataset_mask = datasets == dataset_name
            dataset_subjects = np.unique(subjects[dataset_mask])
            
            # Shuffle
            np.random.shuffle(dataset_subjects)
            
            # Calculate split points
            n_subjects = len(dataset_subjects)
            n_train = int(n_subjects * train_ratio)
            n_val = int(n_subjects * val_ratio)
            
            # Split
            train_subjects.extend(dataset_subjects[:n_train])
            val_subjects.extend(dataset_subjects[n_train:n_train+n_val])
            test_subjects.extend(dataset_subjects[n_train+n_val:])
        
        train_subjects = np.array(train_subjects)
        val_subjects = np.array(val_subjects)
        test_subjects = np.array(test_subjects)
        
    else:
        # Simple random split across all subjects
        all_subjects = np.unique(subjects)
        np.random.shuffle(all_subjects)
        
        n_subjects = len(all_subjects)
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        train_subjects = all_subjects[:n_train]
        val_subjects = all_subjects[n_train:n_train+n_val]
        test_subjects = all_subjects[n_train+n_val:]
    
    # Print statistics
    print(f"Split Statistics:")
    print(f"  Train subjects: {len(train_subjects)}")
    print(f"  Val subjects: {len(val_subjects)}")
    print(f"  Test subjects: {len(test_subjects)}")
    print(f"  Total subjects: {len(train_subjects) + len(val_subjects) + len(test_subjects)}")
    
    return train_subjects, val_subjects, test_subjects

def train_model(model, train_loader, val_loader, n_epochs=50, device='cuda'):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           patience=5)
    
    best_val_acc = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            #DEFINE THE LOSS HERE
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.3f} | '
              f'Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f} | '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('✓ Saved best model')
        print()

def train_diffusion_model(model, train_loader, val_loader, 
                         n_epochs=100, device='cuda', save_dir='diffusion_checkpoints'):
    """Train the diffusion model."""
    from pathlib import Path
    Path(save_dir).mkdir(exist_ok=True)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Cosine learning rate schedule
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(train_loader) * n_epochs,
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            # data: (batch, time, channels) -> need (batch, channels, time)
            data = data.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            noise_pred, noise = model(data, class_labels=labels)
            
            # Compute loss
            loss = nn.functional.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}] '
                      f'Batch [{batch_idx+1}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.permute(0, 2, 1).to(device)
                labels = labels.to(device)
                
                noise_pred, noise = model(data, class_labels=labels)
                loss = nn.functional.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, f'{save_dir}/best_diffusion_model.pth')
            print('✓ Saved best model\n')
        
        # Generate samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("Generating samples...")
            samples = model.generate(
                batch_size=4, 
                class_labels=torch.tensor([0, 1, 2, 3], device=device),
                device=device
            )
            # Save or visualize samples
            np.save(f'{save_dir}/samples_epoch_{epoch+1}.npy', 
                   samples.cpu().numpy())
    
    return model

def evaluate_diffusion_model(model, test_loader, device='cuda', n_samples=100):
    """Evaluate diffusion model quality."""
    from scipy import stats
    from sklearn.metrics import mean_squared_error
    
    model.eval()
    
    # Generate samples
    generated_samples = []
    real_samples = []
    
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            if i >= n_samples // data.shape[0]:
                break
            
            data = data.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            
            # Generate
            gen = model.generate(
                batch_size=data.shape[0],
                class_labels=labels,
                device=device
            )
            
            generated_samples.append(gen.cpu().numpy())
            real_samples.append(data.cpu().numpy())
    
    generated_samples = np.concatenate(generated_samples, axis=0)
    real_samples = np.concatenate(real_samples, axis=0)
    
    # Compute metrics
    # 1. Distribution similarity (per channel)
    ks_stats = []
    for ch in range(generated_samples.shape[1]):
        gen_flat = generated_samples[:, ch, :].flatten()
        real_flat = real_samples[:, ch, :].flatten()
        ks_stat, _ = stats.ks_2samp(gen_flat, real_flat)
        ks_stats.append(ks_stat)
    
    print(f"KS Statistic (lower is better): {np.mean(ks_stats):.4f}")
    
    # 2. Temporal coherence (autocorrelation)
    def compute_autocorr(data):
        autocorrs = []
        for i in range(data.shape[0]):
            for ch in range(data.shape[1]):
                signal = data[i, ch, :]
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorrs.append(autocorr[:50])  # First 50 lags
        return np.mean(autocorrs, axis=0)
    
    gen_autocorr = compute_autocorr(generated_samples)
    real_autocorr = compute_autocorr(real_samples)
    autocorr_mse = mean_squared_error(gen_autocorr, real_autocorr)
    
    print(f"Autocorrelation MSE: {autocorr_mse:.4f}")
    
    # 3. Spectral similarity    
    def compute_psd(data):
        psds = []
        for i in range(data.shape[0]):
            for ch in range(data.shape[1]):
                f, psd = welch(data[i, ch, :], fs=1000, nperseg=256)
                psds.append(psd)
        return np.mean(psds, axis=0), f
    
    gen_psd, f = compute_psd(generated_samples)
    real_psd, _ = compute_psd(real_samples)
    psd_mse = mean_squared_error(gen_psd, real_psd)
    
    print(f"PSD MSE: {psd_mse:.4f}")
    
    return {
        'ks_statistic': np.mean(ks_stats),
        'autocorr_mse': autocorr_mse,
        'psd_mse': psd_mse
    }

# Usage

def main():
    model = EMGClassifier(n_channels=13, n_classes=10)
    train_model(model, train_loader, val_loader, n_epochs=50)

    # Usage
    train_subjects, val_subjects = create_subject_splits('gait_data.h5')

    train_dataset = GaitEMGDataset('gait_data.h5', mode='emg', 
                                subjects=train_subjects)
    val_dataset = GaitEMGDataset('gait_data.h5', mode='emg', 
                                subjects=val_subjects)

    train_loader = DataLoader(train_dataset, batch_size=32, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, 
                            shuffle=False, num_workers=4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='gait_data.h5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create data splits
    train_subjects, val_subjects = create_subject_splits(args.h5_path)
    
    # Create datasets
    train_dataset = GaitEMGDataset(args.h5_path, subjects=train_subjects)
    val_dataset = GaitEMGDataset(args.h5_path, subjects=val_subjects)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4)
    
    # Create model
    n_classes = len(np.unique(train_dataset.labels))
    n_channels = train_dataset.emg_shape[-1]
    model = EMGClassifier(n_channels=n_channels, n_classes=n_classes)
    
    # Train
    train_model(model, train_loader, val_loader, 
               n_epochs=args.epochs, device=args.device)

    model = EMGDiffusionModel(
            emg_channels=13,
            kinematic_channels=9,
            sample_size=1000,
            num_train_timesteps=1000,
            num_class_embeds=10
        )
    
    # Train
    train_diffusion_model(
        model, 
        train_loader, 
        val_loader,
        n_epochs=100,
        device='cuda'
    )
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        # Generate EMG for specific activity and kinematics
        sample_kinematics = torch.randn(4, 9, 200).cuda()  # 4 samples
        sample_labels = torch.tensor([0, 1, 2, 3]).cuda()
        
        generated_emg = model.generate(
            kinematics=sample_kinematics,
            class_labels=sample_labels,
            device='cuda'
        )
        
        print(f"Generated EMG shape: {generated_emg.shape}")
        # (4, 13, 1000) - 4 samples, 13 channels, 1000 timepoints

if __name__ == '__main__':
    main()

#TODO class configuration, loss configuration