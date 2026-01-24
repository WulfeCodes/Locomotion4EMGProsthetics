# EMG–Kinematics–Kinetics Foundation Dataset

## Project Motivation

| Problem | Description |
|--------|-------------|
| Current prosthetics | Industrial powered prosthetics still rely on finite-state machines, have unrealistic cost requirements, and are largely inaccessible. |
| Access gap | Approximately 70% of amputees never receive a prosthetic. |
| Core bottleneck | The lack of large-scale, unified biomechanical datasets prevents learning-based prosthetic control. |

---

## Project Goal

| Objective | Description |
|---------|-------------|
| Data collection | Aggregate open-source EMG, kinematic, and kinetic datasets. |
| Normalization | Convert heterogeneous formats into a homogeneous, indexable representation. |
| ML readiness | Enable foundation model training for locomotion and prosthetic control. |

---

## Reference Papers

All corresponding papers are available here:  
https://drive.google.com/drive/folders/1vcUSvnTsDR734KYHGv7suk0HFPTf5C_a

| Dataset | Paper |
|--------|------|
| Camargo | Not available (dataset included without paper) |
| All others | Included in the Google Drive |

---

## Dataset Distribution

| Dataset     | Stride Count | Percentage |
|------------|--------------|------------|
| Bacek       | 258,418 | 50.09% |
| Macaluso   | 66,035  | 12.80% |
| Camargo    | 53,713  | 10.41% |
| K2Muse     | 40,612  | 7.87% |
| Angelidou  | 40,204  | 7.79% |
| Embry      | 26,846  | 5.20% |
| Grimmer    | 10,772  | 2.09% |
| Hu         | 6,365   | 1.23% |
| Gait120    | 6,310   | 1.22% |
| Moreira    | 2,613   | 0.51% |
| Criekinge  | 2,102   | 0.41% |
| Lencioni   | 1,159   | 0.22% |
| SIAT       | 441     | 0.09% |
| Moghadam   | 290     | 0.06% |

---

## Data Modalities

### Kinematics and Kinetics Tensor Layout

| Dimension | Meaning |
|---------|--------|
| dim(0) | Joint: hip, knee, ankle |
| dim(1) | Motion: adduction, rotation, flexion |

---

### EMG Channel Ordering

| Index | Muscle |
|------|--------|
| 1 | Vastus Lateralis |
| 2 | Rectus Femoris |
| 3 | Vastus Medialis |
| 4 | Tibialis Anterior |
| 5 | Biceps Femoris |
| 6 | Semitendinosus or Semimembranosus |
| 7 | Gastrocnemius Medialis |
| 8 | Gastrocnemius Lateralis |
| 9 | Soleus |
| 10 | Peroneus Longus |
| 11 | Peroneus Brevis |
| 12 | Gluteus Medius |
| 13 | Gluteus Maximus |

---

## Data Availability Masks

Each modality includes a binary mask indicating data presence.

| Value | Meaning |
|-------|--------|
| 1 | Data exists |
| 0 | Data missing |

Masks are provided for EMG, kinematics, and kinetics.

---

## Data Hierarchy

After parsing, data is structured as:

| Level |
|------|
| Patient |
| Activity |
| Trial |
| Optional subtrial |
| Segmented gait cycle |

Saved as a `.pkl` dictionary.

---

## Processing Pipeline

### testEMG1.py

| Stage | Function |
|------|--------|
| EMG filtering | Wavelet, notch, and bandpass filtering |
| Normalization | Applied across datasets |
| Temporal alignment | EMG structured into 13 channels |
| Kinematics and kinetics | Structured into 3 × 3 tensors |
| Output | Stride-level arrays |

---

### syncEMG.py

| Function | Purpose |
|--------|---------|
| SyncAll() | Unified interface for reading all `.pkl` datasets |
| analyze_sample_counts() | Identifies sample rate inconsistencies and stride length variance |

---

### syncSignals.py

| Function | Purpose |
|--------|---------|
| resample_all_datasets() | Resamples EMG to 1kHz and kinematics/kinetics to 200 samples per stride, these can be parameterized per the user's preference. |

---

## convert2DL.py

Transforms stride-level biomechanical data into windowed, learning-ready samples aligned for predictive control and foundation model training.

Core purpose: bridge raw stride data and deep learning by constructing temporally aligned EMG–state–target tuples with deterministic dataset splits.

Key Classes and Functions

SplitDataset
Container class implementing a PyTorch-compatible dataset for a single split (train, val, or test).

Stores EMG windows, kinematic state vectors, gait percentage scalars, torque targets, and metadata

Implements __getitem__ to return tensors ready for GPU training

verify_lengths() sanity-checks that all stored arrays are aligned

WindowedGaitDataParser
Primary dataset parser and window generator.

Responsibilities:

Parse heterogeneous .pkl datasets with dataset-specific logic

Enforce patient-level train/val/test splits (prevents subject leakage)

Align EMG history windows with kinematic sampling

Construct impedance-relevant state representations

Important methods:

assign_patient_to_split()
Deterministically assigns each patient to a split using hashing, ensuring reproducible splits across runs.

compute_omega()
Computes joint angular velocity using causal finite differences from gait-normalized kinematics.

compute_alpha()
Computes joint angular acceleration using second-order finite differences (central when possible).

extract_windows_aligned_to_kinematics()
Core windowing routine:

Maps each kinematic timestep to its corresponding EMG index

Extracts a fixed-length EMG history window (zero-padded if necessary)

Builds the input kinematic state: [θ, ω, α] at time t

Builds the target kinematic state: [θ, ω, α] at time t+1

Optionally attaches torque targets when available

add_stride()
Converts a single stride into many supervised learning windows and appends them to the correct dataset split.

extract_masks()
Normalizes dataset-specific modality masks (EMG / kinematics / kinetics) into a unified format used downstream for masking and loss computation.

parse_*() functions
Dataset-specific parsers that traverse each dataset’s internal hierarchy (patient → trial → stride) and invoke add_stride().

convert_all()
Iterates through all known datasets and performs full conversion into windowed samples.

Output structure (per window):

emg: (13, window_size) EMG history

input_kin_state: (27,) current joint state [θ, ω, α]

input_gait_pct: scalar gait percentage

target_kin_state: (27,) next-step joint state

target_gait_pct: scalar gait percentage

target_torque: (9,) joint torques or zeros if unavailable

metadata: activity, dataset, patient ID, torque availability flag

## trainFM.py

Reference training pipeline for EMG-driven kinematic and impedance prediction using a transformer-based architecture.

Core purpose: demonstrate how the processed dataset can be used to train a foundation-style model that jointly predicts kinematics, gait phase, and impedance parameters.

Key Components

EMGTransformer (nn.Module)
Multi-input transformer model combining EMG time-series and biomechanical state information.

Model inputs:

EMG window (batch, 13, 200)

Current kinematic state (batch, 27)

Current gait percentage (batch, 1)

Architecture overview:

EMG encoder: 1D convolutional stack that embeds EMG time-series into a latent representation

State embeddings: separate embeddings for kinematic state and gait phase

Transformer: encoder–decoder transformer fusing EMG context with state queries

Output heads:

Next-step kinematic state (27)

Next gait percentage (1)

Optional impedance parameters (K, C, M) per joint

Masking support:

EMG, kinematic, and kinetic masks are applied to inputs and losses

Enables training across datasets with missing channels or modalities

compute_impedance_torque()
Implements the classical impedance control law:

τ = K(θᵈ − θ) + C(ωᵈ − ω) + M(αᵈ − α)

Used to translate predicted impedance parameters and kinematic tracking error into joint torques for supervision.

train_transformer()
End-to-end training loop.

Responsibilities:

Handles forward pass, loss computation, and optimization

Supports multi-task losses:

Kinematic prediction loss

Gait phase prediction loss

Optional torque loss (only when ground truth torque exists)

Applies gradient clipping and cosine learning rate scheduling

Saves best-performing model checkpoints

main()
Executable training entry point.

Loads and parses datasets via WindowedGaitDataParser

Builds PyTorch DataLoaders for each split

Instantiates the EMGTransformer with dataset-specific masks

Launches training with configurable hyperparameters

This script is intended as a baseline and research scaffold, not a finalized production training pipeline.


## Storage Constraints

| File Type | Size |
|---------|------|
| Raw datasets | ~700 GB |
| Processed `.pkl` | ~70 GB |

These files included in this repository can be found through: https://drive.google.com/drive/folders/1Kba2_5XaiBluw-rXHpUfm8X4auCholB3?usp=sharing.

  



