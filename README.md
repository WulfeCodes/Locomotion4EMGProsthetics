# EMG–Kinematics–Kinetics Foundation Dataset

## Project motivation

| Problem | Description |
|---|---|
| Current prosthetics | Industrial powered prosthetics still rely on finite-state machines, have unrealistic cost requirements, and are largely inaccessible. |
| Access gap | Approximately 70% of amputees never receive a prosthetic. |
| Core bottleneck | The lack of large-scale, unified biomechanical datasets prevents learning-based prosthetic control. |

## Project goal

| Objective | Description |
|---|---|
| Data collection | Aggregate open-source EMG, kinematic, and kinetic datasets. |
| Normalization | Convert heterogeneous formats into a homogeneous, indexable representation. |
| ML readiness | Enable foundation model training for locomotion and prosthetic control. |

---

## Storage constraints

| File type | Size |
|---|---|
| Raw datasets | ~700 GB |
| Processed `.pkl` | ~70 GB |

Files are available at: https://drive.google.com/drive/folders/1Kba2_5XaiBluw-rXHpUfm8X4auCholB3?usp=sharing

Reference papers are available at: https://drive.google.com/drive/folders/1vcUSvnTsDR734KYHGv7suk0HFPTf5C_a

---

## Dataset distribution

| Dataset | Stride count | % of strides | Patients | % of patients |
|---|---|---|---|---|
| Bacek | 258,418 | 50.09% | 21 | 4.24% |
| Macaluso | 66,035 | 12.80% | 10 | 2.02% |
| Camargo | 53,713 | 10.41% | 22 | 4.44% |
| K2Muse | 40,612 | 7.87% | 30 | 6.06% |
| Angelidou | 40,204 | 7.79% | 9 | 1.82% |
| Embry | 26,846 | 5.20% | 10 | 2.02% |
| Grimmer | 10,772 | 2.09% | 12 | 2.42% |
| Hu | 6,365 | 1.23% | 10 | 2.02% |
| Gait120 | 6,310 | 1.22% | 110 | 22.22% |
| Moreira | 2,613 | 0.51% | 16 | 3.23% |
| Criekinge | 2,102 | 0.41% | 138 | 27.88% |
| Lencioni | 1,159 | 0.22% | 50 | 10.10% |
| SIAT | 441 | 0.09% | 40 | 8.08% |
| Moghadam | 290 | 0.06% | 17 | 3.43% |
| **Total** | | | **495** | **100%** |

> **Note:** Camargo paper not available; all other papers are in the Drive linked above.

---

## Data modalities

### Kinematics and kinetics tensor layout

| Dimension | Meaning |
|---|---|
| dim(0) | Joint: hip, knee, ankle |
| dim(1) | Motion: adduction, rotation, flexion |

### EMG channel ordering

| Index | Muscle |
|---|---|
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

### Data availability masks

Each modality includes a binary mask indicating data presence.

| Value | Meaning |
|---|---|
| 1 | Data exists |
| 0 | Data missing |

Masks are provided for EMG, kinematics, and kinetics.

### Data hierarchy

After parsing, data is structured as:

```
Patient → Activity → Trial → [Optional subtrial] → Segmented gait cycle
```

Saved as a `.pkl` dictionary.

---

## Processing pipeline

### `testEMG1.py`

| Stage | Description |
|---|---|
| EMG filtering | Wavelet, notch, and bandpass filtering |
| Normalization | Applied across datasets |
| Temporal alignment | EMG structured into 13 channels |
| Kinematics and kinetics | Structured into 3×3 tensors |
| Output | Stride-level arrays |

### `syncEMG.py`

| Function | Purpose |
|---|---|
| `SyncAll()` | Unified interface for reading all `.pkl` datasets |
| `analyze_sample_counts()` | Identifies sample rate inconsistencies and stride length variance |

### `syncSignals.py`

| Function | Purpose |
|---|---|
| `resample_all_datasets()` | Resamples EMG to 1 kHz and kinematics/kinetics to 100 Hz (parameterizable) |

---

## `convert2DL.py`

Transforms stride-level biomechanical data into windowed, learning-ready samples aligned for predictive control and foundation model training.

**Core purpose:** Bridge raw stride data and deep learning by constructing temporally aligned EMG–state–target tuples with mutually exclusive patient-level dataset splits.

### Key classes

#### `SplitDataset`
PyTorch-compatible dataset container for a single split (`train`, `val`, or `test`). Stores EMG windows, kinematic state vectors, gait percentage scalars, torque targets, and metadata.

| Method | Description |
|---|---|
| `__getitem__()` | Returns tensors ready for GPU training |
| `verify_lengths()` | Sanity-checks that all stored arrays are aligned |

#### `WindowedGaitDataParser`
Primary dataset parser and window generator.

| Responsibility | Description |
|---|---|
| Parsing | Handles heterogeneous `.pkl` datasets with dataset-specific logic |
| Splits | Enforces patient-level train/val/test splits to prevent subject leakage |
| Alignment | Aligns EMG history windows with kinematic sampling |
| State construction | Builds impedance-relevant state representations |

### Core methods

| Method | Description |
|---|---|
| `assign_patient_to_split()` | Deterministically assigns each patient to a split via hashing for reproducible datasets |
| `compute_omega()` | Computes joint angular velocity via causal finite differences from gait-normalized kinematics |
| `compute_alpha()` | Computes joint angular acceleration via second-order finite differences (central when possible) |
| `extract_windows_aligned_to_kinematics()` | Core windowing routine — maps each kinematic timestep to its EMG index, extracts a fixed-length EMG history window (zero-padded if needed), and builds input/target state pairs |
| `add_stride()` | Converts a single stride into multiple supervised windows and appends to the correct split |
| `extract_masks()` | Normalizes dataset-specific modality masks into a unified format for downstream loss computation |
| `parse_*()` | Dataset-specific parsers traversing each dataset's hierarchy (patient → trial → stride) |
| `convert_all()` | Iterates all known datasets and performs full conversion into windowed samples |

### Output structure (per window)

| Feature | Shape | Description |
|---|---|---|
| `emg` | `(13, window_size)` | EMG history window |
| `input_kin_state` | `(27,)` | Current joint state `[θ, ω, α]` |
| `target_kin_state` | `(27,)` | Next-step joint state; every 9 indices = one differentiation order, ordered hip (roll, yaw, pitch) → knee → ankle |
| `input_gait_pct` | scalar | Current gait percentage |
| `target_gait_pct` | scalar | Next-step gait percentage |
| `target_torque` | `(9,)` | Joint torques (zeros if unavailable) |
| `metadata` | various | Activity type, dataset name, patient ID, torque availability flag |

---

## `trainFM.py`

Reference training pipeline for EMG-driven kinematic and impedance prediction using a transformer-based architecture.

> **Note:** This script is a baseline and research scaffold, not a finalized production training pipeline.

### Command line arguments

#### Paths & logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset_path` | `str` | `D:/EMG/ML_datasets` | Directory containing pickle files |
| `--log_dir` | `str` | `./logs` | TensorBoard / log output directory |
| `--save_plot_dir` | `str` | `/gpfs/.../plots/` | Directory to save plots |
| `--save_model_path` | `str` | `C:/EMG/software/models/IM` | Directory to save model checkpoints |
| `--load_path` | `str` | `None` | Path to load a model checkpoint |

#### Training

| Argument | Type | Default | Description |
|---|---|---|---|
| `--batch_size` | `int` | `128` | Training batch size |
| `--epochs` | `int` | `2` | Number of training epochs |
| `--lr` | `float` | `1e-4` | Learning rate |
| `--device` | `str` | `cuda` | Device to train on (`cuda` or `cpu`) |
| `--num_workers` | `int` | `2` | DataLoader worker count |

#### Model architecture

| Argument | Type | Default | Description |
|---|---|---|---|
| `--d_model` | `int` | `512` | Transformer model dimension |
| `--nhead` | `int` | `4` | Number of attention heads |
| `--num_layers` | `int` | `4` | Number of transformer layers |
| `--use_impedance` | flag | `True` | Enable impedance control with torque prediction |

#### Masking

| Argument | Type | Default | Description |
|---|---|---|---|
| `--artificial_emg_mask_prob` | `float` | `0.15` | Probability of zeroing out an EMG channel during training |
| `--artificial_kin_mask_prob` | `float` | `0.15` | Probability of zeroing out a kinematic dimension during training |

#### Noise augmentation — split flags

| Argument | Default | Description |
|---|---|---|
| `--train_noise` | `False` | Enable signal + temporal jitter noise on train split |
| `--val_noise` | `False` | Enable signal + temporal jitter noise on val split |
| `--test_noise` | `False` | Enable signal + temporal jitter noise on test split |

#### Gaussian signal noise

| Argument | Type | Default | Description |
|---|---|---|---|
| `--emg_noise_std_max` | `float` | `1.0` | Per-channel EMG noise std upper bound: `std_c ~ U[0, std_max]` |
| `--emg_noise_mean_max` | `float` | `0.0` | Per-channel EMG noise mean upper bound: `mean_c ~ U[-mean_max, mean_max]` |
| `--kin_noise_std_max` | `float` | `1.0` | Per-dim kin noise std upper bound |
| `--kin_noise_mean_max` | `float` | `0.0` | Per-dim kin noise mean upper bound |
| `--gait_noise_std_max` | `float` | `0.05` | Gait pct noise std upper bound |
| `--gait_noise_mean_max` | `float` | `0.0` | Gait pct noise mean upper bound |

#### Temporal jitter

| Argument | Type | Default | Description |
|---|---|---|---|
| `--emg_jitter_max` | `int` | `5` | EMG temporal jitter upper bound in window steps (`delta ~ U[0, max]`) |
| `--kin_jitter_max` | `int` | `5` | Kin temporal jitter upper bound (sampled independently from EMG) |
| `--jitter_warmup_steps` | `int` | `0` | Linearly ramp both jitter maxes from 0 → max over N train steps (`0` = off) |
| `--jitter_retries` | `int` | `5` | Max attempts to find a valid jittered index before falling back to original |

---

### Overview

All training logic lives within `meta_train_temperature_loop()`. The function runs for `args.outer_epochs` outer epochs; each epoch trains for `args.steps_per_epoch` steps via `train_val_test_transformer()`. At the end of each outer epoch, the full cached validation set is evaluated, losses are plotted, and results are saved to the configured directory.

Three model checkpoints are saved throughout training:

| Checkpoint | Criterion |
|---|---|
| `best_kin` | Lowest kinematic loss ceiling |
| `best_torque` | Lowest torque loss ceiling |
| `best_avg` | Lowest average of the two |

Loss histories (kinematic and torque, per dataset) are collected and saved as dictionaries.

---

### Data pipeline

#### `SplitDataset`

The `DataLoader` uses `SplitDataset`, which parses any `.pt` files found within `--dataset_path` and exposes them as a unified dataset.

#### `TemperatureScaledStreamer` \[INSERT REFERENCE\]

Controls sampling across heterogeneous sub-datasets via temperature-scaled draw probabilities. For sub-dataset $i$ with $N_i$ samples:

$$P_i = \frac{N_i^\alpha}{\sum_j N_j^\alpha}$$

where $\alpha$ is a temperature parameter that smooths the distribution across non-uniformly sized sub-datasets ($\alpha = 1$ gives proportional sampling; $\alpha \to 0$ gives uniform sampling).

The streamer inherits from `IterableDataset`. Samples are buffered from available sub-dataset chunks and popped on `__iter__`. Once all chunks in a training cycle are exhausted, the buffer resets.

---

### Batch format

| Key | Description |
|---|---|
| `emg` | EMG temporal window (sparse channels) |
| `input_kin_state` | Current kinematic state |
| `input_gait_pct` | Current gait phase percentage |
| `target_kin_state` | Target kinematic state |
| `target_gait_pct` | Target gait phase percentage |
| `target_torque` | Target joint torques |
| `has_torque` | Flag — whether to compute torque loss for this sample |

When `has_torque` is set, torque loss is computed and its corresponding sparse kinematic/torque vectors with masks are included in the loss calculation.

---

### Model architecture — `EMGTransformer`

```
EMG window  ──► 1D CNN (project up → down) ──┐
                                              ├──► Transformer (cross-attention) ──► latent z
Kin state   ──► MLP embedding ───────────────┘
                    ▲
              Positional encoder
```

| Input tensor | Shape | Output head | Shape |
|---|---|---|---|
| EMG window | `(batch, 13, 100)` | Next-step kinematics | `(27,)` |
| Kinematic state | `(batch, 27)` | Next gait percentage | `(1,)` |
| Gait percentage | `(batch, 1)` | Impedance (K, C, M) | per joint (optional) |

The output latent vector $z$ is passed through two parallel heads, each producing a **27-dimensional** vector:

| Head | Output | Description |
|---|---|---|
| Impedance head | $[\mathbf{K}, \mathbf{C}, \mathbf{M}]$ | Log-std + mean (SAC-style); impedance gains |
| Kinematic head | $[\boldsymbol{\theta}, \boldsymbol{\omega}, \boldsymbol{\alpha}]_\text{des}$ | Desired angles, velocities, accelerations |

Together they form the **54-dimensional action vector**.

#### Output vector layout (27D, each head)

Both vectors share the same layout — 3 joints × 3 axes × 3 orders:

| Dims | Joint | Axes |
|---|---|---|
| 0–8 | Hip | Roll (longitudinal), Yaw (vertical), Pitch (lateral) |
| 9–17 | Knee | Roll, Yaw, Pitch |
| 18–26 | Ankle | Roll, Yaw, Pitch |

Within each group of 3: order 0 = $K$ / $\theta$, order 1 = $C$ / $\omega$, order 2 = $M$ / $\alpha$.

---

### Impedance torque computation — `compute_impedance_torque`

For each of the 9 output dimensions per joint:

$$\tau = K(\theta_\text{des} - \theta_\text{curr}) + C(\dot{\theta}_\text{des} - \dot{\theta}_\text{curr}) + M(\ddot{\theta}_\text{des} - \ddot{\theta}_\text{curr})$$

where $K$, $C$, $M$ are the predicted stiffness, damping, and inertia gains respectively.

The resulting torque vector $\boldsymbol{\tau}$ and kinematic vector are each reduced to a scalar loss, summed, and backpropagated with standard gradient descent at learning rate `--lr`.

---

### Key components

#### `train_val_test_transformer()`
Core epoch-level training and evaluation routine. Handles the forward pass, masked multi-task loss computation (kinematics, gait, and optional impedance-derived torque), and optimization steps. During validation (`split_type='val'`), tracks the lowest loss and saves the best-performing model state.

#### `meta_train_transformer_loop()`
Overarching foundation-model data loader and curriculum manager.

- **Chunked loading:** Iterates through physical `.pt` chunks sequentially to manage memory.
- **Dynamic masking:** Automatically updates the `EMGTransformer`'s modality masks (EMG, kinematics, kinetics) on the fly based on the currently loaded chunk's metadata.
- **Validation:** Runs one full pass over every validation chunk, computing `average_total_loss`, `average_kinematic_loss`, and `average_torque_loss` normalized by batch size and number of loss additions — ensuring losses are comparable across datasets with differing modalities. Losses accumulate into a `current_loss_dict` across all sub-dataset chunks; once all eval chunks are parsed, a ceiling check compares each modality's loss against the previous best and saves the corresponding checkpoint if improved.

#### `check_load_time()`
Performance profiling utility. Measures the I/O bottleneck by timing `torch.load()` operations versus `DataLoader` instantiation overhead across all dataset chunks.

---

## `visualizer.py`

Plots loss curves and animations on test data.

---

## Current progress

**Active:**
- RL fine-tuning on SCONEgym

**To do:**
- Test Default actuation for DOFs of RL.

**Known data issues:**
- Criekinge: NaN errors on some torque and EMG channels in original data
- Camargo, Angelidou, Hu: NaN errors from original data
- Macaluso, Angelidou, Camargo: large kinetic range
- Grimmer: short gaits due to heel-strike-to-heel-strike segmentation (will likely keep continuous for Mk2)
- Kinetic normalization: Embry (claimed Nmm/kg), Macaluso (no units given) — follow-up emails sent



