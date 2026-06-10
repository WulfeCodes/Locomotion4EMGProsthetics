
# EMG-Driven Prosthetic Limb Control — Project README

## Table of Contents

- [Project motivation](#project-motivation)
- [Project goal](#project-goal)
- [Storage constraints](#storage-constraints)
- [Dataset distribution](#dataset-distribution)
- [Data modalities](#data-modalities)
  - [Kinematics and kinetics tensor layout](#kinematics-and-kinetics-tensor-layout)
  - [EMG channel ordering](#emg-channel-ordering)
  - [Data availability masks](#data-availability-masks)
  - [Data hierarchy](#data-hierarchy)
- [Processing pipeline](#processing-pipeline)
  - [testEMG1.py](#testemg1py)
  - [syncEMG.py](#syncemgpy)
  - [syncSignals.py](#syncsignalspy)
- [convert2DL.py](#convert2dlpy)
  - [Key classes](#key-classes)
  - [Core methods](#core-methods)
  - [Output structure per window](#output-structure-per-window)
- [trainFM.py](#trainfmpy)
  - [Command line arguments](#command-line-arguments)
  - [Overview](#overview)
  - [Data pipeline](#data-pipeline)
  - [Batch format](#batch-format)
  - [Model architecture](#model-architecture--emgtransformer)
  - [Impedance torque computation](#impedance-torque-computation--compute_impedance_torque)
  - [Key components](#key-components)
- [visualizer.py](#visualizerpy)
- [rltrain.py](#rltrain.py--rl-fine-tuning-for-prosthetic-control)
  - [Environment setup](#environment-setup)
  - [State and action dimensions](#state--action-dimensions)
  - [SAC training loop](#sac-training-loop--rl_train_sac)
  - [PPO training loop](#ppo-training-loop--rl_train_ppo)
  - [Arguments](#arguments)
- [Current progress](#current-progress)

---

## Project Motivation

| Problem | Description |
|---|---|
| Current prosthetics | Industrial powered prosthetics still rely on finite-state machines, have unrealistic cost requirements, and are largely inaccessible. |
| Access gap | Approximately 70% of amputees never receive a prosthetic. |
| Core bottleneck | The lack of large-scale, unified biomechanical datasets prevents learning-based prosthetic control. |

## Project Goal

| Objective | Description |
|---|---|
| Data collection | Aggregate open-source EMG, kinematic, and kinetic datasets. |
| Normalization | Convert heterogeneous formats into a homogeneous, indexable representation. |
| ML readiness | Enable foundation model training for locomotion and prosthetic control. |

---

## Storage Constraints

| File type | Size |
|---|---|
| Raw datasets | ~700 GB |
| Processed `.pkl` | ~70 GB |

Files: https://drive.google.com/drive/folders/1Kba2_5XaiBluw-rXHpUfm8X4auCholB3?usp=sharing  
Reference papers: https://drive.google.com/drive/folders/1vcUSvnTsDR734KYHGv7suk0HFPTf5C_a

---

## Dataset Distribution

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

> **Note:** Camargo paper unavailable; all others are in the Drive linked above.

---

## Data Modalities

### Kinematics and Kinetics Tensor Layout

| Dimension | Meaning |
|---|---|
| dim(0) | Joint: hip, knee, ankle |
| dim(1) | Motion: adduction, rotation, flexion |

### EMG Channel Ordering

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

### Data Availability Masks

| Value | Meaning |
|---|---|
| 1 | Data exists |
| 0 | Data missing |

Masks are provided for EMG, kinematics, and kinetics.

### Data Hierarchy

After parsing, data is structured as:

```
Patient → Activity → Trial → [Optional subtrial] → Segmented gait cycle
```

Saved as a `.pkl` dictionary.

---

## Processing Pipeline

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

### Key Classes

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

### Core Methods

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

### Output Structure (per window)

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

### Command Line Arguments

#### Paths & Logging

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

#### Model Architecture

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

#### Noise Augmentation

| Argument | Default | Description |
|---|---|---|
| `--train_noise` | `False` | Enable signal + temporal jitter noise on train split |
| `--val_noise` | `False` | Enable signal + temporal jitter noise on val split |
| `--test_noise` | `False` | Enable signal + temporal jitter noise on test split |
| `--emg_noise_std_max` | `1.0` | Per-channel EMG noise std upper bound: `std_c ~ U[0, std_max]` |
| `--emg_noise_mean_max` | `0.0` | Per-channel EMG noise mean upper bound: `mean_c ~ U[-mean_max, mean_max]` |
| `--kin_noise_std_max` | `1.0` | Per-dim kin noise std upper bound |
| `--kin_noise_mean_max` | `0.0` | Per-dim kin noise mean upper bound |
| `--gait_noise_std_max` | `0.05` | Gait pct noise std upper bound |
| `--gait_noise_mean_max` | `0.0` | Gait pct noise mean upper bound |
| `--emg_jitter_max` | `5` | EMG temporal jitter upper bound in window steps |
| `--kin_jitter_max` | `5` | Kin temporal jitter upper bound (sampled independently from EMG) |
| `--jitter_warmup_steps` | `0` | Linearly ramp both jitter maxes from 0 → max over N train steps (`0` = off) |
| `--jitter_retries` | `5` | Max attempts to find a valid jittered index before falling back to original |

---

### Overview

All training logic lives within `meta_train_temperature_loop()`. Each outer epoch trains for `args.steps_per_epoch` steps via `train_val_test_transformer()`. At the end of each outer epoch, the full cached validation set is evaluated, losses are plotted, and results saved.

Three model checkpoints are saved throughout training:

| Checkpoint | Criterion |
|---|---|
| `best_kin` | Lowest kinematic loss ceiling |
| `best_torque` | Lowest torque loss ceiling |
| `best_avg` | Lowest average of the two |

### Data Pipeline

#### `TemperatureScaledStreamer`

Controls sampling across heterogeneous sub-datasets via temperature-scaled draw probabilities. For sub-dataset $i$ with $N_i$ samples:

$$P_i = \frac{N_i^\alpha}{\sum_j N_j^\alpha}$$

$\alpha = 1$ gives proportional sampling; $\alpha \to 0$ gives uniform sampling. The streamer inherits from `IterableDataset` and buffers samples from available chunks, resetting once all chunks in a training cycle are exhausted.

### Batch Format

| Key | Description |
|---|---|
| `emg` | EMG temporal window (sparse channels) |
| `input_kin_state` | Current kinematic state |
| `input_gait_pct` | Current gait phase percentage |
| `target_kin_state` | Target kinematic state |
| `target_gait_pct` | Target gait phase percentage |
| `target_torque` | Target joint torques |
| `has_torque` | Flag — whether to compute torque loss for this sample |

### Model Architecture — `EMGTransformer`

```
                    ┌─────────────────┐        ┌──────────────────┐
                    │   EMG window    │        │  Kinematic state │
                    │   (13 × W)      │        │  (27,) + gait %  │
                    └────────┬────────┘        └────────┬─────────┘
                             │                          │
                    ┌────────▼────────┐        ┌────────▼─────────┐
                    │  1D CNN encoder │        │   MLP embedding  │
                    │  project up→down│        │  + positional enc│
                    │  channel mixing │        │  → d_model tokens│
                    └────────┬────────┘        └────────┬─────────┘
                     keys/values                      queries
                             │                          │
                    ┌────────▼──────────────────────────▼─────────┐
                    │              Transformer                     │
                    │  ┌─────────────────────┐ ┌───────────────┐  │
                    │  │   Cross-attention   │→│  FFN + norm   │  │
                    │  │  EMG → keys/values  │ │  × num_layers │  │
                    │  │  kin tokens → query │ └───────────────┘  │
                    │  └─────────────────────┘                    │
                    └────────────────────┬────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     Latent z        │
                              │     (d_model)       │
                              └──┬───────┬───────┬──┘
                                 │       │       │
               ┌─────────────────▼─┐ ┌──▼──┐ ┌──▼─────────────────┐
               │  Kinematic head   │ │Gait │ │  Impedance head     │
               │  2-layer MLP      │ │head │ │  2-layer MLP (SAC)  │
               │  → (27,)          │ │→(1,)│ │  → (27,) log-σ+μ   │
               │  [θ,ω,α]_desired  │ └─────┘ │  [K, C, M] gains   │
               └─────────┬─────────┘         └────────┬────────────┘
                         │                             │
               ┌─────────▼─────────────────────────────▼────────────┐
               │             compute_impedance_torque                │
               │                                                     │
               │  τ = K(θ_d − θ) + C(ω_d − ω) + M(α_d − α)        │
               │                                                     │
               │  applied per dimension across 9D joint space:       │
               │  hip [roll, yaw, pitch] · knee · ankle              │
               └───────────────────────────┬─────────────────────────┘
                                           │
                          ┌────────────────┼───────────────┐
                          │                │               │
                   kin loss          torque loss       gait loss
                          │                │               │
                          └────────────────┴───────────────┘
                                      backprop (lr)
```

#### Output Vector Layout (27D, each head)

| Dims | Joint | Axes | Orders |
|---|---|---|---|
| 0–8 | Hip | Roll (longitudinal), Yaw (vertical), Pitch (lateral) | 0th, 1st, 2nd derivative |
| 9–17 | Knee | Roll, Yaw, Pitch | 0th, 1st, 2nd derivative |
| 18–26 | Ankle | Roll, Yaw, Pitch | 0th, 1st, 2nd derivative |

Kinematic head: order 0 = θ, order 1 = ω, order 2 = α.  
Impedance head: order 0 = K (stiffness), order 1 = C (damping), order 2 = M (inertia).

| Input tensor | Shape | Output head | Shape |
|---|---|---|---|
| EMG window | `(batch, 13, W)` | Next-step kinematics | `(27,)` |
| Kinematic state | `(batch, 27)` | Next gait percentage | `(1,)` |
| Gait percentage | `(batch, 1)` | Impedance gains (K, C, M) | `(27,)` |

### Impedance Torque Computation — `compute_impedance_torque`

For each of the 27 output dimensions (9 per joint × 3 joints):

$$\tau = K(\theta_\text{des} - \theta_\text{curr}) + C(\dot{\theta}_\text{des} - \dot{\theta}_\text{curr}) + M(\ddot{\theta}_\text{des} - \ddot{\theta}_\text{curr})$$

The resulting 9D torque vector and 27D kinematic vector are each reduced to a scalar loss, summed, and backpropagated.

### Key Components

#### `train_val_test_transformer()`
Core epoch-level training and evaluation routine. Handles the forward pass, masked multi-task loss computation (kinematics, gait, and optional impedance-derived torque), and optimization steps. Tracks the lowest validation loss and saves the best model state.

#### `meta_train_transformer_loop()`
Overarching foundation-model data loader and curriculum manager.

- **Chunked loading:** Iterates through physical `.pt` chunks sequentially to manage memory.
- **Dynamic masking:** Automatically updates the `EMGTransformer`'s modality masks (EMG, kinematics, kinetics) on the fly based on the currently loaded chunk's metadata.
- **Validation:** Runs one full pass over every validation chunk, computing `average_total_loss`, `average_kinematic_loss`, and `average_torque_loss` normalized by batch size and number of loss additions. Losses accumulate into `current_loss_dict`; once all eval chunks are parsed, a ceiling check compares each modality's loss against the previous best and saves the corresponding checkpoint if improved.

#### `check_load_time()`
Performance profiling utility. Measures the I/O bottleneck by timing `torch.load()` operations versus `DataLoader` instantiation overhead across all dataset chunks.

---

## `visualizer.py`

Plots loss curves and animations on test data.

---

## `rltrain.py` — RL Fine-Tuning for Prosthetic Control

Trains an RL policy (SAC or PPO) to control a prosthetic limb using EMG signals inside the [SCONE](https://scone.software/doku.php?id=doc:sconegym) simulation environment.

A **DEPrl base controller** drives the host-side muscle activations (intact limb / residual gait). The **prosthetic policy** (`EMGTransformer`, pre-trained via `trainFM.py`) is fine-tuned on top of it to maximise reward for a biomechanically coherent collaborative gait.

### Environment Setup

```
pip install -r scone_sim_requirements.txt
```

Requires a working **sconegym** install — setup guide: https://scone.software/doku.php?id=doc:sconegym

- Custom `.osim` and `.scone` files for prosthetic envs live in `sconegym/data-v1/`
- `gaitgym.py` is the custom env wrapper (observation layout, refresh rate)
- Register the reference env in `sconegym/init_v0.py`

On launch, `rltrain.py` will:
1. Resolve the amputation type → assign EMG and kinematic masks
2. Build or load the `EMGTransformer` policy and `QNetwork` critics
3. Spin up `--num_envs` parallel worker processes (spawned, CUDA-safe)
4. Run `rl_train_sac()` or `rl_train_ppo()` for the chosen number of steps

### State & Action Dimensions

| Component | Dim | Description |
|---|---|---|
| EMG state | `13 × 100` | Windowed EMG channels (1D conv encoder) |
| Kinematic state | `27` | Joint angles / velocities / accelerations |
| Action | `54` | `27-D` kinematic targets + `27-D` impedance parameters (K, C, M) |

Active dimensions are gated per amputation type via `emg_mask` and `kinematic_mask`. Only active action indices contribute to entropy calculations.

### SAC Training Loop — `rl_train_sac()`

Multiprocessing loop using shared memory to minimise host↔GPU transfers. Each step:

```
Phase 1  Wait for ALL workers to write observations   (Flag == 1)
Phase 2  Save completed transitions from previous step
Phase 3  Mass GPU forward pass  (reads directly from shared RAM via .to(device))
Phase 4  Format actions → write to shared memory
Phase 5  Release workers                              (Flag == 2)
Phase 6  SAC update  (1:1 step-to-learn ratio once min_replay_size is reached)
```

**Worker processes** (`worker_loop`): start envs, step with received actions, store SARSA tuples to the replay buffer.

**SAC update** (`train_sac`):
- Standard SAC with twin Q-networks (γ=0.99, τ=0.005) + **Conservative Q-Learning (CQL)** penalty (`cql_weight=5.0`)
- Action sampling: kinematic and impedance vectors drawn from independent 27-D Gaussian heads. Log-std is clamped, reparameterised, squashed, and shifted to `[0,1]`. Log-prob is corrected with the Jacobian (`−dF/dz`)
- Alpha: learnable entropy temperature with optional `LinearSchedule` warm-up override

**Replay buffer** (`NoisyReplayBuffer`): circular buffer with per-slot `worker_id` tracking. Temporal jitter uses `_walk_back()` — scans backward skipping interleaved slots from other workers and stopping at episode boundaries, so δ always means genuine temporal distance within a single env's episode.

### PPO Training Loop — `rl_train_ppo()`

Uses `RolloutBuffer` (extends `NoisyReplayBuffer`) which additionally stores `log_prob` and value estimates at collection time. On-policy rollouts are collected across `--num_envs` workers then used for `--epochs` gradient updates per iteration.

### Arguments

#### Checkpointing & Resume

```
--checkpoint_path       PATH      Load a specific checkpoint file. If provided,
                                  d_model / nhead / num_layers are read from the
                                  checkpoint's model_config (CLI args are fallback).
                                  Default: None

--checkpoint_dir        PATH      Directory for saving model checkpoints.
                                  Default: /gpfs/data/s001/vwulfek1/software/models/SAC

--resume                          Resume training (optimizer states, replay buffer)
                                  from the latest checkpoint in --checkpoint_dir.
                                  Default: False

--deprl_checkpoint      PATH      DEPrl base controller checkpoint (host-side gait).
                                  Default: /gpfs/data/.../step_13500000

--save_model_interval   INT       Save checkpoint every N training steps. Default: 100
--save_plot_interval    INT       Save training plots every N steps. Default: 1
--save_plot_dir         PATH      Output directory for training plots.
                                  Default: C:/EMG/software/tt
```

#### Algorithm Selection

```
--train_sac                       Run rl_train_sac(). Default: False
--train_ppo                       Run rl_train_ppo(). Default: False
```

> At least one must be set. Both can be set to run sequentially.

#### Replay Buffer

```
--replay_buffer_tag     STR       Tag for saving / reloading a named replay buffer.
                                  Default: None
--replay_buffer_size    INT       Circular buffer capacity (transitions). Default: 4096
```

#### Environment

```
--amputation_type       STR       Amputation config. Sets EMG + kinematic masks.
                                  Choices: transfemoral_left, transfemoral_right,
                                           transfemoral_both, transtibial_left,
                                           transtibial_right, transtibial_both
                                  Default: transfemoral_left
--num_envs              INT       Parallel env worker processes. Default: 2
```

#### Training Hyperparameters

```
--batch_size            INT       Mini-batch size for gradient updates. Default: 256
--epochs                INT       Gradient update epochs per iteration. Default: 100
--lr                    FLOAT     Learning rate (all optimizers). Default: 1e-4
--max_training_steps    INT       Total gradient update steps. Default: 100,000
--max_env_steps         INT       Env steps per rollout phase. Default: 20,000
--min_replay_size       INT       Buffer fill threshold before training starts. Default: 50
```

#### Policy (EMGTransformer) Architecture

```
--d_model               INT       Transformer model dimension. Default: 512
--nhead                 INT       Attention heads. Default: 4
--num_layers            INT       Encoder + decoder layers (shared count). Default: 4
--device                STR       'cuda' or 'cpu'. Default: cuda
--torch_compile                   Enable torch.compile(). Default: True
```

#### Critic (QNetwork) Architecture

```
--critic_h_dim              INT     MLP hidden dimension. Default: 512
--critic_emg_window_size    INT     EMG window size fed to critic. Default: 100
--critic_d_model            INT     Critic transformer dimension. Default: 512
--critic_nhead              INT     Critic attention heads. Default: 2
--critic_num_encoder_layers INT     Critic encoder layers. Default: 2
--critic_num_decoder_layers INT     Critic decoder layers. Default: 2
--critic_dim_feedforward    INT     Critic feedforward dimension. Default: 1024
--critic_dropout            FLOAT   Critic dropout rate. Default: 0.1
```

#### Sim-to-Real Noise & Jitter

All magnitude params are **upper bounds** — actual noise is domain-randomised each draw so the policy learns robustness across `[0, max]`, including the clean case.

```
--emg_noise_std_max     FLOAT     Max σ of per-channel Gaussian noise on EMG. Default: 1.0
--emg_noise_mean_max    FLOAT     Max |μ| of per-channel offset on EMG. Default: 0.0
--kin_noise_std_max     FLOAT     Max σ of Gaussian noise on kinematics. Default: 1.0
--kin_noise_mean_max    FLOAT     Max |μ| offset on kinematics. Default: 0.0
--emg_jitter_max        INT       Max temporal jitter δ (samples) for EMG. Default: 200
--kin_jitter_max        INT       Max temporal jitter δ (samples) for kinematics. Default: 5
--jitter_warmup_steps   INT       Steps over which jitter linearly ramps to max. Default: 0
--noise_on_rollout                Apply noise during env rollout collection. Default: False
--noise_on_replay                 Apply noise when sampling from replay buffer. Default: False
```

#### Example

```bash
python rltrain.py \
  --train_sac \
  --amputation_type transfemoral_left \
  --num_envs 4 \
  --batch_size 256 \
  --max_training_steps 100000 \
  --noise_on_rollout \
  --emg_noise_std_max 0.5 \
  --device cuda
```

---

## Current Progress

**Active:**
- RL fine-tuning on SCONEgym
- Meta-training on Moreira, Lencioni, Hu, Moghadam
- Successfully trained on Lencioni, Moreira, Embry, Hu

**To do:**
- Add synthetic noise and masking for training
- Check observation differences between depRL-trained env and new env
- Check effects of zero-state coordinate actuator (likely splitting into left/right prosthetic actuation envs)
- Collect patient info token data (kg, meters)
- Run `syncSignals` and `convert2DL` on Bacek

**Known data issues:**
- Criekinge: NaN errors on some torque and EMG channels in original data
- Camargo, Angelidou, Hu: NaN errors from original data
- Macaluso, Angelidou, Camargo: large kinetic range
- Grimmer: short gaits due to heel-strike-to-heel-strike segmentation (will likely keep continuous for Mk2)
- Kinetic normalization: Embry (claimed Nmm/kg), Macaluso (no units given) — follow-up emails sent
