# EMG–Kinematics–Kinetics Foundation Dataset

## Current Progress
To Do: add synthetic noise and masking for training, check observation differences between depRL trained env and new env, check affects of zero state coordinate actuator(most likely ill just make a left and right prosthetic actuation env j to be safe)

Currently RL-finetuning on SCONEgym 

Performing Meta-Training on Moreira, Lencioni, Hu, Moghadam

Successfully trained on Lencioni, Moreira, Embry, Hu.

Coding up a new input token: (patient info: kg, meter)..collecting data

Need to use run syncSignals, convert2DL on Bacek

Criekinge has NaN errors from original data on some torque and EMG channels,

Camargo, Angelidou, and possibly Hu datasets have NaN errors from original data. 

Macaluso, Angelidou, and Camargo has large kinetic range.

Kinetic Normalization: Emailed: Embry:(Claimed Nmm/kg), Macaluso:(No Units Given)

Grimmer has short Gaits as a result of segmentation of Heel Strike to Heel Strike, (will likely keep continous for Mk2)

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

| Dataset    | Patients |
|------------|----------|
| lencioni   | 50       |
| grimmer    | 12       |
| criekinge  | 138      |
| moghadam   | 17       |
| moreira    | 16       |
| angelidou  | 9        |
| bacek      | 21       |
| hu         | 10       |
| gait120    | 110      |
| camargo    | 22       |
| macaluso   | 10       |
| k2muse     | 30       |
| siat       | 40       |
| embry      | 10       |
| **TOTAL**  | **495**  |

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

## `convert2DL.py`

Transforms stride-level biomechanical data into windowed, learning-ready samples aligned for predictive control and foundation model training.

**Core Purpose:** Bridge raw stride data and deep learning by constructing temporally aligned EMG–state–target tuples with deterministic dataset splits.

---

### Key Classes

#### `SplitDataset`
Container class implementing a PyTorch-compatible dataset for a single split (`train`, `val`, or `test`).
* **Storage:** Stores EMG windows, kinematic state vectors, gait percentage scalars, torque targets, and metadata.
* **Key Methods:**
  * `__getitem__()`: Returns tensors ready for GPU training.
  * `verify_lengths()`: Sanity-checks that all stored arrays are perfectly aligned.

#### `WindowedGaitDataParser`
Primary dataset parser and window generator. 
* **Responsibilities:**
  * Parse heterogeneous `.pkl` datasets with dataset-specific logic.
  * Enforce patient-level train/val/test splits (prevents subject data leakage).
  * Align EMG history windows with kinematic sampling.
  * Construct impedance-relevant state representations.

---

### Core Methods

* **`assign_patient_to_split()`**
  Deterministically assigns each patient to a split using hashing, ensuring reproducible datasets across runs.

* **`compute_omega()`**
  Computes joint angular velocity using causal finite differences from gait-normalized kinematics.

* **`compute_alpha()`**
  Computes joint angular acceleration using second-order finite differences (central when possible).

* **`extract_windows_aligned_to_kinematics()`**
  The core windowing routine. For each step, it:
  1. Maps each kinematic timestep to its corresponding EMG index.
  2. Extracts a fixed-length EMG history window (zero-padded if necessary).
  3. Builds the input kinematic state: `[θ, ω, α]` at time `t`.
  4. Builds the target kinematic state: `[θ, ω, α]` at time `t+1`.
  5. Optionally attaches torque targets when available.

* **`add_stride()`**
  Converts a single stride into multiple supervised learning windows and appends them to the correct dataset split.

* **`extract_masks()`**
  Normalizes dataset-specific modality masks (EMG, kinematics, kinetics) into a unified format used downstream for masking and loss computation.

* **`parse_*()`**
  Dataset-specific parsing functions that traverse each dataset’s internal hierarchy (patient → trial → stride) and invoke `add_stride()`.

* **`convert_all()`**
  Iterates through all known datasets and performs full conversion into windowed samples.

---

### Output Structure (Per Window)

| Feature | Shape | Description |
| :--- | :--- | :--- |
| `emg` | `(13, window_size)` | EMG history window. |
| `input_kin_state` | `(27,)` | Current joint state `[θ, ω, α]`. |
| `target_kin_state`| `(27,)` | Next-step joint state. Each 9 indices represent orders of differentiation for angles. Within a given group, the order is: hip (roll, yaw, pitch) → knee → ankle. |
| `input_gait_pct` | Scalar | Current gait percentage. |
| `target_gait_pct` | Scalar | Next-step gait percentage. |
| `target_torque` | `(9,)` | Joint torques (or zeros if unavailable). |
| `metadata` | Various | Activity type, dataset name, patient ID, and torque availability flag. |

## `trainFM.py`

Reference training pipeline for EMG-driven kinematic and impedance prediction using a transformer-based architecture.

**Core Purpose:** Demonstrate how the processed dataset can be used to train a foundation-style model that jointly predicts kinematics, gait phase, and impedance parameters.

> **Note:** This script is intended as a baseline and research scaffold, not a finalized production training pipeline.

---

### Key Components

#### `EMGTransformer` (`nn.Module`)
Multi-input transformer model combining EMG time-series and biomechanical state information.
* **Architecture Overview:**
  * **EMG Encoder:** 1D convolutional stack that embeds EMG time-series into a latent representation.
  * **State Embeddings:** Separate embeddings for kinematic state and gait phase.
  * **Transformer:** Encoder–decoder architecture fusing EMG context with state queries.
* **Masking Support:** EMG, kinematic, and kinetic masks are applied to inputs and losses, enabling training across heterogeneous datasets with missing channels or modalities.

| Input Tensors | Shape | Output Heads | Shape |
| :--- | :--- | :--- | :--- |
| **EMG Window** | `(batch, 13, 100)` | **Next-Step Kinematics** | `(27)` |
| **Kinematic State** | `(batch, 27)` | **Next Gait Percentage** | `(1)` |
| **Gait Percentage** | `(batch, 1)` | **Impedance (K, C, M)** | Per joint (Optional) |

---

### Physical Modeling & Control

* **`compute_impedance_torque()`**
  Implements the classical impedance control law to translate predicted impedance parameters and kinematic tracking error into joint torques for supervision:
  $\tau = K(\theta^{d} - \theta) + C(\omega^{d} - \omega) + M(\alpha^{d} - \alpha)$

---

### Training & Meta-Learning Loops

* **`train_val_test_transformer()`**
  The core epoch-level training and evaluation routine. 
  * Handles the forward pass, masked multi-task loss computation (kinematics, gait, and optional impedance-derived torque), and optimization steps.
  * Manages dynamic logging of individual loss components.
  * During validation (`split_type='val'`), it tracks the lowest loss and automatically saves the best-performing model state (`best_transformer_model.pth`).

* **`meta_train_transformer_loop()`**
  The overarching foundation-model data loader and curriculum manager.
  * **Proportional Epochs:** Calculates an inverse-proportional epoch count for 14 different datasets (e.g., smaller datasets get more `curr_epoch_iter` passes to prevent larger datasets from dominating the gradients).
  * **Chunked Loading:** Iterates through physical `.pt` chunks sequentially to manage memory.
  * **Dynamic Masking:** Automatically updates the `EMGTransformer`'s modality masks (EMG, kinematics, kinetics) on the fly based on the metadata of the currently loaded data chunk.
  * Invokes `train_val_test_transformer()` for the actual `train`, `val`, and `test` passes on each loaded chunk.

* **`check_load_time()`**
  A performance profiling utility. 
  * Measures the I/O bottleneck by timing the `torch.load()` operations versus the PyTorch `DataLoader` instantiation overhead across all dataset chunks.

---

### Execution

* **`main()`**
  Executable training entry point. Loads/parses datasets, builds DataLoaders, instantiates the `EMGTransformer` with dataset-specific masks, and launches the training loop with configurable hyperparameters.

Loads and parses datasets via WindowedGaitDataParser

Builds PyTorch DataLoaders for each split

Instantiates the EMGTransformer with dataset-specific masks

Launches training with configurable hyperparameters

This script is intended as a baseline and research scaffold, not a finalized production training pipeline.

## visualizer.py
plots loss and animation on test data.

## Storage Constraints

| File Type | Size |
|---------|------|
| Raw datasets | ~700 GB |
| Processed `.pkl` | ~70 GB |

These files included in this repository can be found through: https://drive.google.com/drive/folders/1Kba2_5XaiBluw-rXHpUfm8X4auCholB3?usp=sharing.


  



