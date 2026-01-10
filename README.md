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
| resample_all_datasets() | Resamples EMG to 1 kHz and kinematics/kinetics to 200 samples per stride |

---

### convert2ML.py

Reformats data into an HDF5-backed, memory-efficient, indexable format.

| Field | Length | Example | Purpose |
|------|-------|--------|--------|
| stride_starts | N | 10500 | Start row of stride in raw matrices |
| stride_ends | N | 10650 | End row of stride |
| stride_metadata.patient_id | N | 5 | Global patient ID |
| stride_metadata.dataset_id | N | "criekinge" | Source dataset |
| stride_metadata.activity | N | "walk" | Activity label |

---

## Storage Constraints

| File Type | Size |
|---------|------|
| Raw datasets | ~700 GB |
| Processed `.pkl` | ~70 GB |

These files are not included in this repository due to size limitations.

  



