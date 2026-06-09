# EMG–Kinematics–Kinetics Foundation Dataset

## Table of contents

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
  - [Output structure](#output-structure-per-window)
- [trainFM.py](#trainfmpy)
  - [Command line arguments](#command-line-arguments)
  - [Overview](#overview)
  - [Data pipeline](#data-pipeline)
  - [Batch format](#batch-format)
  - [Model architecture](#model-architecture--emgtransformer)
  - [Impedance torque computation](#impedance-torque-computation--compute_impedance_torque)
  - [Key components](#key-components)
- [visualizer.py](#visualizerpy)
- [Current progress](#current-progress)

---

## Project motivation
...

