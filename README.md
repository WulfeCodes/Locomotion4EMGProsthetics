Industrial powered prosthetics to this data still rely on finite state machines, have unrealistic cost requirements and are generally inaccessible to the average parapalegic, with around 70% of amputees never receiving a prosthetic. 

This project aims to collect open-source EMG, kinematic, and kinetic data, merge into a homogenous indexable format for foundational model training. 

The corresponding papers can be found through: https://drive.google.com/drive/folders/1vcUSvnTsDR734KYHGv7suk0HFPTf5C_a?usp=sharing 
The Camargo dataset does not have a paper, but is included within the datasets.

The distribution of the data is as follows: 
  Dataset            Stride Count   Percentage
  Bacek                   258,418       50.09%
  Macaluso                 66,035       12.80%
  Camargo                  53,713       10.41%
  K2Muse                   40,612        7.87%
  Angelidou                40,204        7.79%
  Embry                    26,846        5.20%
  Grimmer                  10,772        2.09%
  Hu                        6,365        1.23%
  Gait120                   6,310        1.22%
  Moreira                   2,613        0.51%
  Criekinge                 2,102        0.41%
  Lencioni                  1,159        0.22%
  SIAT                        441        0.09%
  Moghadam                    290        0.06%

Files and their respected functionality:
testEMG1.py: 

this parses, normalizes the EMG signals through a wavelet, notch, and bandpass filters through each of the given datasets. It then pads and structures the EMG and kinematic/kinetic data to be of 13,hz and 3,3,points shape respectively. 

The kinematic and kinetic data is structured in the following way: dim(0) = hip,knee,joint. dim(1) = adduction, rotation, and flexion. 
EMG data format: ['Vastus Lateralis','Rectus Femoris','Vastus Medialis','Tibialis Anterior','Biceps Femoris','Semitendinosus or sEMG: semimembranosus','Gastrocnemius Medialis','Gastrocnemius Lateralis','Soleus','Peroneus Longus','Peroneus Brevis',"gluteusmedius",'Gluteus Maximus']

Both the kinematic, kinetic, and emg have respective masks for the data availability where 1 is assigned to those indices where data exists and 0 where they don't. 
The data is then parsed into a dictionary format of patient, activity, trial, (possible subtrial), and segmented cycle format. This is saved into a .pkl: 

syncEMG.py:
  SyncAll(): This is the rosetta stone for accessing the structured .pkl data of each dataset
  analyze_sample_counts(): this counts the number of strides and sample count variance of each dataset, this was primarily used to recognizing which datasets needed additional processing for emg Hz conversion.

syncSignals(): 
  resample_all_datasets(): This unifies the emg data to a unified hz(default 1k) and kinematic/kinetic data to a unified sample count per stride(default 200)

convert2ML.py:
  This reformats into a more RAM friendly indexable format into a .hdf5 file with: 
| Array / Field                    | Length | Example Value (index *i*) | Purpose                                                   |
|----------------------------------|--------|--------------------------|-----------------------------------------------------------|
| `stride_starts`                  | N      | `10500`                  | Points to the start row in the raw EMG/Kinematics matrix. |
| `stride_ends`                    | N      | `10650`                  | Points to the end row in the raw EMG/Kinematics matrix.   |
| `stride_metadata.patient_id`     | N      | `5`                      | Global patient ID associated with this stride.           |
| `stride_metadata.dataset_id`     | N      | `"criekinge"`            | Dataset from which this stride was recorded.             |
| `stride_metadata.activity`       | N      | `"walk"`                 | Activity label for this stride.                           |

Unfortunately, the raw data spans 700GB and is not uploadable in its current format, as well as the pkl which is about 70GB. So they are not included in this current version.

  



