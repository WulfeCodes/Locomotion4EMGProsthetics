import pandas as pd
import os
import numpy as np
import copy

np.seterr(divide='raise', invalid='raise')

from scipy.io import loadmat
from scipy.interpolate import interp1d
import h5py
import matlab.engine
import ezc3d
import c3d
from scipy.signal import resample, iirnotch, filtfilt, butter, find_peaks
import pickle
from pathlib import Path
import pywt
    
eng = matlab.engine.start_matlab()

#TODO if each trial is of a variable cycle count how can Hz be synced?
#TODO collect left leg data (last data parsing task)
#TODO proper interpolation
#TODO does patientMetaData improve performance?
#TODO does both legs and/or just sensor to kinematic data help performance?
#filtering analysis of each data type
#Future datasets: 
# 
#includes kinetic:Schulte, Wang!!
# The ISB GaitLab - Vaughan? NEEDs additional verification from purchase of book
#kirtley: Clinical Gait Analysis, no access!!
#SCHERPEREEL HAS TOO MANY NANs to be useful 
#Kotolova has interesting movement types, but complicated to parse

def detect_strides_vertical(imu_data, sampling_rate=1000):
    """
    Stride detection using vertical acceleration.
    Returns list of (start, end) indices for each stride.
    """
    
    # Get vertical acceleration (y-axis based on your data)
    acc_verticalR = imu_data['RF_accel.y'].values
    acc_verticalL = imu_data['LF_accel.y'].values
    
    # Apply low-pass filter to reduce noise
    def butter_lowpass_filter(data, cutoff=20, fs=1000, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    acc_filteredR = butter_lowpass_filter(acc_verticalR, cutoff=20, fs=sampling_rate)
    acc_filteredL = butter_lowpass_filter(acc_verticalL, cutoff=20, fs=sampling_rate)

    # Find peaks in vertical acceleration (heel strikes show as positive peaks)
    min_stride_samples = int(0.7 * sampling_rate)  # Min 0.4s between strides
    max_stride_samples = int(2.5 * sampling_rate)  # Max 2.5s between strides
    
    peaksR, propertiesR = find_peaks(
        acc_filteredR, 
        distance=min_stride_samples,
        prominence=np.std(acc_filteredR) * 0.3  # Adaptive threshold
    )

    peaksL, propertiesL = find_peaks(
        acc_filteredL, 
        distance=min_stride_samples,
        prominence=np.std(acc_filteredL) * 0.3  # Adaptive threshold
    )
    
    # Create stride list (from one heel strike to the next)
    stridesR = []
    stridesL = []

    for i in range(len(peaksR) - 1):
        start = peaksR[i]
        end = peaksR[i + 1]
        
        # Validate stride duration
        if min_stride_samples <= (end - start) <= max_stride_samples:
            print('R',imu_data.iloc[start]['Time'],imu_data.iloc[end]['Time'])

            stridesR.append({
                'startTime': imu_data.iloc[start]['Time'],
                'endTime': imu_data.iloc[end]['Time'],
                'start': start,
                'end': end,
                'duration_s': (end - start) / sampling_rate
            })

    for i in range(len(peaksL) - 1):
        start = peaksL[i]
        end = peaksL[i + 1]
        
        # Validate stride duration
        if min_stride_samples <= (end - start) <= max_stride_samples:
            print('L',imu_data.iloc[start]['Time'],imu_data.iloc[end]['Time'])
            stridesL.append({
                'startTime': imu_data.iloc[start]['Time'],
                'endTime': imu_data.iloc[end]['Time'],
                'start': start,
                'end': end,
                'duration_s': (end - start) / sampling_rate
            })
    
    return [stridesR, acc_filteredR, peaksR],[stridesL, acc_filteredL, peaksL]

def apply_notch(data, w0=50.0,fs=1926.0):
    # Create the filter (remove exactly 50Hz)
    b, a = iirnotch(w0=w0, Q=100.0, fs=fs)
    # Apply it forward and backward (filtfilt) to avoid phase shift
    return filtfilt(b, a, data)

def apply_bandpass(data, fs=1926.0):
    nyquist = 0.5 * fs
    low = 15.0 / nyquist
    high = 400.0 / nyquist
    
    # Paper specifies Order 7
    b, a = butter(N=7, Wn=[low, high], btype='band')
    return filtfilt(b, a, data)

def apply_wavelet_denoising(data):
    # The paper uses Wavelet Packet Denoising
    # We decompose the signal down to level 9 using 'db7'
    wp = pywt.WaveletPacket(data=data, wavelet='db7', mode='symmetric', maxlevel=9)
    
    # The paper specifies a threshold of 0.08
    threshold = 0.08
    
    # We iterate through every "node" (frequency band) in the packet
    # and "shrink" the noise (Soft Thresholding)
    for node in wp.get_level(9, 'freq'):
        node.data = pywt.threshold(node.data, threshold, mode='soft')
        
    # Reconstruct the clean signal from the packets
    return wp.reconstruct(update=True)

def parseDiNardo(currPath = 'C:/EMG/datasets/DiNardo/surface-electromyographic-signals-collected-during-long-lasting-ground-walking-of-young-able-bodied-subjects-1.0.0'):
    seenSet = []
    for currData in sorted(os.listdir(currPath)):
        if currData[:2] in seenSet:
            continue
        else:
            seenSet.append(currData[:2])
            currHEA = f'{currData[:2]}.hea'
            currDAT = f'{currData[:2]}.dat'

            metaInfo = {}


            with open(f'{currPath}/{currHEA}') as f:
                for line in f:
                    pass





    pass  
def parseScherpereel(currPath = "C:/EMG/datasets/Scherpereel"):
    pass
    #should include extra activities: sit2stand, stand2sit,squat, jump,etc
    listofActivities={"curb_up",'curb_down','dynamic_walk_1_heel-walk',
    'dynamic_walk_1_toe-walk','incline_walk_1_down5','incline_walk_1_down10',
     'incline_walk_1_up5','incline_walk_1_up10','meander_1','normal_walk_1_0-6',
     'normal_walk_1_1-2','normal_walk_1_1-8','normal_walk_1_1-8','normal_walk_1_2-0',
     'normal_walk_1_2-5','obstacle_walk_1','meander_1'}

    for folder in sorted(os.listdir(currPath)):
        activityPath = currPath + '/' + folder
        for currActivity in sorted(os.listdir(activityPath)):
            print("activity",currActivity)
            if currActivity in listofActivities or 'stair' in currActivity:
                dataPath=activityPath+'/'+currActivity
                for currDataType in sorted(os.listdir(dataPath)):
                    break
def parseHood(currPath = "C:/EMG/datasets/Hood/datasets"):
    pass
    #not an EMG dataset
    #1 leg amputated patients
def parseHuGaDB():
    pass
    #no reference angles
#MOISSENET AND WANG NEEDS OpenSim for moment calculation
def parseMoissenet(currPath = "C:/EMG/datasets/Moissenet/data"):
    c3d=ezc3d.c3d("C:/EMG/datasets/Moissenet/data/2014001/2014001_C1_01.c3d")
    print("debug",type(c3d))
    point_labels = c3d['parameters']['ANALOG']['LABELS']['value']

    print("--- NAMES ---")
    print(point_labels)
def parseWang():
    pass
#BOVI only avgs
def parseBovi(currPath = "C:/EMG/datasets/Bovi/1-s2.0-S0966636210002468-mmc3.xls"):
    pass

    data_dict=pd.read_excel(currPath,sheet_name=None)
    for data in sorted(data_dict.keys()):
        data = data.replace(" ","")
        if data == "EMG":
            pass



    # for row in range(len(df)):
    #     currEMG.append(df.iloc[row][-9:].values)
    #     currAngle_and_joint.append(df.iloc[row][1:-9].values)
#TODO DMITROVS EMGs SHOULD BE DOWNSAMPLED
def parseDmitrovData(currPath = "C:/EMG/datasets/Dimitrov/datasets"):
    #TODO split types of data for task type
    #data structure index pattern is as follows: 
    # data[i] = patient 
    # data[i][j] = trial
    patientEMG = []
    patientJointAngles = []

    angles = []

    taskType = ["Level_Ground","Ramp_Ambulation","Stairs_Ambulation"]
    speedType = ["Slow_Speed","Self_Selected_Speed","Fast_Speed"]
    #LEAVING OUT SIDESTEPPING DO WE CONSIDER THESE LEVEL GROUND?
    activityType = ['Walking','Stand_to_Walk','Sit_to_Stand_to_Walk']    
    for patient in os.listdir(currPath):

        currPatientWalkEMG = []
        currPatientWalkJointAngles = []
        currPatientWalkMoments = []

        currPatientStairEMG = []
        currPatientStairAngles = []
        currPatientStairMoments = []

        currPatientRampEMG = []
        currPatientRampAngles = []
        currPatientRampMoments = []

        dataPath = f"{currPath}/{patient}/{patient}"
        data=f"{dataPath}/{patient}.mat"
        eng.eval(f"data = load('{data}');", nargout=0)
        for i,task in enumerate(taskType): 
        #for task in taskType:
            if i ==0:
                
                for speed in speedType:
                    currPatientWalkEMG.append(np.array(eng.eval(f"data.{patient}.RightFoot_GaitCycle_Data.{taskType[i]}.Walking.{speed}.RightLeg_EMG")))
                    currPatientWalkJointAngles.append(eng.eval(f"data.{patient}.RightFoot_GaitCycle_Data.{taskType[i]}.Walking.{speed}.IK"))
                    currPatientWalkJointAngles.append(eng.eval(f"data.{patient}.RightFoot_GaitCycle_Data.{taskType[i]}.Walking.{speed}.ID"))
            
            elif i ==1:
                modes=eng.eval(f"fieldnames(data.{patient}.RightFoot_GaitCycle_Data.{taskType[i]})", nargout=1)
                for mode in modes:
                    speeds=eng.eval(f"fieldnames(data.{patient}.RightFoot_GaitCycle_Data.{taskType[i]}.{mode})", nargout=1)
                    for s,speed in enumerate(speeds):
                        if s == 0:
                            np.zeros()


                    




        patientJointAngles.append(currPatientJointAngles)
        patientEMG.append(currPatientEMG)
    
    print("parsed dmitrov",len(patientJointAngles),len(patientEMG)),
    print("shapes:", len(patientEMG[0]),patientEMG[0][0].shape,
          len(patientJointAngles[0]),patientJointAngles[0][0].shape)
    return patientEMG,patientJointAngles
def parseHunt(currPath = ""):
    pass
    #dataset is for sit motion
##ABOVE DATASETS NOT THE MOVE
def parseGait120(gait120Path = "C:/EMG/datasets/gait120/data"):
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    patientLevelWalkingEMG = []
    patientLevelWalkingAngle = []
    
    patientStairAscentEMG = []
    patientStairAscentAngle = []
    
    patientStairDescentEMG = []
    patientStairDescentAngle = []
    
    patientSlopeAscentEMG = []
    patientSlopeAscentAngle = []
    
    patientSlopeDescentEMG = []
    patientSlopeDescentAngle = []
    
    patientSitToStandEMG = []
    patientSitToStandAngle = []
    
    patientStandToSitEMG = []
    patientStandToSitAngle = []
  
    #1200{SubjectNum=120*len(trials)*len(steps)},101,12 of sEMG data for one typeTrials
    merged = ['Soleus Medialis','Soleus Lateralis']
    EMGs = ['Vastus Lateralis','Rectus Femoris','Vastus Medialis','Tibialis Anterior','Biceps Femoris', 'Semitendinosus','Gastrocnemuis Medialis','Gastrocnemius Lateralis',
            '(Soleus Medialis + Soleus Lateralis)','Peroneus Longus','Peroneus Brevis',0,0]
    
    Angles = ['hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r']

    trials = ['Trial01', 'Trial02', 'Trial03', 'Trial04', 'Trial05']
    steps = ['Step01', 'Step02']

    typeTrials = ["LevelWalking","StairAscent","StairDescent","SlopeAscent",'SlopeDescent',"SitToStand","StandToSit"]

    #TODO add search for .mot GRF vector
    for folder in sorted(os.listdir(gait120Path)):
        for subject in sorted(os.listdir(f"{gait120Path}/{folder}")):
            currPatientLevelWalkingEMG = []
            currPatientLevelWalkingAngle = []
            currPatientStairAscentEMG = []
            currPatientStairAscentAngle = []
            currPatientStairDescentEMG = []
            currPatientStairDescentAngle = []
            currPatientSlopeAscentEMG = []
            currPatientSlopeAscentAngle = []
            currPatientSlopeDescentEMG = []
            currPatientSlopeDescentAngle = []
            currPatientSitToStandEMG = []
            currPatientSitToStandAngle = []
            currPatientStandToSitEMG = []
            currPatientStandToSitAngle = []

            ##EMG MVC check
            maxEMGs=[0] * len(EMGs)
            for dataType in sorted(os.listdir(f"{gait120Path}/{folder}/{subject}")):
                if dataType == 'EMG':
                    currPath=f"{gait120Path}/{folder}/{subject}/{dataType}/ProcessedData.mat"
                    eng.eval(f"data=load('{currPath}')",nargout=0)
                    for i,trialType in enumerate(typeTrials):
                            for trialNum in trials:
                                if trialNum in eng.eval(f'fieldnames(data.{trialType})',nargout=1):
                                    for stepNum in steps:
                                        if stepNum in eng.eval(f'fieldnames(data.{trialType}.{trialNum})',nargout=1):
                                            for e,currEMG in enumerate(EMGs):
                                                if currEMG == 0: continue
                                                if '(' in currEMG:
                                                    currEMGChannel0 = np.array(eng.eval(f'data.{trialType}.{trialNum}.{stepNum}.EMGs_filt.("{merged[0]}")',nargout=1)).T
                                                    currEMGChannel1 = np.array(eng.eval(f'data.{trialType}.{trialNum}.{stepNum}.EMGs_filt.("{merged[1]}")',nargout=1)).T

                                                    maxEMGs[e]=max(maxEMGs[e],
                                                                   (np.percentile(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel0,w0=60.0,fs=2000),fs=2000))),99.5)
                                                                    +
                                                                    np.percentile(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel1,w0=60.0,fs=2000),fs=2000))),99.5))/2)
                                                else:
                                                    currEMGChannel = np.array(eng.eval(f'data.{trialType}.{trialNum}.{stepNum}.EMGs_filt.("{currEMG}")',nargout=1)).T
                                                    maxEMGs[e]=max(maxEMGs[e],np.percentile(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel,w0=60.0,fs=2000),fs=2000))),99.5))

            ##EMG MVC check

            for dataType in sorted(os.listdir(f"{gait120Path}/{folder}/{subject}")):
                if dataType == 'EMG':
                    currPath=f"{gait120Path}/{folder}/{subject}/{dataType}/ProcessedData.mat"
                    eng.eval(f"data=load('{currPath}')",nargout=0)
                    for i,trialType in enumerate(typeTrials):
                            for trialNum in trials:
                                if trialNum in eng.eval(f'fieldnames(data.{trialType})',nargout=1):
                                    for stepNum in steps:
                                        if stepNum in eng.eval(f'fieldnames(data.{trialType}.{trialNum})',nargout=1):
                                            for e,currEMG in enumerate(EMGs):

                                                if currEMG == 0: continue
                                                elif '(' in currEMG:
                                                    currEMGChannel0 = np.array(eng.eval(f'data.{trialType}.{trialNum}.{stepNum}.EMGs_filt.("{merged[0]}")',nargout=1)).T
                                                    currEMGChannel1 = np.array(eng.eval(f'data.{trialType}.{trialNum}.{stepNum}.EMGs_filt.("{merged[1]}")',nargout=1)).T
                                                    if maxEMGs[e] >0:
                                                        emg_filt[e]=((np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel0,w0=60.0,fs=2000),fs=2000)))+
                                                        np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel1,w0=60.0,fs=2000),fs=2000))))/(2*maxEMGs[e])).clip(max=1.0)
                                                    else: 
                                                        emg_filt[e]=((np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel0,w0=60.0,fs=2000),fs=2000)))+
                                                        np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel1,w0=60.0,fs=2000),fs=2000))))/2).clip(max=1.0)
                                                else: 
                                                    currEMGChannel = np.array(eng.eval(f'data.{trialType}.{trialNum}.{stepNum}.EMGs_filt.("{currEMG}")',nargout=1)).T
                                                    if e ==0:
                                                        emg_filt = np.zeros((len(EMGs),currEMGChannel.shape[1]))
                                                    if maxEMGs[e] > 0:
                                                        emg_filt[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel,w0=60.0,fs=2000),fs=2000)))/maxEMGs[e]).clip(max=1.0)
                                                    else: 
                                                        emg_filt[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currEMGChannel,w0=60.0,fs=2000),fs=2000)))).clip(max=1.0)

                                            if i==0:  # LevelWalking
                                                currPatientLevelWalkingEMG.append(emg_filt)
                                            elif i==1:  # StairAscent
                                                currPatientStairAscentEMG.append(emg_filt)
                                            elif i==2:  # StairDescent
                                                currPatientStairDescentEMG.append(emg_filt)
                                            elif i==3:  # SlopeAscent
                                                currPatientSlopeAscentEMG.append(emg_filt)
                                            elif i==4:  # SlopeDescent
                                                currPatientSlopeDescentEMG.append(emg_filt)
                                            elif i==5:  # SitToStand
                                                currPatientSitToStandEMG.append(emg_filt)
                                            else:  # StandToSit
                                                currPatientStandToSitEMG.append(emg_filt)

                elif dataType == "JointAngle":
                    currPath=f"{gait120Path}/{folder}/{subject}/{dataType}"
                    for i,trialType in enumerate(typeTrials):
                        for trialNum in trials:
                            for stepNum in steps:
                                if os.path.exists(f"{currPath}/{trialType}/{trialNum}/{stepNum.lower()}.mot"):
                                    #skip extra lines
                                    header_lines=0
                                    with open(f"{currPath}/{trialType}/{trialNum}/{stepNum.lower()}.mot", "r") as f:
                                        for line in f:
                                            header_lines += 1
                                            if "endheader" in line.lower():
                                                break
                                    df=pd.read_csv(f"{currPath}/{trialType}/{trialNum}/{stepNum.lower()}.mot", skiprows=header_lines, delim_whitespace=True)
                                    for a,angle in enumerate(Angles):
                                        if a ==0:
                                            currAngles=np.zeros((len(Angles),len(df)))
                                            currAnglesFull = np.zeros((len(joints),len(axis),len(df)))
                                            #Angles = ['hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r']
                                        if 'hip' in angle.lower():
                                            if 'flexion' in angle.lower():
                                                currAnglesFull[0,-1] = df[angle].values

                                            elif 'adduction' in angle.lower():
                                                currAnglesFull[0,0] = df[angle].values

                                            elif 'rotation' in angle.lower():
                                                currAnglesFull[0,1] = df[angle].values

                                        elif 'knee' in angle.lower():
                                            currAnglesFull[1,-1] = df[angle].values

                                        elif 'ankle' in angle.lower():
                                            currAnglesFull[-1,-1] = df[angle].values

                                        currAngles[a] = df[angle].values
                                    if i==0:  # LevelWalking
                                        currPatientLevelWalkingAngle.append(np.array(currAnglesFull))
                                    elif i==1:  # StairAscent
                                        currPatientStairAscentAngle.append(np.array(currAnglesFull))
                                    elif i==2:  # StairDescent
                                        currPatientStairDescentAngle.append(np.array(currAnglesFull))
                                    elif i==3:  # SlopeAscent
                                        currPatientSlopeAscentAngle.append(np.array(currAnglesFull))
                                    elif i==4:  # SlopeDescent
                                        currPatientSlopeDescentAngle.append(np.array(currAnglesFull))
                                    elif i==5:  # SitToStand
                                        currPatientSitToStandAngle.append(np.array(currAnglesFull))
                                    else:  # StandToSit
                                        currPatientStandToSitAngle.append(np.array(currAnglesFull))
                                        
            patientLevelWalkingEMG.append(currPatientLevelWalkingEMG)
            patientLevelWalkingAngle.append(currPatientLevelWalkingAngle)
            patientStairAscentEMG.append(currPatientStairAscentEMG)
            patientStairAscentAngle.append(currPatientStairAscentAngle)
            patientStairDescentEMG.append(currPatientStairDescentEMG)
            patientStairDescentAngle.append(currPatientStairDescentAngle)
            patientSlopeAscentEMG.append(currPatientSlopeAscentEMG)
            patientSlopeAscentAngle.append(currPatientSlopeAscentAngle)
            patientSlopeDescentEMG.append(currPatientSlopeDescentEMG)
            patientSlopeDescentAngle.append(currPatientSlopeDescentAngle)
            patientSitToStandEMG.append(currPatientSitToStandEMG)
            patientSitToStandAngle.append(currPatientSitToStandAngle)
            patientStandToSitEMG.append(currPatientStandToSitEMG)
            patientStandToSitAngle.append(currPatientStandToSitAngle)

    EMGMask = [1 if x != 0 else 0 for x in EMGs]
    angleMask = np.zeros((3,3))
    angleMask[0,:] = 1
    angleMask[1:,-1]=1 

    return {
        'mask' : {'emg':EMGMask,'angle':angleMask},
        'right': {
            'levelWalking': {'emg': patientLevelWalkingEMG, 'angle': patientLevelWalkingAngle},
            'stairAscent': {'emg': patientStairAscentEMG, 'angle': patientStairAscentAngle},
            'stairDescent': {'emg': patientStairDescentEMG, 'angle': patientStairDescentAngle},
            'slopeAscent': {'emg': patientSlopeAscentEMG, 'angle': patientSlopeAscentAngle},
            'slopeDescent': {'emg': patientSlopeDescentEMG, 'angle': patientSlopeDescentAngle},
            'sitToStand': {'emg': patientSitToStandEMG, 'angle': patientSitToStandAngle},
            'standToSit': {'emg': patientStandToSitEMG, 'angle': patientStandToSitAngle}
        }
    }

def parseMoreira(currPath = "C:/EMG/datasets/Moreira/MAT_files/MAT_files",emgSampleHz=1000):
    """
    Parse Moreira dataset with stride separation.
    
    Returns data shaped as (trial,stride, num_channels, 1001) where:
    - Each trial produces 2 samples (Stride 1 and Stride 2)
    - EMG: 4 channels per stride [VL, BF, TA, GAL]
    - Angles: 9 channels per stride [Hip_X, Hip_Y, Hip_Z, Knee_X, Knee_Y, Knee_Z, Ankle_X, Ankle_Y, Ankle_Z]
    - Torques: 9 channels per stride (same structure as angles)
    - Pelvis data (indices 0-5) is excluded
    """
    axisOrder = ['pitch','roll','yaw']
    joints = ['hip','knee','ankle']
    rotationOrder = ['hip_pitch','hip_roll','hip_yaw',
                 'knee_pitch','knee_roll','knee_yaw',
                 'ankle_pitch','ankle_roll','ankle_yaw']
    #         # --- COLUMN MAPPING: EMG DATA (8 Cols) ---
#     # Format: [Index] Label : Muscle Name

#     # 0: St1_VL   : Stride 1 Vastus Lateralis (Thigh/Quad)
#     # 1: St2_VL   : Stride 2 Vastus Lateralis
#     # 2: St1_BF   : Stride 1 Biceps Femoris (Hamstring)
#     # 3: St2_BF   : Stride 2 Biceps Femoris
#     # 4: St1_TA   : Stride 1 Tibialis Anterior (Shin)
#     # 5: St2_TA   : Stride 2 Tibialis Anterior
#     # 6: St1_GAL  : Stride 1 Gastrocnemius Lateralis (Calf)
#     # 7: St2_GAL  : Stride 2 Gastrocnemius Lateralis

#         # ==========================================
#     #  COLUMN MAPPING: JOINT ANGLES (Degrees)
#     # ==========================================
#     # Sign Convention Source: [cite: 511-516]
#     # X (Sagittal):   (+) Flexion/Dorsiflexion | (-) Extension/Plantarflexion
#     # Y (Coronal):    (+) Adduction/Inversion  | (-) Abduction/Eversion
#     # Z (Transversal):(+) Internal Rotation    | (-) External Rotation 
#     # Note: Ankle Z (+) is Adduction / (-) Abduction

#     # --- PELVIS ANGLES ---
#     # 0 : St1_Pelvis_X (Sagittal)   - Stride 1 Pelvis Tilt
#     # 1 : St1_Pelvis_Y (Coronal)    - Stride 1 Pelvis Obliquity
#     # 2 : St1_Pelvis_Z (Transversal)- Stride 1 Pelvis Rotation
#     # 3 : St2_Pelvis_X (Sagittal)   - Stride 2 Pelvis Tilt
#     # 4 : St2_Pelvis_Y (Coronal)    - Stride 2 Pelvis Obliquity
#     # 5 : St2_Pelvis_Z (Transversal)- Stride 2 Pelvis Rotation

#     # --- HIP ANGLES ---
#     # 6 : St1_Hip_X (Sagittal)      - Stride 1 Hip Flexion (+) / Extension (-)
#     # 7 : St1_Hip_Y (Coronal)       - Stride 1 Hip Adduction (+) / Abduction (-)
#     # 8 : St1_Hip_Z (Transversal)   - Stride 1 Hip Int. Rot (+) / Ext. Rot (-)
#     # 9 : St2_Hip_X (Sagittal)      - Stride 2 Hip Flexion (+) / Extension (-)
#     # 10: St2_Hip_Y (Coronal)       - Stride 2 Hip Adduction (+) / Abduction (-)
#     # 11: St2_Hip_Z (Transversal)   - Stride 2 Hip Int. Rot (+) / Ext. Rot (-)

#     # --- KNEE ANGLES ---
#     # 12: St1_Knee_X (Sagittal)     - Stride 1 Knee Flexion (+) / Extension (-)
#     # 13: St1_Knee_Y (Coronal)      - Stride 1 Knee Adduction (+) / Abduction (-)
#     # 14: St1_Knee_Z (Transversal)  - Stride 1 Knee Int. Rot (+) / Ext. Rot (-)
#     # 15: St2_Knee_X (Sagittal)     - Stride 2 Knee Flexion (+) / Extension (-)
#     # 16: St2_Knee_Y (Coronal)      - Stride 2 Knee Adduction (+) / Abduction (-)
#     # 17: St2_Knee_Z (Transversal)  - Stride 2 Knee Int. Rot (+) / Ext. Rot (-)

#     # --- ANKLE ANGLES ---
#     # 18: St1_Ankle_X (Sagittal)    - Stride 1 Dorsiflexion (+) / Plantarflexion (-)
#     # 19: St1_Ankle_Y (Coronal)     - Stride 1 Inversion (+) / Eversion (-)
#     # 20: St1_Ankle_Z (Transversal) - Stride 1 Adduction (+) / Abduction (-)
#     # 21: St2_Ankle_X (Sagittal)    - Stride 2 Dorsiflexion (+) / Plantarflexion (-)
#     # 22: St2_Ankle_Y (Coronal)     - Stride 2 Inversion (+) / Eversion (-)
#     # 23: St2_Ankle_Z (Transversal) - Stride 2 Adduction (+) / Abduction (-)

#     # ==========================================
#     #  COLUMN MAPPING: JOINT TORQUES (N.m/kg)
#     # ==========================================
#     # Sign Convention Source: [cite: 517-522]
#     # X (Sagittal):   (+) Flexion Moment    | (-) Extension Moment
#     # Y (Coronal):    (+) Adduction Moment  | (-) Abduction Moment
#     # Z (Transversal): See specific joint below (conventions vary by joint)

#     # --- PELVIS TORQUES ---
#     # 0 : St1_Pelvis_X (Sagittal)   - Stride 1 Pelvis Tilt Moment
#     # 1 : St1_Pelvis_Y (Coronal)    - Stride 1 Pelvis Obliquity Moment
#     # 2 : St1_Pelvis_Z (Transversal)- Stride 1 Pelvis Rotation Moment
#     # 3 : St2_Pelvis_X (Sagittal)   - Stride 2 Pelvis Tilt Moment
#     # 4 : St2_Pelvis_Y (Coronal)    - Stride 2 Pelvis Obliquity Moment
#     # 5 : St2_Pelvis_Z (Transversal)- Stride 2 Pelvis Rotation Moment

#     # --- HIP TORQUES ---
#     # 6 : St1_Hip_X (Sagittal)      - Stride 1 Hip Flexion (+) / Extension (-)
#     # 7 : St1_Hip_Y (Coronal)       - Stride 1 Hip Adduction (+) / Abduction (-)
#     # 8 : St1_Hip_Z (Transversal)   - Stride 1 Hip Ext. Rot (+) / Int. Rot (-)
#     # 9 : St2_Hip_X (Sagittal)      - Stride 2 Hip Flexion (+) / Extension (-)
#     # 10: St2_Hip_Y (Coronal)       - Stride 2 Hip Adduction (+) / Abduction (-)
#     # 11: St2_Hip_Z (Transversal)   - Stride 2 Hip Ext. Rot (+) / Int. Rot (-)

#     # --- KNEE TORQUES ---
#     # 12: St1_Knee_X (Sagittal)     - Stride 1 Knee Flexion (+) / Extension (-)
#     # 13: St1_Knee_Y (Coronal)      - Stride 1 Knee Adduction (+) / Abduction (-)
#     # 14: St1_Knee_Z (Transversal)  - Stride 1 Knee Int. Rot (+) / Ext. Rot (-)
#     # 15: St2_Knee_X (Sagittal)     - Stride 2 Knee Flexion (+) / Extension (-)
#     # 16: St2_Knee_Y (Coronal)      - Stride 2 Knee Adduction (+) / Abduction (-)
#     # 17: St2_Knee_Z (Transversal)  - Stride 2 Knee Int. Rot (+) / Ext. Rot (-)

#     # --- ANKLE TORQUES ---
#     # 18: St1_Ankle_X (Sagittal)    - Stride 1 Dorsiflexion (+) / Plantarflexion (-)
#     # 19: St1_Ankle_Y (Coronal)     - Stride 1 Inversion (+) / Eversion (-)
#     # 20: St1_Ankle_Z (Transversal) - Stride 1 Adduction (+) / Abduction (-)
#     # 21: St2_Ankle_X (Sagittal)    - Stride 2 Dorsiflexion (+) / Plantarflexion (-)
#     # 22: St2_Ankle_Y (Coronal)     - Stride 2 Inversion (+) / Eversion (-)
#     # 23: St2_Ankle_Z (Transversal) - Stride 2 Adduction (+) / Abduction (-)
    
    # Initialize storage for both legs
    MoreiraEMGs = ['VL',0,0,'TA','BF',0,0,'GAL',0,0,0,0,0]
    moreira_emg_index = {
        'VL': 0,
        'TA': 3,
        'BF': 4,
        'GAL': 7,
        0: [1, 2, 5, 6, 8, 9, 10, 11, 12]  # or handle these separately
    }
    OriginalEMGOrder = ['VL','BF','TA','GAL']
    Right_patientEMG = []
    Right_patientJointAngle = []
    Right_patientTorque = []
    Right_patientGRF = []
    
    Left_patientEMG = []
    Left_patientJointAngle = []
    Left_patientTorque = []
    Left_patientGRF = []

    velocityList = ["V1", "V2", "V3", "V4", "V15", "V25", "V35"]
    
    for participant in sorted(os.listdir(currPath)):
        print(participant)
        Right_currPatientEMG = []
        Right_currPatientJointAngles = []
        Right_currPatientTorque = []
        Right_currPatientGRF = []
        
        Left_currPatientEMG = []
        Left_currPatientJointAngles = []
        Left_currPatientTorque = []
        Left_currPatientGRF = []

        data = f"{currPath}/{participant}/Processed_Data.mat"
        eng.eval(f"data = load('{data}');", nargout=0)
        
        for velocity in velocityList:
            if participant[-2] == "1":
                # Load RIGHT leg data
                eng.eval(f"temp_emg_R = data.Subject{participant[-2:]}_pro.{velocity}.R.EMGs_filt;", nargout=0)
                eng.eval(f"temp_angle_R = data.Subject{participant[-2:]}_pro.{velocity}.R.Angles;", nargout=0)
                eng.eval(f"temp_torque_R = data.Subject{participant[-2:]}_pro.{velocity}.R.Torques;", nargout=0)
                eng.eval(f"temp_grf_R = data.Subject{participant[-2:]}_pro.{velocity}.R.GRF;", nargout=0)
                
                # Load LEFT leg data
                eng.eval(f"temp_emg_L = data.Subject{participant[-2:]}_pro.{velocity}.L.EMGs_filt;", nargout=0)
                eng.eval(f"temp_angle_L = data.Subject{participant[-2:]}_pro.{velocity}.L.Angles;", nargout=0)
                eng.eval(f"temp_torque_L = data.Subject{participant[-2:]}_pro.{velocity}.L.Torques;", nargout=0)
                eng.eval(f"temp_grf_L = data.Subject{participant[-2:]}_pro.{velocity}.L.GRF;", nargout=0)
                print('check:',velocity,participant)
                part = participant[-2:]


                strideCheck=eng.eval(f'data.Subject{participant[-2:]}_pro.{velocity}.L.Angles{{1,2}}.Properties.VariableNames',nargout=1)
                strideCheck0=eng.eval(f'data.Subject{participant[-2:]}_pro.{velocity}.R.Angles{{1,2}}.Properties.VariableNames',nargout=1)
                    
            else:
                if ((velocity == "V4" or velocity == "V35" or velocity == 'V1') and participant[-1] == "4"):
                    continue
                
                # Load RIGHT leg data
                part = participant[-1]

                eng.eval(f"temp_emg_R = data.Subject{participant[-1]}_pro.{velocity}.R.EMGs_filt;", nargout=0)
                eng.eval(f"temp_angle_R = data.Subject{participant[-1]}_pro.{velocity}.R.Angles;", nargout=0)
                eng.eval(f"temp_torque_R = data.Subject{participant[-1]}_pro.{velocity}.R.Torques;", nargout=0)
                eng.eval(f"temp_grf_R = data.Subject{participant[-1]}_pro.{velocity}.R.GRF;", nargout=0)
                
                # Load LEFT leg data
                eng.eval(f"temp_emg_L = data.Subject{participant[-1]}_pro.{velocity}.L.EMGs_filt;", nargout=0)
                eng.eval(f"temp_angle_L = data.Subject{participant[-1]}_pro.{velocity}.L.Angles;", nargout=0)
                eng.eval(f"temp_torque_L = data.Subject{participant[-1]}_pro.{velocity}.L.Torques;", nargout=0)
                eng.eval(f"temp_grf_L = data.Subject{participant[-1]}_pro.{velocity}.L.GRF;", nargout=0)
                print('check:',velocity,participant)
                strideCheck=eng.eval(f'data.Subject{participant[-1]}_pro.{velocity}.L.Angles{{1,2}}.Properties.VariableNames',nargout=1)
                strideCheck0=eng.eval(f'data.Subject{participant[-1]}_pro.{velocity}.R.Angles{{1,2}}.Properties.VariableNames',nargout=1)

            for s in strideCheck:
                if 'St2' in s:
                    print('found some st2')
                    ok=True
                    
                    break
                else: ok=False
            
            # Process trials for RIGHT leg

            for trial in range(int(eng.eval("size(temp_emg_R,1)"))):
                stride1RTime = eng.eval(f'data.Subject{part}_pro.{velocity}.R.Stride_Time.St1_Time({trial+1})',nargout=1)
                stride2RTime = eng.eval(f'data.Subject{part}_pro.{velocity}.R.Stride_Time.St2_Time({trial+1})',nargout=1)
                stride1RCount = int(stride1RTime * emgSampleHz)
                stride2RCount = int(stride2RTime * emgSampleHz)

                trial_emg = np.array(eng.eval(f"table2array(temp_emg_R{{{trial+1}, 2}})", nargout=1))
                trial_angle = np.array(eng.eval(f"table2array(temp_angle_R{{{trial+1}, 2}})", nargout=1))
                trial_torque = np.array(eng.eval(f"table2array(temp_torque_R{{{trial+1}, 2}})", nargout=1))
                trial_grf = np.array(eng.eval(f"table2array(temp_grf_R{{{trial+1}, 2}})", nargout=1))
                assert trial_emg.shape[1]==8

                for e in range(trial_emg.shape[1]):
                    if e ==0:
                        formattedEMG = np.zeros((2,trial_emg.shape[1]//2,trial_emg.shape[0]))

                    formattedEMG[e%2,e//2]=trial_emg[:,e].T

                StrideEMG = []
                old_indices = np.linspace(0, 1, formattedEMG.shape[-1])

                for z in range(formattedEMG.shape[0]):

                    if z==1:
                        fullEMG = np.zeros((len(MoreiraEMGs),stride2RCount))
                        new_indices = np.linspace(0, 1, stride2RCount)
                    else:
                        fullEMG = np.zeros((len(MoreiraEMGs),stride1RCount))
                        new_indices = np.linspace(0, 1, stride1RCount)


                    for y in range(formattedEMG.shape[1]):

                        interpolator = interp1d(old_indices, formattedEMG[z,y,:], kind='linear')
                        resampled_emg = interpolator(new_indices)
        
                        fullEMG[moreira_emg_index[OriginalEMGOrder[y]]]=resampled_emg
                    StrideEMG.append(np.array(fullEMG))
                                                        
                formattedEMG = StrideEMG

                channel_idx = 0
                for t in range(6, trial_torque.shape[1], 3):
                    if channel_idx == 0:
                        formattedTorque = np.zeros((2,(trial_torque.shape[1]//2)-3,trial_torque.shape[0]))
                        formattedTorqueFull = np.zeros((2,len(joints),len(axisOrder),trial_torque.shape[0]))
                    stride = (t // 3) % 2  # Alternates 0,1,0,1,0,1...
                    joint = channel_idx // 2  # 0,0,1,1,2,2 for hip,hip,knee,knee,ankle,ankle
                    formattedTorque[stride, joint*3:(joint*3)+3] = trial_torque[:, t:t+3].T
                    channel_idx += 1

                channel_idx = 0
                for a in range(6, trial_angle.shape[1], 3):
                    if channel_idx == 0: 
                        formattedAngle = np.zeros((2,(trial_angle.shape[1]//2)-3,trial_angle.shape[0]))
                        formattedAngleFull = np.zeros((2,len(joints),len(axisOrder),trial_angle.shape[0]))
                    stride = (a // 3) % 2  # Alternates 0,1,0,1,0,1...
                    joint = channel_idx // 2  # 0,0,1,1,2,2 for hip,hip,knee,knee,ankle,ankle
                    formattedAngle[stride, joint*3:(joint*3)+3] = trial_angle[:, a:a+3].T
                    channel_idx += 1
                # Transpose to (num_channels, 1001) and append both strides
                ##CODE FOR REFORMATTING
                #OriginalAxisOrder = ['pitch','roll','yaw']

                for i,rotationType in enumerate(rotationOrder):
                    if i<3:
                        if 'pitch' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,0,-1,:]=formattedAngle[j,0,:]
                                formattedTorqueFull[j,0,-1,:]=formattedTorque[j,0,:]

                        elif 'yaw' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,0,1,:]=formattedAngle[j,2,:]
                                formattedTorqueFull[j,0,1,:]=formattedTorque[j,2,:]

                        elif 'roll' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,0,0,:]=formattedAngle[j,1,:]
                                formattedTorqueFull[j,0,0,:]=formattedTorque[j,1,:]
                    
                    elif i<6:
                        if 'pitch' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,1,-1,:]=formattedAngle[j,0+3,:]
                                formattedTorqueFull[j,1,-1,:]=formattedTorque[j,0+3,:]

                        elif 'yaw' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,1,1,:]=formattedAngle[j,2+3,:]
                                formattedTorqueFull[j,1,1,:]=formattedTorque[j,2+3,:]

                        elif 'roll' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,1,0,:]=formattedAngle[j,1+3,:]
                                formattedTorqueFull[j,1,0,:]=formattedTorque[j,1+3,:]
                    
                    else: 
                        if 'pitch' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,-1,-1,:]=formattedAngle[j,0+6,:]
                                formattedTorqueFull[j,-1,-1,:]=formattedTorque[j,0+6,:]

                        elif 'yaw' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,-1,1,:]=formattedAngle[j,-1,:]
                                formattedTorqueFull[j,-1,1,:]=formattedTorque[j,-1,:]

                        elif 'roll' in rotationType.lower():
                            for j in range(formattedAngle.shape[0]):
                                formattedAngleFull[j,-1,0,:]=formattedAngle[j,1+6,:]
                                formattedTorqueFull[j,-1,0,:]=formattedTorque[j,1+6,:]

                ##CODE FOR REFORMATTING

                Right_currPatientEMG.append(formattedEMG)
                
                Right_currPatientJointAngles.append(formattedAngleFull)

                Right_currPatientTorque.append(formattedTorqueFull)

                  
            # Process trials for LEFT leg
            for trial in range(int(eng.eval("size(temp_emg_L,1)"))):
                trial_emg = np.array(eng.eval(f"table2array(temp_emg_L{{{trial+1}, 2}})", nargout=1))
                trial_angle = np.array(eng.eval(f"table2array(temp_angle_L{{{trial+1}, 2}})", nargout=1))
                trial_torque = np.array(eng.eval(f"table2array(temp_torque_L{{{trial+1}, 2}})", nargout=1))
                trial_grf = np.array(eng.eval(f"table2array(temp_grf_L{{{trial+1}, 2}})", nargout=1))

                stride1LTime = eng.eval(f'data.Subject{part}_pro.{velocity}.L.Stride_Time.St1_Time({trial+1})',nargout=1)
                if ok:
                    stride2LTime = eng.eval(f'data.Subject{part}_pro.{velocity}.L.Stride_Time.St2_Time({trial+1})',nargout=1)
                    stride2LCount = int(stride2LTime * emgSampleHz)

                stride1LCount = int(stride1LTime * emgSampleHz)

                for e in range(trial_emg.shape[1]):
                    if e ==0 and ok:
                        assert trial_emg.shape[1]==8
                        formattedEMG = np.zeros((2,trial_emg.shape[1]//2,trial_emg.shape[0]))
                    elif e==0 and not ok:
                        assert trial_emg.shape[1]==4

                        formattedEMG = np.zeros((trial_emg.shape[1],trial_emg.shape[0]))
                    if ok:
                        formattedEMG[e%2,e//2]=trial_emg[:,e].T
                    else:
                        formattedEMG[e]=trial_emg[:,e].T
                old_indices = np.linspace(0, 1, formattedEMG.shape[-1])

                #put in full shape
                StrideEMG = []

                if ok: 
                    for z in range(formattedEMG.shape[0]):

                        if z==0:
                            new_indices = np.linspace(0,1,stride1LCount)
                            fullEMG = np.zeros((len(MoreiraEMGs),stride1LCount))

                        else: 
                            new_indices = np.linspace(0,1,stride2LCount)
                            fullEMG = np.zeros((len(MoreiraEMGs),stride2LCount))


                        for y in range(formattedEMG.shape[1]):
                            interpolator = interp1d(old_indices, formattedEMG[z,y,:], kind='linear')
                            resampled_emg = interpolator(new_indices)
                            fullEMG[moreira_emg_index[OriginalEMGOrder[y]]]=resampled_emg
                        StrideEMG.append(np.array(fullEMG))
                    formattedEMG=StrideEMG
                
                else:
                    new_indices = np.linspace(0,1,stride1LCount)
                    fullEMG = np.zeros((len(MoreiraEMGs),stride1LCount))
                    for y in range(formattedEMG.shape[0]):
                        interpolator = interp1d(old_indices, formattedEMG[y,:], kind='linear')
                        resampled_emg = interpolator(new_indices)
                        fullEMG[moreira_emg_index[OriginalEMGOrder[y]]]=resampled_emg
                    StrideEMG.append(np.array(fullEMG))
                    formattedEMG = StrideEMG

                ##
                if ok: startDex = 6
                else: startDex = 0
                channel_idx = 0
                for t in range(startDex, trial_torque.shape[1], 3):
                    if t==startDex and ok:
                        formattedTorqueFull = np.zeros((2,len(joints),len(axisOrder),trial_angle.shape[0]))
                        formattedTorque = np.zeros((2,(trial_torque.shape[1]//2)-3,trial_torque.shape[0]))
                    elif t==startDex and not ok:
                        formattedTorque = np.zeros((trial_torque.shape[1],trial_torque.shape[0]))
                        formattedTorqueFull=np.zeros((len(joints),len(axisOrder),trial_angle.shape[0]))

                    if ok: 
                        stride = (t // 3) % 2  # Alternates 0,1,0,1,0,1...
                        joint = channel_idx // 2  # 0,0,1,1,2,2 for hip,hip,knee,knee,ankle,ankle
                        formattedTorque[stride, joint*3:(joint*3)+3] = trial_torque[:, t:t+3].T
                        channel_idx += 1
                    else: 
                        joint_idx = (t - startDex) // 3  # 0,1,2 for hip,knee,ankle
                        formattedTorque[joint_idx*3:(joint_idx*3)+3] = trial_torque[:, t:t+3].T

                channel_idx = 0
                for a in range(startDex, trial_angle.shape[1], 3):
                    if a==startDex and ok:
                        formattedAngle = np.zeros((2,(trial_angle.shape[1]//2)-3,trial_angle.shape[0]))
                        formattedAngleFull = np.zeros((2,len(joints),len(axisOrder),trial_angle.shape[0]))

                    elif a==startDex and not ok:
                        formattedAngle = np.zeros((trial_angle.shape[1],trial_angle.shape[0]))
                        formattedAngleFull = np.zeros((len(joints),len(axisOrder),trial_angle.shape[0]))

                    if ok: 
                        stride = (a // 3) % 2  # Alternates 0,1,0,1,0,1...
                        joint = channel_idx // 2  # 0,0,1,1,2,2 for hip,hip,knee,knee,ankle,ankle
                        formattedAngle[stride, joint*3:(joint*3)+3] = trial_angle[:, a:a+3].T
                        channel_idx += 1
                    else: 
                        joint_idx = (a - startDex) // 3  # 0,1,2 for hip,knee,ankle
                        formattedAngle[joint_idx*3:(joint_idx*3)+3] = trial_angle[:, a:a+3].T
                # Transpose to (num_channels, 1001) and append both strides
                ##
                if ok: parseDex = 2
                else: parseDex = 1

                for i,rotationType in enumerate(rotationOrder):
                    if i<3:
                        if 'pitch' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,0,-1,:]=formattedAngle[j,0,:]
                                    formattedTorqueFull[j,0,-1,:]=formattedTorque[j,0,:]
                                else:
                                    formattedAngleFull[0,-1,:]=formattedAngle[0,:]
                                    formattedTorqueFull[0,-1,:]=formattedTorque[0,:]

                        elif 'yaw' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,0,1,:]=formattedAngle[j,2,:]
                                    formattedTorqueFull[j,0,1,:]=formattedTorque[j,2,:]
                                else:
                                    formattedAngleFull[0,1,:]=formattedAngle[2,:]
                                    formattedTorqueFull[0,1,:]=formattedTorque[2,:]

                        elif 'roll' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,0,0,:]=formattedAngle[j,1,:]
                                    formattedTorqueFull[j,0,0,:]=formattedTorque[j,1,:]
                                else:
                                    formattedAngleFull[0,0,:]=formattedAngle[1,:]
                                    formattedTorqueFull[0,0,:]=formattedTorque[1,:]
                    
                    elif i<6:
                        if 'pitch' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,1,-1,:]=formattedAngle[j,0+3,:]
                                    formattedTorqueFull[j,1,-1,:]=formattedTorque[j,0+3,:]
                                else:
                                    formattedAngleFull[1,-1,:]=formattedAngle[3,:]
                                    formattedTorqueFull[1,-1,:]=formattedTorque[3,:]

                        elif 'yaw' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,1,1,:]=formattedAngle[j,2+3,:]
                                    formattedTorqueFull[j,1,1,:]=formattedTorque[j,2+3,:]
                                else:
                                    formattedAngleFull[1,1,:]=formattedAngle[5,:]
                                    formattedTorqueFull[1,1,:]=formattedTorque[5,:]

                        elif 'roll' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,1,0,:]=formattedAngle[j,1+3,:]
                                    formattedTorqueFull[j,1,0,:]=formattedTorque[j,1+3,:]
                                else:
                                    formattedAngleFull[1,0,:]=formattedAngle[4,:]
                                    formattedTorqueFull[1,0,:]=formattedTorque[4,:]
                    
                    else: 
                        if 'pitch' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,-1,-1,:]=formattedAngle[j,0+6,:]
                                    formattedTorqueFull[j,-1,-1,:]=formattedTorque[j,0+6,:]
                                else:
                                    formattedAngleFull[-1,-1,:]=formattedAngle[6,:]
                                    formattedTorqueFull[-1,-1,:]=formattedTorque[6,:]

                        elif 'yaw' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,-1,1,:]=formattedAngle[j,-1,:]
                                    formattedTorqueFull[j,-1,1,:]=formattedTorque[j,-1,:]
                                else:
                                    formattedAngleFull[-1,1,:]=formattedAngle[-1,:]
                                    formattedTorqueFull[-1,1,:]=formattedTorque[-1,:]

                        elif 'roll' in rotationType.lower():
                            for j in range(parseDex):
                                if ok:
                                    formattedAngleFull[j,-1,0,:]=formattedAngle[j,1+6,:]
                                    formattedTorqueFull[j,-1,0,:]=formattedTorque[j,1+6,:]
                                else:
                                    formattedAngleFull[-1,0,:]=formattedAngle[7,:]
                                    formattedTorqueFull[-1,0,:]=formattedTorque[7,:]
                if not ok:
                    formattedAngleFull = np.expand_dims(formattedAngleFull, axis=0)
                    formattedTorqueFull = np.expand_dims(formattedTorqueFull, axis=0)
                Left_currPatientEMG.append(formattedEMG)
                
                Left_currPatientJointAngles.append(formattedAngleFull)
                
                Left_currPatientTorque.append(formattedTorqueFull)

        # Stack all trials*strides for this patient
        Right_patientEMG.append(Right_currPatientEMG)
        Right_patientTorque.append(Right_currPatientTorque)
        Right_patientJointAngle.append(Right_currPatientJointAngles)
        
        Left_patientEMG.append(Left_currPatientEMG)
        Left_patientTorque.append(Left_currPatientTorque)
        Left_patientJointAngle.append(Left_currPatientJointAngles)

    leftEMGMask =  [1 if x != 0 else 0 for x in MoreiraEMGs]
    rightEMGMask =  [1 if x != 0 else 0 for x in MoreiraEMGs]
    leftAngleMask = np.ones((3,3))
    rightAngleMask = np.ones((3,3))
    leftMomentMask = np.ones((3,3))
    rightMomentMask = np.ones((3,3))

    return {
        'mask' : {'left':{'emg':leftEMGMask, 'angle':leftAngleMask ,'kinetic':leftMomentMask } , 
                  'right': {'emg':rightEMGMask, 'angle':rightAngleMask,'kinetic':rightMomentMask}},
        'walk': {
            'right': {
                'emg': Right_patientEMG,        # List of arrays: each (trial*2, 4, 1001)
                'angle': Right_patientJointAngle,  # List of arrays: each (trial*2, 9, 1001)
                'kinetic': Right_patientTorque,    # List of arrays: each (trial*2, 9, 1001)
            },
            'left': {
                'emg': Left_patientEMG,
                'angle': Left_patientJointAngle,
                'kinetic': Left_patientTorque,
            }
        }
    }

def parseSIAT(currPath = "C:/EMG/datasets/SIAT_LLMD20230404/SIAT_LLMD20230404"):

    #index by patient: data[i] index by trial data[i][j]

    merged=['sEMG: upper tibialis anterior',
        'sEMG: lower tibialis anterior']

    emgs = [0, 'sEMG: rectus femoris',  'sEMG: vastus medialis', '(sEMG tibialis anterior merged)', 0, 'sEMG: semimembranosus',   'sEMG: medial gastrocnemius', 'sEMG: lateral gastrocnemius',  'sEMG: soleus', 0, 0,0, 0]     

    # List of Kinetic column headers (LEFT leg)
    kinetics = [
        'Kinetic: left hip adduction torque',
        'Kinetic: left hip flexion torque',
        'Kinetic: left knee flexion torque',
        'Kinetic: left ankle flexion torque',
    ]

    # List of Joint Angle (Kinematic) column headers (LEFT leg)
    angles = [
        'Kinematic: left hip adduction angle',
        'Kinematic: left hip flexion angle',
        'Kinematic: left knee flexion angle',
        'Kinematic: left ankle flexion angle',
    ]
    
    activities = ['WAK','UPS', 'DNS']  # 'STC' 'STDUP', 'SITDN', left out (static, stand up, sit down)
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    # Initialize patient-level storage
    PatientWalkEMG = []
    PatientWalkKinetic = []
    PatientWalkAngle = []
    
    PatientStairUpEMG = []
    PatientStairUpKinetic = []
    PatientStairUpAngle = []
    
    PatientStairDownEMG = []
    PatientStairDownKinetic = []
    PatientStairDownAngle = []
    
    PatientStandUpEMG = []
    PatientStandUpKinetic = []
    PatientStandUpAngle = []
    
    PatientSitDownEMG = []
    PatientSitDownKinetic = []
    PatientSitDownAngle = []
    
    for subject in os.listdir(currPath):
        print(subject)
        
        currPatientWalkEMG = []
        currPatientWalkKinetic = []
        currPatientWalkAngle = []
        
        currPatientStairUpEMG = []
        currPatientStairUpKinetic = []
        currPatientStairUpAngle = []
        
        currPatientStairDownEMG = []
        currPatientStairDownKinetic = []
        currPatientStairDownAngle = []
        
        currPatientStandUpEMG = []
        currPatientStandUpKinetic = []
        currPatientStandUpAngle = []
        
        currPatientSitDownEMG = []
        currPatientSitDownKinetic = []
        currPatientSitDownAngle = []

        ##EMG NORMALIZATION

        maxEMGs = [0] * len(emgs)

        for num, activity in enumerate(activities):
            dataPath = f"{currPath}/{subject}/Data/{subject}_{activity}_Data.csv"
            labelPath = f"{currPath}/{subject}/Labels/{subject}_{activity}_Label.csv"
            dl = pd.read_csv(labelPath,header=0)
            df = pd.read_csv(dataPath, header=0)

            currEMG = [None] * len(emgs)

            for e, emg in enumerate(emgs):
                if emg == 0: continue
                if '(' in emg:
                    currEMG[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(df[merged[0]].values)))))+
                    np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(df[merged[1]].values))))))/2
                    maxEMGs[e]=max(np.percentile(currEMG[e],99.5),maxEMGs[e])

                else:
                    currEMG[e] = np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(df[emg].values))))) 
                    maxEMGs[e]=max(np.percentile(currEMG[e],99.5),maxEMGs[e])

        for num, activity in enumerate(activities):
            dataPath = f"{currPath}/{subject}/Data/{subject}_{activity}_Data.csv"
            labelPath = f"{currPath}/{subject}/Labels/{subject}_{activity}_Label.csv"
            dl = pd.read_csv(labelPath,header=0)
            df = pd.read_csv(dataPath, header=0)

            # Status: [1: Heel Strike, 2 :Maximum Stance Flexion, 
            # 3: Maximum Stance Extension, 4: Toe-Off, 5. Maximum Swing Flexion]
            # Separate Heel Strike to Heel Strike in the label file should be treated as diff trials
            # Groups: 1-5: right, 6-10: left

            leftHeelStrikeTimes = []

            for rowNum,row in dl.iterrows():
                if pd.isna(row['Status']) and row['Group'] >5:
                    continue

                elif row['Group'] >5 and dl.iloc[rowNum]['Status']==1 and dl.iloc[rowNum-1]['Status']!=1:
                    leftHeelStrikeTimes.append(row['Time'])
                
            cycleAngles = []
            cycleMoments = []
            cycleEMGs = []

            for leftNum, currLeftTime in enumerate(leftHeelStrikeTimes):

                if leftNum==len(leftHeelStrikeTimes)-1:
                    break

                time_diff_first = (df['Time'] - currLeftTime).abs()
                firstDex = time_diff_first.idxmin()
                
                time_diff_next = (df['Time'] - leftHeelStrikeTimes[leftNum+1]).abs()
                nextDex = time_diff_next.idxmin()
                
                # Optional: verify the match is close enough (within 0.01 seconds)
                if time_diff_first.min() > 0.01 or time_diff_next.min() > 0.01:
                    print('wut (chris pratt)')
                    continue  # Skip if times don't match closely enough
                
                cycle_length=nextDex-firstDex

                currEMG = np.zeros((len(emgs), cycle_length))

                currAngleFull = np.zeros((len(joints),len(axis) ,cycle_length))
                currMomentFull = np.zeros((len(joints),len(axis) ,cycle_length))

                for a, angle in enumerate(angles):
                    if 'hip' in angle.lower() and 'flexion' in angle.lower():
                        currAngleFull[0,-1] =  df.iloc[firstDex:nextDex][angle].values
                    
                    elif 'hip' in angle.lower() and 'adduction' in angle.lower():
                        currAngleFull[0,0] =  df.iloc[firstDex:nextDex][angle].values
            
                    elif 'knee' in angle.lower():
                        currAngleFull[1,-1] =  df.iloc[firstDex:nextDex][angle].values

                    elif 'ankle' in angle.lower():
                        currAngleFull[-1,-1] =  df.iloc[firstDex:nextDex][angle].values

                for k, kinetic in enumerate(kinetics):
                    if 'hip' in kinetic.lower() and 'flexion' in kinetic.lower():
                        currMomentFull[0,-1] =  df.iloc[firstDex:nextDex][kinetic].values
                    
                    elif 'hip' in kinetic.lower() and 'adduction' in kinetic.lower():
                        currMomentFull[0,0] =  df.iloc[firstDex:nextDex][kinetic].values
            
                    elif 'knee' in kinetic.lower():
                        currMomentFull[1,-1] =  df.iloc[firstDex:nextDex][kinetic].values

                    elif 'ankle' in kinetic.lower():
                        currMomentFull[-1,-1] =  df.iloc[firstDex:nextDex][kinetic].values

                for e, emg in enumerate(emgs):
                    if emg == 0: continue
                    if '(' in emg:
                        currEMG[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(df.iloc[firstDex:nextDex][merged[0]].values)))))+
                        np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(df.iloc[firstDex:nextDex][merged[1]].values))))))/2

                    else:
                        currEMG[e] = np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(df.iloc[firstDex:nextDex][emg].values)))))

                    if maxEMGs[e]>0:
                        currEMG[e]/=maxEMGs[e]

                cycleMoments.append(currMomentFull)
                cycleAngles.append(currAngleFull)
                cycleEMGs.append(currEMG)    
                    
            if num ==0:
                currPatientWalkKinetic.append(cycleMoments)
                currPatientWalkAngle.append(cycleAngles)
                currPatientWalkEMG.append(cycleEMGs)
            elif num==1:
                currPatientStairUpEMG.append(cycleEMGs)
                currPatientStairUpKinetic.append(cycleMoments)
                currPatientStairUpAngle.append(cycleAngles)
            elif num==2:
                currPatientStairDownEMG.append(cycleEMGs)
                currPatientStairDownKinetic.append(cycleMoments)
                currPatientStairDownAngle.append(cycleAngles)
                    
        assert len(currPatientStairDownEMG) == 1 and \
                len(currPatientWalkEMG) == 1 and \
                len(currPatientStairUpEMG) == 1 
                # len(currPatientStandUpEMG) == 1 and \
                # len(currPatientSitDownEMG) == 1 and \

        PatientWalkAngle.append(currPatientWalkAngle)
        PatientWalkEMG.append(currPatientWalkEMG)
        PatientWalkKinetic.append(currPatientWalkKinetic)
        
        PatientStairUpAngle.append(currPatientStairUpAngle)
        PatientStairUpEMG.append(currPatientStairUpEMG)
        PatientStairUpKinetic.append(currPatientStairUpKinetic)
        
        PatientStairDownAngle.append(currPatientStairDownAngle)
        PatientStairDownEMG.append(currPatientStairDownEMG)
        PatientStairDownKinetic.append(currPatientStairDownKinetic)
        
        # PatientStandUpAngle.append(currPatientStandUpAngle)
        # PatientStandUpEMG.append(currPatientStandUpEMG)
        # PatientStandUpKinetic.append(currPatientStandUpKinetic)
        
        # PatientSitDownAngle.append(currPatientSitDownAngle)
        # PatientSitDownEMG.append(currPatientSitDownEMG)
        # PatientSitDownKinetic.append(currPatientSitDownKinetic)

    kineticMask = np.zeros((len(joints),len(axis))) 
    kinematicMask = np.zeros((len(joints),len(axis))) 
    emgMask = [1 if x != 0 else 0 for x in emgs]
    kineticMask[0,-1] = 1
    kineticMask[0,0] = 1
    kineticMask[1:,-1] = 1
    kinematicMask[0,0] = 1
    kinematicMask[0,-1] = 1
    kinematicMask[1:,-1] = 1

    return {
        'masks': {'left': {'emg':emgMask,'angle':kinematicMask,'kinetic':kineticMask}},

        'walk': {
            'left': {'emg': PatientWalkEMG, 'angle': PatientWalkAngle, 'kinetic': PatientWalkKinetic}
        },
        'stair_up': {
            'left': {'emg': PatientStairUpEMG, 'angle': PatientStairUpAngle, 'kinetic': PatientStairUpKinetic}
        },
        'stair_down': {
            'left': {'emg': PatientStairDownEMG, 'angle': PatientStairDownAngle, 'kinetic': PatientStairDownKinetic}
        },
        # 'sit_to_stand': {
        #     'left': {'emg': PatientStandUpEMG, 'angle': PatientStandUpAngle, 'kinetic': PatientStandUpKinetic}
        # },
        # 'stand_to_sit': {
        #     'left': {'emg': PatientSitDownEMG, 'angle': PatientSitDownAngle, 'kinetic': PatientSitDownKinetic}
        # }
    }

def parseLencioni(currPath = "C:/EMG/datasets/Lencioni",desiredEMGFreq=1000):
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']
    
    angles = ["PelvisFlx","PelvisAdd","PelvisRot","HipFlx","HipAdd","HipRot","KneeFlx","KneeAdd","KneeRot","AnkleFlx","AnkleAdd","AnkleRot"]
    tru_emgs = {
        "Tibialis Anterior": 0,
        "Soleus": 1,
        "Gastrocnemius Medialis": 2,
        "Peroneus Longus": 3,
        "Rectus Femoris": 4,
        "Vastus Medialis": 5,
        "Biceps Femoris": 6,
        "Gluteus Maximus": 7
    }
    emgs = [0, 'Rectus Femoris', 'Vastus Medialis', 'Tibialis Anterior',  'Biceps Femoris', 0 ,'Gastrocnemius Medialis', 0, 'Soleus', 'Peroneus Longus', 0, 0,'Gluteus Maximus']
    kinetics = ["HipFlxMom","HipAddMom","HipRotMom","KneeFlxMom","KneeAddMom","KneeRotMom","AnkleFlxMom","AnkleAddMom","AnkleRotMom"]
    FPs = ['GRFF_X','GRFF_Y','GRFF_Z','Cop_X','Cop_Y','Cop_Z','Torque_X','Torque_Y','Torque_Z']

    tasks = ['Walking','StepUp','StepDown']
    #EMGs and kinetics are taken about the dominant patient leg
    #not including heel or toe walking
    patientWalkKinetic = []
    patientWalkAngle = []
    patientWalkGRF = []
    patientWalkEMG = []

    patientStairDownKinetic = []
    patientStairDownAngle = []
    patientStairDownGRF = []
    patientStairDownEMG = []

    patientStairUpKinetic = []
    patientStairUpAngle = []
    patientStairUpGRF = []
    patientStairUpEMG = []

    for patient in os.listdir(currPath):

        currWalkAngle = []
        currWalkMoment = []
        currWalkEMG = []
        currWalkGRF = []

        
        currStairUpAngle = []
        currStairUpMoment = []
        currStairUpEMG = []
        currStairUpGRF = []

        currStairDownAngle = []
        currStairDownMoment = []
        currStairDownEMG = []
        currStairDownGRF = [] 
        ##each patient has 42 samples
            ##each angle element is 12,101
            ##each emg is 8,921

        data=os.path.join(currPath,patient)
        eng.eval(f"data = load('{data}');", nargout=0)
        sizeR=eng.eval("size(data.s.Data)")
        emgFrequency = eng.eval('data.s.EMGFeq')
        kinFrequency = eng.eval('data.s.KinFreq')
        #normalization logic    
        maxEMG = [0] * 8
        for i in range(1,int(sizeR[0][1])+1):
            currEMG=np.array(eng.eval(f"data.s.Data({i}).EMG"))
            for j in range(currEMG.shape[0]):
                maxEMG[j]=max(np.percentile(currEMG[j],99.5),maxEMG[j])

        #normalization logic

        for i in range(1,int(sizeR[0][1])+1):
            currTask=eng.eval(f"data.s.Data({i}).Task").replace(" ","")

            EMGnorm = np.array(eng.eval(f"data.s.Data({i}).EMG"))
            #normalize and re index before saving!
            for j in range(EMGnorm.shape[0]):
                if maxEMG[j]>0:
                    EMGnorm[j]=(np.abs(EMGnorm[j]/maxEMG[j])).clip(max=1.0)
                else:
                    EMGnorm[j]=(np.abs(EMGnorm[j])).clip(max=1.0)

            if emgFrequency==1000:
                EMGcurr = np.zeros((len(emgs),EMGnorm.shape[1]))

            else:
                old_points=np.linspace(0,1,EMGnorm.shape[1])
                new_count=int(desiredEMGFreq*(EMGnorm.shape[1]/emgFrequency))
                new_points = np.linspace(0,1,new_count)
                EMGcurr = np.zeros((len(emgs),new_count))

            for j, currEMG in enumerate(emgs):
                if currEMG == 0: continue
                else: 
                    if emgFrequency==1000:
                        EMGcurr[j]=EMGnorm[tru_emgs[currEMG]]
                    else: 
                        interpolator=interp1d(old_points,EMGnorm[tru_emgs[currEMG]])
                        new_emg = interpolator(new_points)
                        EMGcurr[j]=new_emg

            EMGnorm = EMGcurr
            ##FIT KINEMATIC AND KINETIC
            kinematicFreq=np.array(eng.eval(f"size(data.s.Data({i}).Ang)"))
            kinematicFull = np.zeros((len(joints),len(axis),int(kinematicFreq[0][1])))
            kineticFreq = np.array(eng.eval(f"size(data.s.Data({i}).Ang)"))
            kineticFull = np.zeros((len(joints),len(axis),int(kineticFreq[0][1])))

            kinematicUnfiltered=np.array(eng.eval(f"data.s.Data({i}).Ang"))
            kineticUnfiltered=np.array(eng.eval(f"data.s.Data({i}).Mom"))

            for j in range(3,kinematicUnfiltered.shape[0]):
                if j <6: 
                    if 'flx' in angles[j].lower():
                        kinematicFull[0,-1]=kinematicUnfiltered[j]
                    
                    elif 'add' in angles[j].lower():
                        kinematicFull[0,0]=kinematicUnfiltered[j]

                    elif 'rot' in angles[j].lower():
                        kinematicFull[0,1]=kinematicUnfiltered[j]

                elif j<9:
                    if 'flx' in angles[j].lower():
                        kinematicFull[1,-1]=kinematicUnfiltered[j]
                    
                    elif 'add' in angles[j].lower():
                        kinematicFull[1,0]=kinematicUnfiltered[j]

                    elif 'rot' in angles[j].lower():
                        kinematicFull[1,1]=kinematicUnfiltered[j]
                
                else: 
                    if 'flx' in angles[j].lower():
                        kinematicFull[-1,-1]=kinematicUnfiltered[j]
                    
                    elif 'add' in angles[j].lower():
                        kinematicFull[-1,0]=kinematicUnfiltered[j]

                    elif 'rot' in angles[j].lower():
                        kinematicFull[-1,1]=kinematicUnfiltered[j]

            for j in range(kineticUnfiltered.shape[0]):
                if j <3: 
                    if 'flx' in kinetics[j].lower():
                        kineticFull[0,-1]=kineticUnfiltered[j]
                    
                    elif 'add' in kinetics[j].lower():
                        kineticFull[0,0]=kineticUnfiltered[j]

                    elif 'rot' in kinetics[j].lower():
                        kineticFull[0,1]=kineticUnfiltered[j]

                elif j<6:
                    if 'flx' in kinetics[j].lower():
                        kineticFull[1,-1]=kineticUnfiltered[j]
                    
                    elif 'add' in kinetics[j].lower():
                        kineticFull[1,0]=kineticUnfiltered[j]

                    elif 'rot' in kinetics[j].lower():
                        kineticFull[1,1]=kineticUnfiltered[j]
                
                else: 
                    if 'flx' in kinetics[j].lower():
                        kineticFull[-1,-1]=kineticUnfiltered[j]
                    
                    elif 'add' in kinetics[j].lower():
                        kineticFull[-1,0]=kineticUnfiltered[j]

                    elif 'rot' in kinetics[j].lower():
                        kineticFull[-1,1]=kineticUnfiltered[j]

            if currTask=='Walking':
                
                currWalkAngle.append(kinematicFull)
                currWalkGRF.append(np.array(eng.eval(f"data.s.Data({i}).Grf")))
                currWalkMoment.append(kineticFull)
                currWalkEMG.append(EMGnorm)

            elif currTask=='StepUp':
                
                currStairUpAngle.append(kinematicFull)
                currStairUpGRF.append(np.array(eng.eval(f"data.s.Data({i}).Grf")))
                currStairUpMoment.append(kineticFull)

                currStairUpEMG.append(EMGnorm)
            elif currTask =='StepDown':
                currStairDownAngle.append(kinematicFull)
                currStairDownGRF.append(np.array(eng.eval(f"data.s.Data({i}).Grf")))
                currStairDownMoment.append(kineticFull)
                currStairDownEMG.append(EMGnorm)
        
        patientWalkEMG.append(currWalkEMG)
        patientWalkAngle.append(currWalkAngle)
        patientWalkGRF.append(currWalkGRF)
        patientWalkKinetic.append(currWalkMoment)

        patientStairDownEMG.append(currStairDownEMG)
        patientStairDownAngle.append(currStairDownAngle)
        patientStairDownGRF.append(currStairDownGRF)
        patientStairDownKinetic.append(currStairDownMoment)

        patientStairUpEMG.append(currStairUpEMG)
        patientStairUpAngle.append(currStairUpAngle)
        patientStairUpGRF.append(currStairUpGRF)
        patientStairUpKinetic.append(currStairUpMoment)
    
    kineticMask = np.ones((len(joints),len(axis)))
    kinematicMask = np.ones((len(joints),len(axis)))
    emgMask = [1 if x != 0 else 0 for x in emgs]

    return {
        'mask': {'emg':emgMask,'kinetic':kineticMask,'angle':kinematicMask},
        'step up': {
            'emg':patientStairUpEMG,
            'kinetic':patientStairUpKinetic,
            'grf':patientStairUpGRF,
            'angle':patientStairUpAngle
        },

        'step down': {
            'emg':patientStairDownEMG,
            'kinetic':patientStairDownKinetic,
            'grf':patientStairDownGRF,
            'angle':patientStairDownAngle
        },

        'walk': {
            'emg':patientWalkEMG,
            'kinetic':patientWalkKinetic,
            'grf':patientWalkGRF,
            'angle':patientWalkAngle
        }
    }

def parseUCIrvine(patientPath = "C:/EMG/datasets/UC_Irvine"):
    #EMG: RF,BF, VM, ST
    UCIrvineEMGs = [0, 'RF', 'VM', 0, 'BF', 'ST', 0, 0, 0,0,0,0,0]
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    #angle Knee
    patientPath0 = patientPath + "/N_TXT"
    patientPath1 = patientPath + "/A_TXT"
    patientEMG = []
    patientJointAngle = []
    #dataset is 4 emg and 1 knee of abled and disabled 
    for patient0,patient1 in zip(sorted(os.listdir(patientPath0)), sorted(os.listdir(patientPath1))):
        if patient0[-5]=='r':

            data0=pd.read_csv(patientPath0+f"/{patient0}",skiprows=8,delim_whitespace=True)
            maxEMG = [0]*4
            currEMG = [0]*4

            for e in range(4):
                currEMG[e] = np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(data0.iloc[:,e].values),w0=60.0,fs=1000),fs=1000)))
                maxEMG[e] = max(maxEMG[e],np.percentile(currEMG[e],99.5))
            for e in range(4):
                if maxEMG[e]>0:
                    currEMG[e]=(currEMG[e]/maxEMG[e]).clip(max=1.0)
            fullEMG = np.zeros((len(UCIrvineEMGs),currEMG[0].shape[0]))
            fullAngle = np.zeros((len(joints),len(axis),len(data0)))
            
            fullAngle[1,-1]=np.array(data0.iloc[:,-1].values)
            fullEMG[1]=currEMG[0]
            fullEMG[4]=currEMG[1]
            fullEMG[2]=currEMG[2]
            fullEMG[5]=currEMG[3]
            
            patientEMG.append(fullEMG)
            
            patientJointAngle.append(fullAngle)

        if patient1[-5]=='r':

            data01=pd.read_csv(patientPath1+f"/{patient1}",skiprows=8,delim_whitespace=True)
            maxEMG = [0]*4

            currEMG = [0]*4

            for e in range(4):
                currEMG[e] = np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(data01.iloc[:,e].values),w0=60.0,fs=1000),fs=1000)))
                maxEMG[e] = max(maxEMG[e],np.percentile(currEMG[e],99.5))
            for e in range(4):
                if maxEMG[e]>0:
                    currEMG[e]=(currEMG[e]/maxEMG[e]).clip(max=1.0)
            fullEMG = np.zeros((len(UCIrvineEMGs),currEMG[0].shape[0]))
            fullAngle = np.zeros((len(joints),len(axis),len(data01)))
            
            fullAngle[1,-1]=np.array(data01.iloc[:,-1].values)
            fullEMG[1]=currEMG[0]
            fullEMG[4]=currEMG[1]
            fullEMG[2]=currEMG[2]
            fullEMG[5]=currEMG[3]
            
            patientEMG.append(fullEMG)
            
            patientJointAngle.append(fullAngle)
    
    kinematicMask = np.zeros((len(joints),len(axis))) 
    kinematicMask[1,-1]=1
    emgMask = [1 if x != 0 else 0 for x in UCIrvineEMGs]

    return {'mask':{'emg':emgMask,'angle':kinematicMask},
            'walk':{'emg':patientEMG,'angle':patientJointAngle}}

def parseCamargo(currPath = "C:/EMG/datasets/Camargo"):
    #NOTE STATIC DATA HERE
    #index by data[i] = patient data[i][j] = trial

    emgTypes = ["vastuslateralis",0,"vastusmedialis","tibialisanterior","bicepsfemoris",
            "semitendinosus","gastrocmed",0,"soleus",0,0,"gluteusmedius", 0]
    
    angles=[ 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r','mtp_angle_r']

    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    #force_plates = ["FP5_px","FP5_py","FP5_pz","FP5_moment_x","FP5_moment_y","FP5_moment_z"]

    #ONLY INCLUDING RIGHT due to emg availability

    kinetics = [
        "hip_flexion_r_moment",
        "hip_adduction_r_moment",
        "hip_rotation_r_moment",
        "knee_angle_r_moment",
        "ankle_angle_r_moment",
        "subtalar_angle_r_moment",
        "mtp_angle_r_moment",
    ]

    
    tasks = ["levelground","ramp","stair","treadmill"]

    patientRampEMG = []
    patientStairEMG = []
    patientWalkEMG = []
    patientRampAngle = []
    patientStairAngle = []
    patientWalkAngle = []

    patientWalkKinetic = []
    patientStairKinetic = []
    patientRampKinetic = []

    # patientWalkFP = []
    # patientStairFP = []
    # patientRampFP = []

    for patient in sorted(os.listdir(currPath)):

        currPatientRampEMG = []
        currPatientStairEMG = []
        currPatientWalkEMG = []
        currPatientRampAngle = []
        currPatientStairAngle = []
        currPatientWalkAngle = []

        currPatientWalkKinetic = []
        currPatientStairKinetic = []
        currPatientRampKinetic = []

        # currPatientWalkFP = []
        # currPatientRampFP = []
        # currPatientStairFP = []


        patientPath = f"{currPath}/{patient}"
        for folderCount,folder in enumerate(sorted(os.listdir(patientPath))):
            print(folder)
            if folderCount>0:
                continue
            else:
                #MVC Normalization
                EMGMax = [0] * len(emgTypes)

                for t,taskType in enumerate(os.listdir(f"{patientPath}/{folder}")):
                    if taskType in tasks:
                        taskPath = f"{patientPath}/{folder}/{taskType}"

                        for emg in os.listdir(f"{taskPath}/emg"):
                            emgPath = os.path.join(taskPath,"emg",emg)

                            eng.eval(f'emgdata = load("{emgPath}");',nargout=0)

                            created = False
                            #accumulate emg
                            for i,idName in enumerate(emgTypes):
                                if idName == 0: continue
                                idName = idName.replace(" ","")
                                    
                                EMGMax[i] = max(EMGMax[i],np.percentile(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(eng.eval(f"emgdata.data.{idName}")).flatten(),w0=60,fs=1000),fs=1000))),99.5))


                #MVC Normalization

                for t,taskType in enumerate(os.listdir(f"{patientPath}/{folder}")):
                    if taskType in tasks:
                        taskPath = f"{patientPath}/{folder}/{taskType}"

                        for emg,fp,ik,id,gc in zip(sorted(os.listdir(f"{taskPath}/emg")),
                                                 sorted(os.listdir(f"{taskPath}/fp")),
                                                 sorted(os.listdir(f"{taskPath}/ik")),
                                                 sorted(os.listdir(f"{taskPath}/id")),
                                                 sorted(os.listdir(f"{taskPath}/gcRight"))):
                            if emg == fp == ik == id == gc:
                                emgPath = os.path.join(taskPath,"emg",emg)
                                fpPath = os.path.join(taskPath,"fp",fp)
                                ikPath = os.path.join(taskPath,"ik",ik)
                                idPath = os.path.join(taskPath,"id",id)
                                gcPath = os.path.join(taskPath,"gcRight",gc)

                                eng.eval(f'emgdata = load("{emgPath}");',nargout=0)
                                eng.eval(f'fpdata = load("{fpPath}");',nargout=0)
                                eng.eval(f'ikdata = load("{ikPath}");',nargout=0)
                                eng.eval(f'iddata = load("{idPath}");',nargout=0)
                                eng.eval(f'gcdata = load("{gcPath}");',nargout=0)

                                created = False
                                
                                alone = True
                                fDex = None
                                eDex = None
                                startTime = []
                                endTime = []
                                gaitCyclePercents=np.array(eng.eval(f'gcdata.data.HeelStrike',nargout=1)).flatten()
                                for currIdx, percentValue in enumerate(gaitCyclePercents):
                                    if percentValue==100:
                                        if alone==True: continue
                                        else: 
                                            si=eng.eval(f'gcdata.data.Header({currIdx+1})',nargout=1)
                                            endTime.append((currIdx,si))
                                            ss = eng.eval(f'gcdata.data.Header({fDex+1})',nargout=1)
                                            startTime.append((fDex,ss))
                                            alone=True

                                    elif percentValue==0:
                                        fDex = currIdx
                                        alone=False
                                #accumulate emg
                                strideEMG = []
                                strideKinetic = []
                                strideKinematic = []
                                assert len(startTime)==len(endTime)
                                for timeBegin, timeEnd in zip(startTime,endTime):
                                    unnorm=np.array(eng.eval('emgdata.data.Header',nargout=1)).flatten()
                                    rowBegin = np.argmin(np.abs(unnorm - timeBegin[1]))
                                    rowEnd = np.argmin(np.abs(unnorm - timeEnd[1]))

                                    if rowEnd-rowBegin<500:
                                        print('woa woa woa')
                                        continue
                                                              
                                    val_begin = eng.eval(f'emgdata.data.Header({rowBegin+1})', nargout=1)
                                    if abs(val_begin - timeBegin[1]) >= .001:
                                        print(
                                            f"[FAIL] Header rowBegin mismatch:\n"
                                            f"  Index: {rowBegin+1}\n"
                                            f"  Expected: {timeBegin[1]}\n"
                                            f"  Got:      {val_begin}"
                                        )
                                        continue

                                    val_end = eng.eval(f'emgdata.data.Header({rowEnd+1})', nargout=1)
                                    if abs(val_end - timeEnd[1]) >= .001:
                                        print(
                                            f"[FAIL] Header rowEnd mismatch:\n"
                                            f"  Index: {rowEnd+1}\n"
                                            f"  Expected: {timeEnd[1]}\n"
                                            f"  Got:      {val_end}"
                                        )
                                        continue

                                    for i,idName in enumerate(emgTypes):
                                        if idName ==0: continue
                                        idName = idName.replace(" ","")
                                        
                                        if not created:
                                            size = eng.eval("size(emgdata.data)")
                                            currEMG=np.zeros((len(emgTypes),(rowEnd-rowBegin)))
                                            created=True

                                        if EMGMax[i]>0:
                                            currEMG[i] = (np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(eng.eval(f"emgdata.data.{idName}({rowBegin+1}:{rowEnd})",nargout=1)).flatten(),w0=60,fs=1000),fs=1000)))/EMGMax[i]).clip(max=1.0)
                                        else: 
                                            currEMG[i] = (np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(np.array(eng.eval(f"emgdata.data.({rowBegin+1}:{rowEnd})",nargout=1)).flatten(),w0=60,fs=1000),fs=1000)))).clip(max=1.0)

                                    created = False

                                    for i,ikName in enumerate(angles):
                                        ikName = ikName.replace(" ","")

                                        if not created:
                                            size=eng.eval(f"size(ikdata.data.{ikName})")
                                            currIK=np.zeros((len(angles),timeEnd[0]-timeBegin[0]))
                                            currIKFull = np.zeros((len(joints),len(joints),timeEnd[0]-timeBegin[0]))
                                            created=True
                                        currData=np.array(eng.eval(f"ikdata.data.{ikName}({timeBegin[0]+1}:{timeEnd[0]})",nargout=1)).flatten()
                                        if 'hip' in ikName.lower():
                                            if 'adduction' in ikName.lower():
                                                currIKFull[0,0]=currData

                                            elif 'rotation' in ikName.lower():
                                                currIKFull[0,1]=currData

                                            elif 'flexion' in ikName.lower():
                                                currIKFull[0,-1]=currData
                                        elif 'knee' in ikName.lower():
                                            currIKFull[1,-1]=currData

                                        elif 'ankle' in ikName.lower():
                                            currIKFull[-1,-1]=currData

                                    created = False

                                    #accumulate id
                                    for i,idName in enumerate(kinetics):
                                        idName = idName.replace(" ","")

                                        if not created:
                                            size = eng.eval("size(iddata.data)")
                                            currID=np.zeros((len(kinetics),timeEnd[0]-timeBegin[0]))
                                            currIDFull = np.zeros((len(joints),len(joints),timeEnd[0]-timeBegin[0]))

                                            created=True
                                        currData=np.array(eng.eval(f"iddata.data.{idName}({timeBegin[0]+1}:{timeEnd[0]})")).flatten()
                                        if 'hip' in idName.lower():
                                            if 'adduction' in idName.lower():
                                                currIDFull[0,0]=currData

                                            elif 'rotation' in idName.lower():
                                                currIDFull[0,1]=currData

                                            elif 'flexion' in idName.lower():
                                                currIDFull[0,-1]=currData
                                        elif 'knee' in idName.lower():
                                            currIDFull[1,-1]=currData

                                        elif 'ankle' in ikName.lower():
                                            currIDFull[-1,-1]=currData

                                    created = False

                                    # for i, fpName in enumerate(force_plates):
                                    #     fpName = fpName.replace(" ","")
                                        
                                    #     if not created:
                                    #         size = eng.eval("size(fpdata.data)")
                                    #         currFP=np.zeros(((len(force_plates),int(size[0][0]))))   
                                    #         created=True
                                    #     currFP[i]=np.array(eng.eval(f"fpdata.data.{fpName}")).flatten()
                                    currIK = currIKFull
                                    currID = currIDFull
                                    strideEMG.append(currEMG)
                                    strideKinematic.append(currIK)
                                    strideKinetic.append(currID)

                                if taskType == 'levelground' or taskType =='treadmill':
                                    currPatientWalkAngle.append(strideKinematic)
                                    currPatientWalkEMG.append(strideEMG)
                                    currPatientWalkKinetic.append(strideKinetic)
                                    #currPatientWalkFP.append(currFP)
                                    
        
                                elif taskType == 'ramp':
                                    
                                    currPatientRampAngle.append(strideKinematic)
                                    currPatientRampEMG.append(strideEMG)
                                    currPatientRampKinetic.append(strideKinetic)
                                    #currPatientRampFP.append(currFP)
                                
                                elif taskType == 'stair':
                                    
                                    currPatientStairAngle.append(strideKinematic)
                                    currPatientStairEMG.append(strideEMG)
                                    currPatientStairKinetic.append(strideKinetic)
                                    #currPatientStairFP.append(currFP)      
        
                            else:
                                print("error in parsing",emg,fp,ik,id)
        patientRampAngle.append(currPatientRampAngle)
        patientRampEMG.append(currPatientRampEMG)
        patientRampKinetic.append(currPatientRampKinetic)
        #patientRampFP.append(currPatientRampFP)
        
        patientStairAngle.append(currPatientStairAngle)
        patientStairEMG.append(currPatientStairEMG)
        patientStairKinetic.append(currPatientStairKinetic)
        #patientStairFP.append(currPatientStairFP)

        
        patientWalkAngle.append(currPatientWalkAngle)
        patientWalkEMG.append(currPatientWalkEMG)
        patientWalkKinetic.append(currPatientWalkKinetic)
        #patientWalkFP.append(currPatientWalkFP)
    
    EMGMask = [1 if x != 0 else 0 for x in emgTypes]
    KineticMask = np.zeros((len(joints),len(axis))) 
    KinematicMask = np.zeros((len(joints),len(axis)))
    KineticMask[0,:] = 1
    KineticMask[1:,-1] = 1 
    KinematicMask[0,:] = 1
    KinematicMask[1:,-1] = 1

        
    return {'mask':{"emg":EMGMask,"angle":KinematicMask,"kinetic":KineticMask},
        'right':{
            "walk": {
                "emg": patientWalkEMG,
                "angle": patientWalkAngle,
                "kinetic": patientWalkKinetic,
          #      "fp": patientWalkFP
            },
            "stair": {
                "emg": patientStairEMG,
                "angle": patientStairAngle,
                "kinetic": patientStairKinetic,
         #       "fp": patientStairFP
            },
            "ramp": {
                "emg": patientRampEMG,
                "angle": patientRampAngle,
                "kinetic": patientRampKinetic,
           #     "fp": patientRampFP
            },
        }}

def parseCriekinge(currPath = "C:/EMG/datasets/Criekinge",emgSampleHz=1000,originalEMGSampleHz=100):
    #for healthy, only taking right leg, for post-stroke taking only stroke side
    #^^Governed by variable segmentationType

    #current stroke data only includes examples where all sensors are present ie for RFnorm
    
    EMGsOriginal = ["GASnorm","RFnorm","VLnorm","BFnorm","STnorm","TAnorm","ERSnorm"]
    EMGs=["VLnorm", "RFnorm",0, "TAnorm", "BFnorm", "STnorm","GASnorm", 0,0,0,0,0,0 ]

    Forces = ["AnkleMoment","HipMoment","KneeMoment"]#"KneeForce","HipForce","AnkleForce",
    Angles = ["HipAngles","KneeAngles","AnkleAngles"]#"PelvisAngles","FootProgressAngles"
    
    # Right side data structures
    allEMGs_R = []
    allForces_R = []
    allAngles_R = []
    
    # Left side data structures
    allEMGs_L = []
    allForces_L = []
    allAngles_L = []
    
    # Stroke side data structures
    allEMGs_stroke = []
    allForces_stroke = []
    allAngles_stroke = []
    
    ranges = [139,50]

    for y,file in enumerate(sorted(os.listdir(currPath))):

        filePath=os.path.join(currPath,file)

        eng.eval(f'data = load("{filePath}")',nargout=0)

        for subject in range(1,int(ranges[y])):
            
            if y == 0:  # Healthy subjects - collect both R and L side
                segmentationChoices = ['RsideSegm_RsideData', 'LsideSegm_LsideData']
            else:  # Stroke subjects - only stroke side
                segmentationChoices = ['NsideSegm_NsideData']#,'PsideSegm_PsideData']

            for seg_idx, segmentationChoice in enumerate(segmentationChoices):
                for a,angle in enumerate(Angles):
                    currDataAllAxis = np.stack([np.array(eng.eval(f'data.Sub({subject}).{segmentationChoice}.{angle}.{var}', nargout=1)).T for var in ['y', 'z', 'x']], axis=-1)
                    
                    if a == 0 and subject == 13 and y ==1:
                        currPatientAngle = np.zeros((len(Angles),10,1001,3),dtype=float)

                    elif a ==0:
                        currPatientAngle = np.zeros(((len(Angles),) + currDataAllAxis.shape),dtype=float)

                    currPatientAngle[a]=currDataAllAxis[:10] if subject == 13 and y ==1 else currDataAllAxis
                currPatientAngle = np.transpose(currPatientAngle,(1,0,3,2))


                for f,force in enumerate(Forces):
                    currDataAllAxis = np.stack([np.array(eng.eval(f'data.Sub({subject}).{segmentationChoice}.{force}.{var}', nargout=1)).T for var in ['y', 'z', 'x']], axis=-1)
                    
                    if f == 0 and subject == 13 and y ==1:
                        currPatientForce = np.zeros((len(Forces),10,1001,3),dtype=float)

                    elif f == 0:
                        currPatientForce = np.zeros(((len(Forces),) + currDataAllAxis.shape),dtype=float)

                    currPatientForce[f]=currDataAllAxis[:10] if subject == 13 and y ==1 else currDataAllAxis
                #currPatientAngle = np.transpose(currPatientAngle,(1,0,3,2))
                currPatientForce = np.transpose(currPatientForce,(1,0,3,2))

                currDataAllAxis = np.array(eng.eval(f'data.Sub({subject}).{segmentationChoice}.{EMGs[0]}.n', nargout=1)).T
                strideEMGs = []
                print(currDataAllAxis.shape[0])
                
                for stride in range(int(currDataAllAxis.shape[0])):

                    if segmentationChoice == 'NsideSegm_NsideData':
                        timeStarts = eng.eval(f'data.Sub({subject}).events.N_ICstart({stride+1})',nargout=1)
                        timeEnds = eng.eval(f'data.Sub({subject}).events.N_ICstop({stride+1})',nargout=1)

                    elif segmentationChoice == 'RsideSegm_RsideData':
                        timeStarts = eng.eval(f'data.Sub({subject}).events.R_ICstart({stride+1})',nargout=1)
                        timeEnds = eng.eval(f'data.Sub({subject}).events.R_ICstop({stride+1})',nargout=1)

                    elif segmentationChoice == 'LsideSegm_LsideData':
                        timeStarts = eng.eval(f'data.Sub({subject}).events.L_ICstart({stride+1})',nargout=1)
                        timeEnds = eng.eval(f'data.Sub({subject}).events.L_ICstop({stride+1})',nargout=1)
                    
                    duration_frames = abs(timeEnds - timeStarts)
                    duration_seconds = duration_frames / originalEMGSampleHz  # 100 Hz
                    new_count = int(duration_seconds * emgSampleHz)
                   
                    for e,emg in enumerate(EMGs):
                        if emg == 0: continue

                        currDataAllAxis = np.array(eng.eval(f'data.Sub({subject}).{segmentationChoice}.{emg}.n(:,{int(stride)+1})', nargout=1)).flatten()
                    
                        old_indices = np.linspace(0, 1, 1001)
                        new_indices = np.linspace(0, 1, new_count)
                        interpolator = interp1d(old_indices, currDataAllAxis, kind='linear')
                        resampled_emg = interpolator(new_indices)

                        #8,1001s
                        if e == 0:

                            currPatientEMG = np.zeros((len(EMGs),new_count),dtype=float)

                        currPatientEMG[e]=resampled_emg
                    strideEMGs.append(currPatientEMG)

                assert len(strideEMGs) == currPatientForce.shape[0] == currPatientAngle.shape[0]
                        

                # Append to appropriate data structure based on y and segmentation choice
                if y == 0:  # Healthy subjects
                    if seg_idx == 0:  # Right side
                        allAngles_R.append(currPatientAngle)
                        allEMGs_R.append(strideEMGs)
                        allForces_R.append(currPatientForce)
                    else:  # Left side
                        allAngles_L.append(currPatientAngle)
                        allEMGs_L.append(strideEMGs)
                        allForces_L.append(currPatientForce)
                else:  # Stroke subjects
                    allAngles_stroke.append(currPatientAngle)
                    allEMGs_stroke.append(strideEMGs)
                    allForces_stroke.append(currPatientForce)
    angleMask = np.ones((3,3))   
    kineticMask = np.ones((3,3))         
      
    emgMask = [1 if x != 0 else 0 for x in EMGs]
    return {
        'mask': {'emg':emgMask,'angle':angleMask,'kinetics':kineticMask},
        "walk": {
            "right": {
                'emg': allEMGs_R,
                'angle': allAngles_R,
                'kinetics': allForces_R,
            },
            "left": {
                'emg': allEMGs_L,
                'angle': allAngles_L,
                'kinetics': allForces_L,
            },
            "stroke": {
                'emg': allEMGs_stroke,
                'angle': allAngles_stroke,
                'kinetics': allForces_stroke,
            }
        }
    }

def parseK2Muse(currPath = "C:/EMG/datasets/k2muse/ProcessedData"):
    #just right foot data here
    trialTypes=['LG_0','AS_5','AS_10','DS_10','DS_5']

    ListofAngles = ['HipAngles','KneeAngles','AnkleAngles']#,'RPelvisAngles','RFootProgressAngles'}
    ListofMoments = ['HipMoment','KneeMoment','AnkleMoment']#'GroundReactionMoment'

    OriginalListofREMGs = ['TA','MG','LG','SOL','RF','VLO','VMO','BF','SEM']
    OriginalListofLEMGs = ['TA','LG','RF','BF']

    ListofREMGs = ['VLO', 'RF', 'VMO', 'TA', 'BF','SEM', 'MG', 'LG', 'SOL',0,0,0,0]
    emg_index = {
        'VLO': 0,
        'RF': 1,
        'VMO': 2,
        'TA': 3,
        'BF': 4,
        'SEM': 5,
        'MG': 6,
        'LG': 7,
        'SOL': 8,
        0: 12 
    }
    ListofLEMGs =  [0,'RF', 0,'TA','BF',0,'LG',0,0,0,0,0,0]

    ListofForces = ['RGroundReactionForce']
    
    AllDownRampREMG = []
    AllDownRampLEMG = []
    AllDownRampLAngle = []
    AllDownRampRAngle = []
    AllDownRampLMoment = []
    AllDownRampRMoment = []
    AllUpRampREMG = []
    AllUpRampLEMG = []
    AllUpRampLAngle = []
    AllUpRampRAngle = []
    AllUpRampLMoment = []
    AllUpRampRMoment = []
    AllWalkREMG = []
    AllWalkLEMG = []
    AllWalkLAngle = []
    AllWalkRAngle = []
    AllWalkLMoment = []
    AllWalkRMoment = []
    for i,patient in enumerate(sorted(os.listdir(currPath))):
        patientPath=currPath+'/' +patient
        eng.eval(f"data = load('{patientPath}')",nargout=0)
        print('loaded',patientPath)

        fatigueTypes=eng.eval(f"fieldnames(data.{patient[:-4]}.NormalizedData)",nargout=1)

        ######################################################
        PatientDownRampREMG = []
        PatientDownRampLEMG = []
        PatientDownRampLAngle = []
        PatientDownRampRAngle = []
        PatientDownRampLMoment = []
        PatientDownRampRMoment = []
        PatientUpRampREMG = []
        PatientUpRampLEMG = []
        PatientUpRampLAngle = []
        PatientUpRampRAngle = []
        PatientUpRampLMoment = []
        PatientUpRampRMoment = []
        PatientWalkREMG = []
        PatientWalkLEMG = []
        PatientWalkLAngle = []
        PatientWalkRAngle = []
        PatientWalkLMoment = []
        PatientWalkRMoment = []

        #find patient MVC
        EMGMax = [0]*len(ListofREMGs)

        for f,fatigueType in enumerate(fatigueTypes):
            fatigueType = fatigueType.replace(" ","")
            trialCheck=eng.eval(f"fieldnames(data.{patient[:-4]}.NormalizedData.{fatigueType})",nargout=1)

            for j,trialType in enumerate(trialTypes):
                if trialType not in trialCheck:
                    continue

                trialNum=eng.eval(f"fieldnames(data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1)",nargout=1)
                trialType = trialType.replace(" ","")

                for k,currTrial in enumerate(trialNum):
                    #minTrialSize is the number of strides
                    currSize=eng.eval(f"size(data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.EMGData.Right)",nargout=1)

                    for m in range(int(currSize[0][0])):
                        #TODO pass this currRData in the EMG ALSO FIND MVC
                        currRData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.EMGData.Right{{{m+1}}}",nargout=1)).T
                        for e in range(currRData.shape[0]):
                            #2000 Hz
                            currRData[e] = np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currRData[e],fs=2000),fs=2000)))
                            EMGMax[emg_index[OriginalListofREMGs[e]]]=max(np.percentile(currRData[e],99.5),EMGMax[emg_index[OriginalListofREMGs[e]]])

        #find patient MVC

        for f,fatigueType in enumerate(fatigueTypes):
            fatigueType = fatigueType.replace(" ","")
            trialCheck=eng.eval(f"fieldnames(data.{patient[:-4]}.NormalizedData.{fatigueType})",nargout=1)

            for j,trialType in enumerate(trialTypes):
                if trialType not in trialCheck:
                    continue

                trialNum=eng.eval(f"fieldnames(data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1)",nargout=1)

                trialRAngle = []
                trialLAngle = []
                trialRMoment = []
                trialLMoment = []
                trialREMG = []
                trialLEMG = []
                
                for k,currTrial in enumerate(trialNum):
                    #FIND MINIMUM STRIDE FOR DATA HOMOGENEITY
                    minTrialSize = 100

                    currLength=np.array(eng.eval(f"size(data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.EMGData.Right)",nargout=1))
                    minTrialSize=min(int(currLength[0][0]),minTrialSize)
                    currLength=np.array(eng.eval(f"size(data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.EMGData.Left)",nargout=1))
                    minTrialSize=min(int(currLength[0][0]),minTrialSize)
                    
                    for l,currAngle in enumerate(ListofAngles):
                       currData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Angles.L{currAngle}.rsGaitData",nargout=1)).transpose(2,1,0)
                       minTrialSize=min(currData.shape[0],minTrialSize)
                       currData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Angles.R{currAngle}.rsGaitData",nargout=1)).transpose(2,1,0)
                       minTrialSize=min(currData.shape[0],minTrialSize)
                    for l,currMoment in enumerate(ListofMoments):
                       currData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Moments.L{currMoment}.rsGaitData",nargout=1)).transpose(2,1,0)
                       minTrialSize=min(currData.shape[0],minTrialSize)
                       currData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Moments.R{currMoment}.rsGaitData",nargout=1)).transpose(2,1,0)
                       minTrialSize=min(currData.shape[0],minTrialSize)

                    for currForce in ListofForces:
                        currData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Forces.{currForce}.rsGaitData",nargout=1)).transpose(2,1,0)
                        minTrialSize=min(currData.shape[0],minTrialSize)
                    #FIND MINIMUM STRIDE FOR DATA HOMOGENEITY

                    for a,currAngle in enumerate(ListofAngles):
                       #200x3x31
                       currAngle=currAngle.replace(" ","")
                       currRData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Angles.R{currAngle}.rsGaitData",nargout=1)).transpose(2,1,0)
                       currLData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Angles.L{currAngle}.rsGaitData",nargout=1)).transpose(2,1,0)

                       if currRData.shape[0]!=minTrialSize or currLData.shape[0]!=minTrialSize:
                           currLData = currLData[:minTrialSize]
                           currRData = currRData[:minTrialSize]

                       if a==0:
                            #assuming both the left and right data are the same, but problems may arise due to different hz!!
                            currPatientRAngles = np.zeros((len(ListofAngles),minTrialSize,currRData.shape[1],currRData.shape[2]))
                            currPatientLAngles = np.zeros((len(ListofAngles),minTrialSize,currRData.shape[1],currRData.shape[2]))

                       currPatientRAngles[a]=currRData
                       currPatientLAngles[a]=currLData
       
                    for l,currMoment in enumerate(ListofMoments):
                       currMoment=currMoment.replace(" ","")
                       currRData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Moments.R{currMoment}.rsGaitData",nargout=1)).transpose(2,1,0)
                       currLData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Moments.L{currMoment}.rsGaitData",nargout=1)).transpose(2,1,0)
                        #assuming both the left and right data are the same, but problems may arise due to different hz!!
                        #worst case we just have python lists~~

                       if currRData.shape[0]!=minTrialSize or currLData.shape[0]!=minTrialSize:
                           currRData = currRData[:minTrialSize]
                           currLData = currLData[:minTrialSize]

                       if l==0:
                            currPatientRMoments = np.zeros((len(ListofMoments),minTrialSize,currRData.shape[1],currRData.shape[2]))
                            currPatientLMoments = np.zeros((len(ListofMoments),minTrialSize,currRData.shape[1],currRData.shape[2]))
                           
                       currPatientRMoments[l]=currRData
                       currPatientLMoments[l]=currLData
                        
                    # for l,currForce in enumerate(ListofForces):
                    #    currForce=currForce.replace(" ","")
                    #    currData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.Forces.{currForce}.rsGaitData",nargout=1)).transpose(2,0,1)
                       
                    #    if currData.shape[0]!=minTrialSize:
                    #        currData = currData[:minTrialSize]

                    #    if not createdForces:
                    #         currPatientForces = np.zeros((len(fatigueTypes),len(trialNum),len(ListofForces))+currData.shape,dtype=float)
                           
                    #         createdForces=True
                    #    currPatientForces[f,k,l]=currData
    
                    #minTrialSize is the number of strides
                    CycleREMG = []
                    CycleLEMG = []
                    CycleRMoment = []
                    CycleLMoment = []
                    CycleRAngle = []
                    CycleLAngle = []

                    currPatientLAngles=currPatientLAngles.transpose(1,0,2,3)
                    currPatientLMoments=currPatientLMoments.transpose(1,0,2,3)
                    currPatientRAngles=currPatientRAngles.transpose(1,0,2,3)
                    currPatientRMoments=currPatientRMoments.transpose(1,0,2,3)
                    #coordinate and shape conversion
                    #assumed stored order of y,x,z
                    tempXRMom = currPatientRMoments[:,:,1,:]
                    tempYRMom = currPatientRMoments[:,:,0,:]
                    tempZRMom = currPatientRMoments[:,:,2,:]

                    currPatientRMoments[:,:,0,:] = tempXRMom
                    currPatientRMoments[:,:,1,:] = tempZRMom
                    currPatientRMoments[:,:,2,:] = tempYRMom

                    tempXLMom = currPatientLMoments[:,:,1,:]
                    tempYLMom = currPatientLMoments[:,:,0,:]
                    tempZLMom = currPatientLMoments[:,:,2,:]

                    currPatientLMoments[:,:,0,:] = tempXLMom
                    currPatientLMoments[:,:,1,:] = tempZLMom
                    currPatientLMoments[:,:,2,:] = tempYLMom

                    tempXLMom = currPatientLAngles[:,:,1,:]
                    tempYLMom = currPatientLAngles[:,:,0,:]
                    tempZLMom = currPatientLAngles[:,:,2,:]
                    print(currPatientLAngles[:,:,1,:].shape[0])

                    currPatientLAngles[:,:,0,:] = tempXLMom
                    currPatientLAngles[:,:,1,:] = tempZLMom
                    currPatientLAngles[:,:,2,:] = tempYLMom

                    tempXRMom = currPatientRAngles[:,:,1,:]
                    tempYRMom = currPatientRAngles[:,:,0,:]
                    tempZRMom = currPatientRAngles[:,:,2,:]

                    currPatientRAngles[:,:,0,:] = tempXRMom
                    currPatientRAngles[:,:,1,:] = tempZRMom
                    currPatientRAngles[:,:,2,:] = tempYRMom

                    for m in range(minTrialSize):
                        currRData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.EMGData.Right{{{m+1}}}",nargout=1)).T
                        currLData=np.array(eng.eval(f"data.{patient[:-4]}.NormalizedData.{fatigueType}.{trialType}.S_1.{currTrial}.EMGData.Left{{{m+1}}}",nargout=1)).T
                        EMGR = np.zeros((len(ListofREMGs),currRData.shape[1])) 

                        for e in range(int(currRData.shape[0])):
                            if EMGMax[emg_index[OriginalListofREMGs[e]]]>0:
                                EMGR[emg_index[OriginalListofREMGs[e]]]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currRData[e],fs=2000),fs=2000)))/EMGMax[emg_index[OriginalListofREMGs[e]]]).clip(max=1.0)
                            else: EMGR[emg_index[OriginalListofREMGs[e]]]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currRData[e],fs=2000),fs=2000)))).clip(max=1.0)
                        CycleREMG.append(EMGR)
                        CycleLEMG.append(currLData)

                        CycleLAngle.append(currPatientLAngles[m])
                        CycleRAngle.append(currPatientRAngles[m])
                        CycleRMoment.append(currPatientRMoments[m])
                        CycleLMoment.append(currPatientLMoments[m])
                        
                    trialREMG.append(CycleREMG)
                    trialLEMG.append(CycleLEMG)
                    trialRAngle.append(CycleRAngle)
                    trialLAngle.append(CycleLAngle)
                    trialRMoment.append(CycleRMoment)
                    trialLMoment.append(CycleLMoment)                    
                
                if j==0:
                    PatientWalkLAngle.append(trialLAngle)
                    PatientWalkRAngle.append(trialRAngle)
                    PatientWalkLMoment.append(trialLMoment)
                    PatientWalkRMoment.append(trialRMoment)
                    PatientWalkLEMG.append(trialLEMG)
                    PatientWalkREMG.append(trialREMG)
                if j==1 or j==2:
                    PatientUpRampLAngle.append(trialLAngle)
                    PatientUpRampRAngle.append(trialRAngle)
                    PatientUpRampLMoment.append(trialLMoment)
                    PatientUpRampRMoment.append(trialRMoment)
                    PatientUpRampLEMG.append(trialLEMG)
                    PatientUpRampREMG.append(trialREMG)

                if j==3 or j==4:
                    PatientDownRampLAngle.append(trialLAngle)
                    PatientDownRampRAngle.append(trialRAngle)
                    PatientDownRampLMoment.append(trialLMoment)
                    PatientDownRampRMoment.append(trialRMoment)
                    PatientDownRampLEMG.append(trialLEMG)
                    PatientDownRampREMG.append(trialREMG)

        ####################################################################################################

        #AllDataTypeIndex = [patient,trial,cycleNum,hz]

        # Walking Data
        AllWalkLAngle.append(PatientWalkLAngle)
        AllWalkRAngle.append(PatientWalkRAngle)
        AllWalkLMoment.append(PatientWalkLMoment)
        AllWalkRMoment.append(PatientWalkRMoment)
        AllWalkLEMG.append(PatientWalkLEMG)
        AllWalkREMG.append(PatientWalkREMG)

        # Up Ramp Data
        AllUpRampLAngle.append(PatientUpRampLAngle)
        AllUpRampRAngle.append(PatientUpRampRAngle)
        AllUpRampLMoment.append(PatientUpRampLMoment)
        AllUpRampRMoment.append(PatientUpRampRMoment)
        AllUpRampLEMG.append(PatientUpRampLEMG)
        AllUpRampREMG.append(PatientUpRampREMG)

        # Down Ramp Data
        AllDownRampLAngle.append(PatientDownRampLAngle)
        AllDownRampRAngle.append(PatientDownRampRAngle)
        AllDownRampLMoment.append(PatientDownRampLMoment)
        AllDownRampRMoment.append(PatientDownRampRMoment)
        AllDownRampLEMG.append(PatientDownRampLEMG)
        AllDownRampREMG.append(PatientDownRampREMG)

        print('onto the next patient')

    jointMask = np.ones((3,3))
    emgMask=[1 if x != 0 else 0 for x in ListofREMGs]

    return {
        'mask': {'right': {'emg':emgMask,'angle':jointMask,'kinetic':jointMask}},
        'right': {
            "walk": {
                "emg": AllWalkREMG,
                "angle": AllWalkRAngle,
                "kinetic": AllWalkRMoment
            },
            "up_ramp": {
                "emg": AllUpRampREMG,
                "angle": AllUpRampRAngle,
                "kinetic": AllUpRampRMoment
            },
            "down_ramp": {
                "emg": AllDownRampREMG,
                "angle": AllDownRampRAngle,
                "kinetic": AllDownRampRMoment
            }
        }}
        #not including left because the emg was recorded in ultrasound not mvc through aSEMG
        # 'left': {
        #     "walk": {
        #         "emg": AllWalkLEMG,
        #         "angle": AllWalkLAngle,
        #         "kinetic": AllWalkLMoment
        #     },
        #     "up_ramp": {
        #         "emg": AllUpRampLEMG,
        #         "angle": AllUpRampLAngle,
        #         "kinetic": AllUpRampLMoment
        #     },
        #     "down_ramp": {
        #         "emg": AllDownRampLEMG,
        #         "angle": AllDownRampLAngle,
        #         "kinetic": AllDownRampLMoment
        #     }
        # }   

def parseBacek(currPath = "C:/EMG/datasets/Bacek"):
    # Treadmill 
    dont = {'MetabolicResting', 'NormFactors', 'Pref_End', 'Pref_Start'}
    
    # Right leg sensors
    OriginalRightEMGs = ['Right_BicFem', 'Right_GastroLat', 'Right_GastroMed', 'Right_GlutMax', 
                 'Right_RecFem', 'Right_Semitend', 'Right_TibAnt', 'Right_VastLat']
    RightAngles = ['Hip_Right', 'Knee_Right', 'Ankle_Right']#Pelvis as well
    RightGRFs = ['LM_Right', 'AP_Right', 'V_Right']

    joints = ['hip','knee','ankle']
    axis = ['roll', 'yaw', 'pitch']
    
    # Left leg sensors
    OriginalLeftEMGs = ['Left_BicFem', 'Left_GastroLat', 'Left_GastroMed', 'Left_GlutMax', 
                'Left_RecFem', 'Left_Semitend', 'Left_TibAnt', 'Left_VastLat']

    LeftEMGs = ['Left_VastLat','Left_RecFem', 0, 'Left_TibAnt', 'Left_BicFem', 'Left_Semitend','Left_GastroMed', 'Left_GastroLat', 0,0,0,0,'Left_GlutMax']
    RightEMGs = ['Right_VastLat','Right_RecFem', 0, 'Right_TibAnt', 'Right_BicFem', 'Right_Semitend','Right_GastroMed', 'Right_GastroLat', 0,0,0,0,'Right_GlutMax']

    LeftAngles = ['Hip_Left', 'Knee_Left', 'Ankle_Left']
    LeftGRFs = ['LM_Left', 'AP_Left', 'V_Left']

    # Initialize storage for both legs
    Right_PatientEMG = []
    Right_PatientAngles = []
    Right_PatientGRFs = []
    
    Left_PatientEMG = []
    Left_PatientAngles = []
    Left_PatientGRFs = []

    for folder in sorted(os.listdir(currPath)):
        print(folder)

        for file in os.listdir(f"{currPath}/{folder}"):
                               
            if 'Energetics' in file:
                RightEMGMax = [0] * len(RightEMGs)
                LeftEMGMax = [0] * len(LeftEMGs)
                
                eng.eval(f"data = load('{currPath}/{folder}/{file}');", nargout=0)
                trials = sorted(eng.eval('fieldnames(data.segEnergetics)', nargout=1))
                energeticsFile = file[:-14]+'Mechanics.mat'
                print(energeticsFile)
                eng.eval(f"indexData = load('{currPath}/{folder}/{energeticsFile}')",nargout=0)
                Right_PatientEMGTrials = []
                Left_PatientEMGTrials = []
                
                for currTrial in trials:
                    if currTrial not in dont:
                        # Process RIGHT leg EMGs
                        
                        #MVC calculation
                        for i, currEMG in enumerate(RightEMGs):
                            if currEMG==0: continue
                            EMGcurr = np.array(eng.eval(f'data.segEnergetics.{currTrial}.EMG.Activity.{currEMG}', nargout=1))
                            RightEMGMax[i] = max(np.abs(np.percentile(apply_wavelet_denoising(EMGcurr),99.5)),RightEMGMax[i])
                        
                        for i, currEMG in enumerate(LeftEMGs):
                            if currEMG==0: continue
                            EMGcurr = np.array(eng.eval(f'data.segEnergetics.{currTrial}.EMG.Activity.{currEMG}', nargout=1))
                            LeftEMGMax[i] = max(np.abs(np.percentile(apply_wavelet_denoising(EMGcurr),99.5)),LeftEMGMax[i])

                        #MVC calculation
                for currTrial in trials:
                    if currTrial not in dont:

                        dummySize = eng.eval(f'size(data.segEnergetics.{currTrial}.EMG.Activity.{currEMG})', nargout=1)
                        prevDexes=np.linspace(0,1,int(dummySize[0][1]))

                        strideCount = int(dummySize[0][0])
                        strideREMG = []
                        strideLEMG = []
                        strideRAngle = []
                        strideLAngle = []

                        for stride in range(strideCount):

                            for i, currEMG in enumerate(RightEMGs):
                                if currEMG==0: continue
                                if i == 0:
                                    indexCount = eng.eval(f'indexData.segMechanics.{currTrial}.GaitIndex.RHS_1kHz({stride+1})', nargout=1)
                                    nexDexCount = eng.eval(f'indexData.segMechanics.{currTrial}.GaitIndex.RHS_1kHz({stride+2})', nargout=1)
                                    indexCount = int(nexDexCount - indexCount)
                                    newDexes = np.linspace(0,1,indexCount)


                                    Right_EMGData = np.zeros((len(RightEMGs), indexCount))
                                EMGcurr = np.array(eng.eval(f'data.segEnergetics.{currTrial}.EMG.Activity.{currEMG}({stride+1},:)', nargout=1)).flatten()
                                interpolator = interp1d(prevDexes, EMGcurr, kind='linear')
                                resampled_emg = interpolator(newDexes)
                                
                                if RightEMGMax[i] > 0:
                                    Right_EMGData[i] = (np.abs(apply_wavelet_denoising(resampled_emg))/RightEMGMax[i]).clip(max=1.0)
                                else: 
                                    Right_EMGData[i] = (np.abs(apply_wavelet_denoising(resampled_emg))).clip(max=1.0)
                            strideREMG.append(Right_EMGData)

                        
                            # Process LEFT leg EMGs
                            for i, currEMG in enumerate(LeftEMGs):
                                if currEMG==0: continue
                                if i == 0:
                                    indexCount = eng.eval(f'indexData.segMechanics.{currTrial}.GaitIndex.LHS_1kHz({stride+1})', nargout=1)
                                    nexDexCount = eng.eval(f'indexData.segMechanics.{currTrial}.GaitIndex.LHS_1kHz({stride+2})', nargout=1)
                                    indexCount = int(nexDexCount - indexCount)
                                    newDexes = np.linspace(0,1,indexCount)

                                    Left_EMGData = np.zeros((len(LeftEMGs), indexCount))
                                EMGcurr = np.array(eng.eval(f'data.segEnergetics.{currTrial}.EMG.Activity.{currEMG}({stride+1},:)', nargout=1)).flatten()
                                interpolator = interp1d(prevDexes, EMGcurr, kind='linear')
                                resampled_emg = interpolator(newDexes)
                                if LeftEMGMax[i] > 0:
                                    Left_EMGData[i] = (np.abs(apply_wavelet_denoising(resampled_emg))/LeftEMGMax[i]).clip(max=1.0)
                                else: 
                                    Left_EMGData[i] = (np.abs(apply_wavelet_denoising(resampled_emg))).clip(max=1.0)
                            strideLEMG.append(Left_EMGData)
                        Left_PatientEMGTrials.append(strideLEMG)
                        Right_PatientEMGTrials.append(strideREMG)


            elif 'Mechanics' in file:
                eng.eval(f"data = load('{currPath}/{folder}/{file}');", nargout=0)
                trials = sorted(eng.eval(f"fieldnames(data.segMechanics)", nargout=1))

                Right_PatientAngleTrials = []
                Right_PatientGRFTrials = []
                Left_PatientAngleTrials = []
                Left_PatientGRFTrials = []
                
                for currTrial in trials:
                    if currTrial not in dont:
                        # Process RIGHT leg GRFs
                        for i, currGRF in enumerate(RightGRFs):
                            if i == 0:
                                size = eng.eval(f"size(data.segMechanics.{currTrial}.GRF.{currGRF})", nargout=1)
                                Right_currTrialGRFs = np.zeros((len(RightGRFs), int(size[0][0]), int(size[0][1])))
                            Right_currTrialGRFs[i] = np.array(eng.eval(f"data.segMechanics.{currTrial}.GRF.{currGRF}", nargout=1))

                        # Process LEFT leg GRFs
                        for i, currGRF in enumerate(LeftGRFs):
                            if i == 0:
                                size = eng.eval(f"size(data.segMechanics.{currTrial}.GRF.{currGRF})", nargout=1)
                                Left_currTrialGRFs = np.zeros((len(LeftGRFs), int(size[0][0]), int(size[0][1])))
                            Left_currTrialGRFs[i] = np.array(eng.eval(f"data.segMechanics.{currTrial}.GRF.{currGRF}", nargout=1))

                        # Process RIGHT leg Angles
                        for i, currAngle in enumerate(RightAngles):
                            if i == 0:
                                size = eng.eval(f"size(data.segMechanics.{currTrial}.Angles.{currAngle})", nargout=1)
                                Right_currTrialAngles = np.zeros((len(RightAngles), int(size[0][0]), int(size[0][1])))
                                Right_currTrialAnglesFull = np.zeros((len(joints), len(axis) ,int(size[0][0]), int(size[0][1])))
                            Right_currTrialAngles[i] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))
                            if 'hip' in currAngle.lower():
                                Right_currTrialAnglesFull[0,-1] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))
                            
                            elif 'knee' in currAngle.lower():
                                Right_currTrialAnglesFull[1,-1] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))

                            elif 'ankle' in currAngle.lower():
                                Right_currTrialAnglesFull[-1,-1] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))

                        # Process LEFT leg Angles
                        for i, currAngle in enumerate(LeftAngles):
                            if i == 0:
                                size = eng.eval(f"size(data.segMechanics.{currTrial}.Angles.{currAngle})", nargout=1)
                                Left_currTrialAngles = np.zeros((len(LeftAngles), int(size[0][0]), int(size[0][1])))
                                Left_currTrialAnglesFull = np.zeros((len(joints), len(axis) ,int(size[0][0]), int(size[0][1])))
                            if 'hip' in currAngle.lower():
                                Left_currTrialAnglesFull[0,-1] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))
                            
                            elif 'knee' in currAngle.lower():
                                Left_currTrialAnglesFull[1,-1] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))

                            elif 'ankle' in currAngle.lower():
                                Left_currTrialAnglesFull[-1,-1] = np.array(eng.eval(f"data.segMechanics.{currTrial}.Angles.{currAngle}", nargout=1))
                        Left_currTrialAnglesFull=Left_currTrialAnglesFull.transpose(2,0,1,3)
                        Right_currTrialAnglesFull=Right_currTrialAnglesFull.transpose(2,0,1,3)

                        Right_PatientAngleTrials.append(Right_currTrialAnglesFull)
                        Right_PatientGRFTrials.append(Right_currTrialGRFs)
                        Left_PatientAngleTrials.append(Left_currTrialAnglesFull)
                        Left_PatientGRFTrials.append(Left_currTrialGRFs)

        Right_PatientEMG.append(Right_PatientEMGTrials)
        Right_PatientGRFs.append(Right_PatientGRFTrials)
        Right_PatientAngles.append(Right_PatientAngleTrials)
        
        Left_PatientEMG.append(Left_PatientEMGTrials)
        Left_PatientGRFs.append(Left_PatientGRFTrials)
        Left_PatientAngles.append(Left_PatientAngleTrials)
    leftEMGMask = [1 if x != 0 else 0 for x in LeftEMGs]
    rightEMGMask = [1 if x != 0 else 0 for x in RightEMGs]
    leftAngleMask = np.zeros((len(joints),len(axis)))
    rightAngleMask = np.zeros((len(joints),len(axis)))
    leftAngleMask[:,-1] = 1
    rightAngleMask[:,-1] = 1

    
    return {
        'mask': {'left':{'emg':leftEMGMask,'angle':leftAngleMask},'right':{'emg':rightEMGMask,'angle':rightAngleMask}},
        'walk': {
            'right': {
                'emg': Right_PatientEMG,
                'angle': Right_PatientAngles,
                'grf': Right_PatientGRFs
            },
            'left': {
                'emg': Left_PatientEMG,
                'angle': Left_PatientAngles,
                'grf': Left_PatientGRFs
            }
        }
    }

def parseHu(currPath = "C:/EMG/datasets/Hu/data"):
    #patient data can be indexed with data[i] being the patient and data[i][j] being the trial
    
    #ID 0 = Sitting
    #ID 1 = LG Walking
    #ID 2 = Ramp Ascent
    #ID 3 = Ramp Descent
    #ID 4 = Stair Ascent
    #ID 5 = Stair Descent
    #ID 6 = Standing
    
    # Right leg sensors
    RightAngles = ['Right_Knee', 'Right_Ankle']

    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    OriginalRightEMGs = ['Right_TA', 'Right_MG', 'Right_SOL', 'Right_BF', 'Right_ST', 'Right_VL', 'Right_RF']
    
    # Left leg sensors
    LeftAngles = ['Left_Knee', 'Left_Ankle']
    OriginalLeftEMGs = ['Left_TA', 'Left_MG', 'Left_SOL', 'Left_BF', 'Left_ST', 'Left_VL', 'Left_RF']
    RightEMGs = 	['Right_VL','Right_RF',0,'Right_TA','Right_BF','Right_ST', 'Right_MG', 0, 'Right_SOL',0,0,0,0]
    LeftEMGs = 	['Left_VL','Left_RF',0,'Left_TA','Left_BF','Left_ST', 'Left_MG', 0, 'Left_SOL',0,0,0,0]

    # Initialize storage for all activity types - RIGHT leg
    Right_patientWalkEMG = []
    Right_patientWalkAngle = []
    Right_patientStairUpEMG = []
    Right_patientStairUpAngle = []
    Right_patientStairDownEMG = []
    Right_patientStairDownAngle = []
    Right_patientRampUpEMG = []
    Right_patientRampUpAngle = []
    Right_patientRampDownEMG = []
    Right_patientRampDownAngle = []
    # Right_patientSitToStandEMG = []
    # Right_patientSitToStandAngle = []
    # Right_patientStandToSitEMG = []
    # Right_patientStandToSitAngle = []
    
    # Initialize storage for all activity types - LEFT leg
    Left_patientWalkEMG = []
    Left_patientWalkAngle = []
    Left_patientStairUpEMG = []
    Left_patientStairUpAngle = []
    Left_patientStairDownEMG = []
    Left_patientStairDownAngle = []
    Left_patientRampUpEMG = []
    Left_patientRampUpAngle = []
    Left_patientRampDownEMG = []
    Left_patientRampDownAngle = []
    Left_patientSitToStandEMG = []
    Left_patientSitToStandAngle = []
    Left_patientStandToSitEMG = []
    Left_patientStandToSitAngle = []

    for folder in os.listdir(currPath):
        print('folder',folder)
        # Initialize trial storage for RIGHT leg
        Right_TrialWalkEMG = []
        Right_TrialWalkAngle = []
        Right_TrialStairUpEMG = []
        Right_TrialStairUpAngle = []
        Right_TrialStairDownEMG = []
        Right_TrialStairDownAngle = []
        Right_TrialRampUpEMG = []
        Right_TrialRampUpAngle = []
        Right_TrialRampDownEMG = []
        Right_TrialRampDownAngle = []
        # Right_TrialSitToStandEMG = []
        # Right_TrialSitToStandAngle = []
        # Right_TrialStandToSitEMG = []
        # Right_TrialStandToSitAngle = []
        
        # Initialize trial storage for LEFT leg
        Left_TrialWalkEMG = []
        Left_TrialWalkAngle = []
        Left_TrialStairUpEMG = []
        Left_TrialStairUpAngle = []
        Left_TrialStairDownEMG = []
        Left_TrialStairDownAngle = []
        Left_TrialRampUpEMG = []
        Left_TrialRampUpAngle = []
        Left_TrialRampDownEMG = []
        Left_TrialRampDownAngle = []
        # Left_TrialSitToStandEMG = []
        # Left_TrialSitToStandAngle = []
        # Left_TrialStandToSitEMG = []
        # Left_TrialStandToSitAngle = []

        #MVC Finding
        LeftEMGMax = [0] * len(LeftEMGs)
        RightEMGMax = [0] * len(RightEMGs)
        dataPath = f"{currPath}/{folder}/{folder}/Processed"
        for file in sorted(os.listdir(dataPath)):
            data = pd.read_csv(f"{dataPath}/{file}")
            modeSeen = None
            currModeStartIndex = 0
            
            for i, row in data.iterrows():
                if row['Mode'] != modeSeen:
                    if i == 0:
                        modeSeen = row['Mode']
                        currModeStartIndex = i
                    else:
                        
                        for e, currEMG in enumerate(RightEMGs):
                            if currEMG==0: continue
                            RightEMGMax[e]=max(np.percentile(np.abs(apply_wavelet_denoising(data[currModeStartIndex:i][currEMG].values)),99.5),RightEMGMax[e])
                        
                        for e, currEMG in enumerate(LeftEMGs):
                            if currEMG==0: continue
                            LeftEMGMax[e]=max(np.percentile(np.abs(apply_wavelet_denoising(data[currModeStartIndex:i][currEMG].values)),99.5),LeftEMGMax[e])
                        
                        modeSeen = row['Mode']
                        currModeStartIndex = i
            
            for e, currEMG in enumerate(RightEMGs):
                if currEMG == 0: continue
                RightEMGMax[e]=max(np.percentile(np.abs(apply_wavelet_denoising(data[currModeStartIndex:len(data)][currEMG].values)),99.5),RightEMGMax[e])
            
            for e, currEMG in enumerate(LeftEMGs):
                if currEMG ==0: continue
                LeftEMGMax[e]=max(np.percentile(np.abs(apply_wavelet_denoising(data[currModeStartIndex:len(data)][currEMG].values)),99.5),LeftEMGMax[e])

        #MVC finding^

        dataPath = f"{currPath}/{folder}/{folder}/Processed"
        for file in sorted(os.listdir(dataPath)):
            data = pd.read_csv(f"{dataPath}/{file}")
            modeSeen = None
            currModeStartIndex = 0

            ##find heel contact points
            rightContactPoints = []
            leftContactPoints = []
            for index, row in data.iterrows():

                # Check if both are NaN, then break
                if pd.isna(row['Right_Heel_Contact']) and pd.isna(row['Left_Heel_Contact']):
                    break
                
                # Add right contact point if not NaN
                if pd.notna(row['Right_Heel_Contact']):
                    rightContactPoints.append(int(row['Right_Heel_Contact'] - 1))
                
                # Add left contact point if not NaN
                if pd.notna(row['Left_Heel_Contact']):
                    leftContactPoints.append(int(row['Left_Heel_Contact'] - 1))     
            #ASSUMING LENGTH OF CONTACT POINTS IS ODD AND ARE GROUPED IN PAIRS OF TWO FOR A TRIAL

            for h,HS_R_Idx in enumerate(rightContactPoints):

                if h==len(rightContactPoints)-1:
                    break

                currModeSeen=data.iloc[HS_R_Idx]['Mode']
                nextModeSeen=data.iloc[rightContactPoints[h+1]]['Mode']

                if currModeSeen != nextModeSeen:
                    continue

                Nxt_HS_R_Idx =rightContactPoints[h+1]

                strideLength = Nxt_HS_R_Idx-HS_R_Idx
                HS_R_Idx = HS_R_Idx

                Right_currTrialEMG = np.zeros((len(RightEMGs), strideLength))
                Right_currTrialAngle = np.zeros((len(RightAngles), strideLength))
                Right_currTrialAngleFull = np.zeros((len(joints), len(axis),strideLength))

                for e, currEMG in enumerate(RightEMGs):
                    if currEMG==0: continue
                    Right_currTrialEMG[e] = ((np.abs(apply_wavelet_denoising(data.iloc[HS_R_Idx:Nxt_HS_R_Idx][currEMG].values)))/RightEMGMax[e]).clip(max=1.0)
                for a, currAngle in enumerate(RightAngles):
                    if 'knee' in currAngle.lower():
                        Right_currTrialAngleFull[1,-1]=data.iloc[HS_R_Idx:Nxt_HS_R_Idx][currAngle].values
                    if 'ankle' in currAngle.lower():    
                        Right_currTrialAngleFull[-1,-1]=data.iloc[HS_R_Idx:Nxt_HS_R_Idx][currAngle].values
                    Right_currTrialAngle[a] = data.iloc[HS_R_Idx:Nxt_HS_R_Idx][currAngle].values
                Right_currTrialAngle = Right_currTrialAngleFull

                if int(currModeSeen) == 1:  # Level ground walking
                    Right_TrialWalkAngle.append(Right_currTrialAngle)
                    Right_TrialWalkEMG.append(Right_currTrialEMG)

                elif int(currModeSeen) == 4:  # Stair Ascent
                    Right_TrialStairUpAngle.append(Right_currTrialAngle)
                    Right_TrialStairUpEMG.append(Right_currTrialEMG)
                
                elif int(currModeSeen) == 5:  # Stair Descent
                    Right_TrialStairDownAngle.append(Right_currTrialAngle)
                    Right_TrialStairDownEMG.append(Right_currTrialEMG)
                
                elif int(currModeSeen) == 2:  # Ramp Ascent
                    Right_TrialRampUpAngle.append(Right_currTrialAngle)
                    Right_TrialRampUpEMG.append(Right_currTrialEMG)
                
                elif int(currModeSeen) == 3:  # Ramp Descent
                    Right_TrialRampDownAngle.append(Right_currTrialAngle)
                    Right_TrialRampDownEMG.append(Right_currTrialEMG)

                # elif int(currModeSeen) == 0 and row['Mode'] == 6:  # Sit to Stand
                #     Right_TrialSitToStandAngle.append(Right_currTrialAngle)
                #     Right_TrialSitToStandEMG.append(Right_currTrialEMG)
                
                # elif int(currModeSeen) == 6 and row['Mode'] == 0:  # Stand to Sit
                #     Right_TrialStandToSitAngle.append(Right_currTrialAngle)
                #     Right_TrialStandToSitEMG.append(Right_currTrialEMG)

            for h,HS_L_Idx in enumerate(leftContactPoints):
                if h==len(leftContactPoints)-1:
                    break

                currModeSeen=data.iloc[HS_L_Idx]['Mode']
                nextModeSeen=data.iloc[leftContactPoints[h+1]]['Mode']

                if currModeSeen != nextModeSeen:
                    continue

                Nxt_HS_L_Idx =int(leftContactPoints[h+1])
                HS_L_Idx = int(HS_L_Idx)

                strideLength = int(Nxt_HS_L_Idx-HS_L_Idx)

                Left_currTrialEMG = np.zeros((len(LeftEMGs), strideLength))
                Left_currTrialAngle = np.zeros((len(LeftAngles), strideLength))
                Left_currTrialAngleFull = np.zeros((len(joints), len(axis),strideLength))

                for e, currEMG in enumerate(LeftEMGs):
                    if currEMG==0: continue
                    Left_currTrialEMG[e] = ((np.abs(apply_wavelet_denoising(data.iloc[HS_L_Idx:Nxt_HS_L_Idx][currEMG].values)))/LeftEMGMax[e]).clip(max=1.0)
                for a, currAngle in enumerate(LeftAngles):
                    if 'knee' in currAngle.lower():
                        Left_currTrialAngleFull[1,-1]=data.iloc[HS_L_Idx:Nxt_HS_L_Idx][currAngle].values
                    if 'ankle' in currAngle.lower():    
                        Left_currTrialAngleFull[-1,-1]=data.iloc[HS_L_Idx:Nxt_HS_L_Idx][currAngle].values
                    Left_currTrialAngle[a] = data.iloc[HS_L_Idx:Nxt_HS_L_Idx][currAngle].values
                Left_currTrialAngle = Left_currTrialAngleFull
        
                if int(currModeSeen) == 1:  # Level ground walking
                    Left_TrialWalkAngle.append(Left_currTrialAngle)
                    Left_TrialWalkEMG.append(Left_currTrialEMG)

                elif int(currModeSeen) == 4:  # Stair Ascent
                    Left_TrialStairUpAngle.append(Left_currTrialAngle)
                    Left_TrialStairUpEMG.append(Left_currTrialEMG)
                
                elif int(currModeSeen) == 5:  # Stair Descent
                    Left_TrialStairDownAngle.append(Left_currTrialAngle)
                    Left_TrialStairDownEMG.append(Left_currTrialEMG)
                
                elif int(currModeSeen) == 2:  # Ramp Ascent
                    Left_TrialRampUpAngle.append(Left_currTrialAngle)
                    Left_TrialRampUpEMG.append(Left_currTrialEMG)
                
                elif int(currModeSeen) == 3:  # Ramp Descent
                    Left_TrialRampDownAngle.append(Left_currTrialAngle)
                    Left_TrialRampDownEMG.append(Left_currTrialEMG)
                
            ##New code above

        Right_patientWalkEMG.append(Right_TrialWalkEMG)
        Right_patientWalkAngle.append(Right_TrialWalkAngle)
        Right_patientStairUpEMG.append(Right_TrialStairUpEMG)
        Right_patientStairUpAngle.append(Right_TrialStairUpAngle)
        Right_patientStairDownEMG.append(Right_TrialStairDownEMG)
        Right_patientStairDownAngle.append(Right_TrialStairDownAngle)
        Right_patientRampUpEMG.append(Right_TrialRampUpEMG)
        Right_patientRampUpAngle.append(Right_TrialRampUpAngle)
        Right_patientRampDownEMG.append(Right_TrialRampDownEMG)
        Right_patientRampDownAngle.append(Right_TrialRampDownAngle)
        # Right_patientSitToStandEMG.append(Right_TrialSitToStandEMG)
        # Right_patientSitToStandAngle.append(Right_TrialSitToStandAngle)
        # Right_patientStandToSitEMG.append(Right_TrialStandToSitEMG)
        # Right_patientStandToSitAngle.append(Right_TrialStandToSitAngle)
        
        # Append trial data to patient lists - LEFT leg
        Left_patientWalkEMG.append(Left_TrialWalkEMG)
        Left_patientWalkAngle.append(Left_TrialWalkAngle)
        Left_patientStairUpEMG.append(Left_TrialStairUpEMG)
        Left_patientStairUpAngle.append(Left_TrialStairUpAngle)
        Left_patientStairDownEMG.append(Left_TrialStairDownEMG)
        Left_patientStairDownAngle.append(Left_TrialStairDownAngle)
        Left_patientRampUpEMG.append(Left_TrialRampUpEMG)
        Left_patientRampUpAngle.append(Left_TrialRampUpAngle)
        Left_patientRampDownEMG.append(Left_TrialRampDownEMG)
        Left_patientRampDownAngle.append(Left_TrialRampDownAngle)
        # Left_patientSitToStandEMG.append(Left_TrialSitToStandEMG)
        # Left_patientSitToStandAngle.append(Left_TrialSitToStandAngle)
        # Left_patientStandToSitEMG.append(Left_TrialStandToSitEMG)
        # Left_patientStandToSitAngle.append(Left_TrialStandToSitAngle)

        leftAngleMask = np.zeros((len(joints),len(axis)))
        rightAngleMask = np.zeros((len(joints),len(axis)))
        leftAngleMask[1:,-1]= 1
        rightAngleMask[1:,-1]= 1
        LeftEMGMask = [1 if x != 0 else 0 for x in LeftEMGs]
        RightEMGMask = [1 if x != 0 else 0 for x in RightEMGs]

    return {
        'masks': {
            'left':{'emg':LeftEMGMask,'angles':leftAngleMask},
            'right':{'emg':RightEMGMask,'angles':rightAngleMask},
        },
        'walk': {
            'right': {'emg': Right_patientWalkEMG, 'angle': Right_patientWalkAngle},
            'left': {'emg': Left_patientWalkEMG, 'angle': Left_patientWalkAngle}
        },
        'ramp_up': {
            'right': {'emg': Right_patientRampUpEMG, 'angle': Right_patientRampUpAngle},
            'left': {'emg': Left_patientRampUpEMG, 'angle': Left_patientRampUpAngle}
        },
        'ramp_down': {
            'right': {'emg': Right_patientRampDownEMG, 'angle': Right_patientRampDownAngle},
            'left': {'emg': Left_patientRampDownEMG, 'angle': Left_patientRampDownAngle}
        },
        'stair_up': {
            'right': {'emg': Right_patientStairUpEMG, 'angle': Right_patientStairUpAngle},
            'left': {'emg': Left_patientStairUpEMG, 'angle': Left_patientStairUpAngle}
        },
        'stair_down': {
            'right': {'emg': Right_patientStairDownEMG, 'angle': Right_patientStairDownAngle},
            'left': {'emg': Left_patientStairDownEMG, 'angle': Left_patientStairDownAngle}
        },
        # 'sit_to_stand': {
        #     'right': {'emg': Right_patientSitToStandEMG, 'angle': Right_patientSitToStandAngle},
        #     'left': {'emg': Left_patientSitToStandEMG, 'angle': Left_patientSitToStandAngle}
        # },
        # 'stand_to_sit': {
        #     'right': {'emg': Right_patientStandToSitEMG, 'angle': Right_patientStandToSitAngle},
        #     'left': {'emg': Left_patientStandToSitEMG, 'angle': Left_patientStandToSitAngle}
        # }
    }

def processAngelidou(currPath = 'C:/EMG/datasets/Angelidou/data'):

    #sensors are also available
    #only grabbing right
    emgs=['RTA','RGA','RVL','RRF','RBF','LTA','LGA','LVL','LRF','LBF']
    right_emgs =['RTA','RGA','RVL','RRF','RBF']
    ##preprocessing data
    eng.eval('addpath("C:/EMG/datasets/Angelidou")',nargout=0)
    dont = ['participantInfo','gaitEvents','unit']
    ##each emg element is 300x1, after processing shape should be the same

    for penv,patient in enumerate(sorted(os.listdir(currPath))):
        if penv>8: continue
        eng.eval(f'data=load("{currPath}/{patient}")',nargout=0)
        speeds=eng.eval(f'fieldnames(data.{patient[:-4]})',nargout=1)
        for speed in speeds:
            speed=speed.replace(" ","")
            if speed not in dont:
                stiffness=eng.eval(f'fieldnames(data.{patient[:-4]}.{speed})',nargout=1)
                stiffness_arg = "{" + ", ".join([f"'{s}'"  for s in stiffness if s not in dont]) + "}"
                # for currStiffness in stiffness:
                #currStiffness=currStiffness.replace(" ","")
                #if currStiffness not in dont:
                eng.eval(f'currData=data.{patient[:-4]}.{speed}',nargout=0)
                for currEMG in emgs:
                    try:
                        if currEMG not in dont:
                            emg_arg = f"{{'{currEMG}'}}"
                            print(f"  Processing {patient[:-4]} / {speed} / {currEMG}...")
                            eng.eval(f"[~, timeseriesFullTrial] = timeseriesConvert(currData, {stiffness_arg}, {{'EMG'}}, {emg_arg});", nargout=0)
                            eng.eval(f"processed_data = processEMG(timeseriesFullTrial.EMG.{currEMG});", nargout=0)

                            frame_pointer = 1

                            for stiff in stiffness:
                                if stiff not in dont:

                                    # how many trials for this stiffness?
                                    num_trials = int(eng.eval(
                                        f"size(currData.{stiff}.EMG.{currEMG}, 1)"
                                    ))

                                    # initialize as empty cell array
                                    eng.eval(
                                        f"data.{patient[:-4]}.{speed}.{stiff}.EMG_Processed.{currEMG} = cell({num_trials}, 1);",
                                        nargout=0
                                    )

                                    for trial in range(num_trials):

                                        num_frames = int(eng.eval(
                                            f"size(currData.{stiff}.EMG.{currEMG}{{{trial+1}}}, 1)"
                                        ))

                                        eng.eval(
                                            f"segment = processed_data({frame_pointer}:{frame_pointer + num_frames - 1});",
                                            nargout=0
                                        )

                                        eng.eval(
                                            f"data.{patient[:-4]}.{speed}.{stiff}.EMG_Processed.{currEMG}{{{trial+1}}} = segment;",
                                            nargout=0
                                        )

                                        frame_pointer += num_frames

                    except Exception as e:
                        print(f"ERROR processing {patient[:-4]} / {speed} / {currEMG}: {e}")
                        continue
        eng.eval(f"save('C:/EMG/datasets/Angelidou/processedData/{patient[:-4]}Processed.mat', 'data','-v7.3')", nargout=0)                    #timeseriesConvert -> process EMG
            ##preprocessdata

def parseAngelidou(currPath='C:/EMG/datasets/Angelidou/processedData'):
    #data[i] = patient data[i][j] = trial data[i][j][k] = dataType data[i][j][k][l] = data
    Originalright_emgs = ['RTA', 'RGA', 'RVL', 'RRF', 'RBF']
    right_moments = ['RHip', 'RKnee', 'RAnkle']
    right_angles = ['RHip', 'RKnee', 'RAnkle']
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    Originalleft_emgs = ['LTA', 'LGA', 'LVL', 'LRF', 'LBF']
    left_moments = ['LHip', 'LKnee', 'LAnkle']
    left_angles = ['LHip', 'LKnee', 'LAnkle']

    right_emgs=["RVL","RRF", 0,"RTA","RBF", 0,"RGA",0,0,0,0,0,0]
    left_emgs=["LVL","LRF", 0,"LTA","LBF", 0,"LGA",0,0,0,0,0,0]

    dont = ['participantInfo', 'gaitEvents', 'unit']

    # Initialize storage for both legs
    right_totalEMGs = []
    right_totalAngles = []
    right_totalMoments = []
    
    left_totalEMGs = []
    left_totalAngles = []
    left_totalMoments = []

    for patient in sorted(os.listdir(currPath)):
        
        right_patientEMGs = []
        right_patientAngles = []
        right_patientMoments = []
        
        left_patientEMGs = []
        left_patientAngles = []
        left_patientMoments = []
        
        eng.eval(f'data=load("{currPath}/{patient}")', nargout=0)
        speeds = sorted(eng.eval(f'fieldnames(data.data.{patient[:5]})'))
        
        for speed in speeds:
            if speed not in dont:
                stiffnesses = sorted(eng.eval(f'fieldnames(data.data.{patient[:5]}.{speed})', nargout=1))
                
                for stiffness in stiffnesses:
                    if stiffness not in dont:
                        trialCount = eng.eval(f'size(data.data.{patient[:5]}.{speed}.{stiffness}.EMG_Processed.{right_emgs[0]})', nargout=1)
                        trialCount = int(trialCount[0][0])
                        
                        for t in range(trialCount):
                            # Process RIGHT leg EMGs
                            for e, emg in enumerate(right_emgs):
                                if emg==0: 
                                    continue
                                currData = np.array(eng.eval(f'data.data.{patient[:5]}.{speed}.{stiffness}.EMG_Processed.{emg}{{{t+1}}}', nargout=1)).flatten()
                                if e == 0:
                                    sampleCount = np.array(eng.eval(f'size(data.data.{patient[:5]}.{speed}.{stiffness}.EMG_Processed.{emg}{{{t+1}}})', nargout=1))
                                    right_currEmg = np.zeros((len(right_emgs),currData.shape[0]))

                                right_currEmg[e] = currData

                            # Process LEFT leg EMGs
                            for e, emg in enumerate(left_emgs):
                                if emg==0: 
                                    continue
                                currData = np.array(eng.eval(f'data.data.{patient[:5]}.{speed}.{stiffness}.EMG_Processed.{emg}{{{t+1}}}', nargout=1)).flatten()
                                if e == 0:
                                    sampleCount = np.array(eng.eval(f'size(data.data.{patient[:5]}.{speed}.{stiffness}.EMG_Processed.{emg}{{{t+1}}})', nargout=1))
                                    left_currEmg = np.zeros((len(left_emgs),currData.shape[0]))
                                left_currEmg[e] = currData

                            # Process RIGHT leg Angles
                            for a, angle in enumerate(right_angles):
                                currData = np.array(eng.eval(f'data.data.{patient[:5]}.{speed}.{stiffness}.jointAngles.{angle}{{{t+1}}}', nargout=1)).flatten()
                                if a == 0:
                                    sampleCount = np.array(eng.eval(f'size(data.data.{patient[:5]}.{speed}.{stiffness}.jointAngles.{angle}{{{t+1}}})', nargout=1))
                                    right_currAngle = np.zeros((len(right_angles),currData.shape[0]))
                                    right_currAngleFull = np.zeros((len(right_angles),3,currData.shape[0]))

                                if 'hip' in angle.lower():
                                    right_currAngleFull[0][-1] = currData

                                 
                                elif 'knee' in angle.lower():
                                    right_currAngleFull[1][-1] = currData

                                elif 'ankle' in angle.lower():
                                    right_currAngleFull[-1][-1] = currData

                                right_currAngle[a] = currData


                            # Process LEFT leg Angles
                            for a, angle in enumerate(left_angles):
                                currData = np.array(eng.eval(f'data.data.{patient[:5]}.{speed}.{stiffness}.jointAngles.{angle}{{{t+1}}}', nargout=1)).flatten()

                                if a == 0:
                                    sampleCount = np.array(eng.eval(f'size(data.data.{patient[:5]}.{speed}.{stiffness}.jointAngles.{angle}{{{t+1}}})', nargout=1))
                                    left_currAngle = np.zeros((len(left_angles),currData.shape[0]))
                                    left_currAngleFull =  np.zeros((len(left_angles),3,currData.shape[0]))

                                if 'hip' in angle.lower():
                                    left_currAngleFull[0][-1] = currData
                      
                                elif 'knee' in angle.lower():
                                    left_currAngleFull[1][-1] = currData
     
                                elif 'ankle' in angle.lower():
                                    left_currAngleFull[-1][-1] = currData

                                left_currAngle[a] = currData


                            # Process RIGHT leg Moments
                            for m, moment in enumerate(right_moments):
                                currData = np.array(eng.eval(f'data.data.{patient[:5]}.{speed}.{stiffness}.jointMoments.{moment}{{{t+1}}}', nargout=1)).flatten()
                                if m == 0:
                                    right_currMoment = np.zeros((len(right_moments),currData.shape[0]))
                                    right_currMomentFull =  np.zeros((len(right_moments),3,currData.shape[0]))

                                if 'hip' in moment.lower():
                                    right_currMomentFull[0][-1] = currData

                                elif 'knee' in moment.lower():
                                    right_currMomentFull[1][-1] = currData

                                elif 'ankle' in moment.lower():
                                    right_currMomentFull[-1][-1] = currData
   
                                right_currMoment[m] = currData

                            # Process LEFT leg Moments
                            for m, moment in enumerate(left_moments):
                                currData = np.array(eng.eval(f'data.data.{patient[:5]}.{speed}.{stiffness}.jointMoments.{moment}{{{t+1}}}', nargout=1)).flatten()

                                if m == 0:
                                    left_currMoment = np.zeros((len(left_moments),currData.shape[0]))
                                    left_currMomentFull =  np.zeros((len(left_moments),3,currData.shape[0]))

                                if 'hip' in moment.lower():
                                    left_currMomentFull[0][-1] = currData
                         
                                elif 'knee' in moment.lower():
                                    left_currMomentFull[1][-1] = currData
    

                                elif 'ankle' in moment.lower():
                                    left_currMomentFull[-1][-1] = currData
                                
                                left_currMoment[m] = currData
                        
                        # Append trial data for both legs
                            right_patientEMGs.append(right_currEmg)
                            right_patientAngles.append(right_currAngleFull)
                            right_patientMoments.append(right_currMomentFull)
                            
                            left_patientEMGs.append(left_currEmg)
                            left_patientAngles.append(left_currAngleFull)
                            left_patientMoments.append(left_currMomentFull)
        
        # Append patient data for both legs
        right_totalEMGs.append(right_patientEMGs)
        right_totalAngles.append(right_patientAngles)
        right_totalMoments.append(right_patientMoments)
        
        left_totalEMGs.append(left_patientEMGs)
        left_totalAngles.append(left_patientAngles)
        left_totalMoments.append(left_patientMoments)
    
    rightEMGMask = [1 if x != 0 else 0 for x in right_emgs]
    leftEMGMask = [1 if x != 0 else 0 for x in left_emgs]
    rightAngleMask = np.zeros((3,3))
    leftAngleMask = np.zeros((3,3))
    rightMomentMask = np.zeros((3,3))
    leftMomentMask = np.zeros((3,3))
    rightAngleMask[:,-1] = 1
    leftAngleMask[:,-1] = 1
    leftMomentMask[:,-1] = 1
    rightMomentMask[:,-1] = 1
    
    return {
        'mask': {'left':{'emg':leftEMGMask,'angle':leftAngleMask,'kinetic':leftMomentMask},
                 'right':{'emg':rightEMGMask,'angle':rightAngleMask,'kinetic':rightMomentMask}},
        'walk': {
            'right': {
                'emg': right_totalEMGs,
                'angle': right_totalAngles,
                'kinetic': right_totalMoments
            },
            'left': {
                'emg': left_totalEMGs,
                'angle': left_totalAngles,
                'kinetic': left_totalMoments
            }
        }
    }

def parseMoghadam(Path = 'C:/EMG/datasets/Moghadam/Surrogate_modelling_to_predict_gait_time_series-main/Gait_Time_Series'):
    # Right leg EMGs
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']

    OriginalRightEMGs = ['RTibialisAnterior','RSoleus','RGastrocnemiusMedialis','RBicepsFemoris',
                 'RSemitendinosus','RRectusFemoris','RVastusLateralis','RGluteusMaximus']
    
    # Left leg EMGs
    OriginalLeftEMGs = ['LTibialisAnterior','LSoleus','LGastrocnemiusMedialis','LBicepsFemoris',
                'LSemitendinosus','LRectusFemoris','LVastusLateralis','LGluteusMaximus']
    
    LeftEMGs = ['LVastusLateralis','LRectusFemoris',0,'LTibialisAnterior','LBicepsFemoris','LSemitendinosus', 'LGastrocnemiusMedialis',0,'LSoleus',0,0,0,"LGluteusMaximus"]
    RightEMGs = ['RVastusLateralis','RRectusFemoris',0,'RTibialisAnterior','RBicepsFemoris','RSemitendinosus', 'RGastrocnemiusMedialis',0,'RSoleus',0,0,0,"RGluteusMaximus"]

    LeftEMGDict = {
        'LTibialisAnterior': 3,
        'LSoleus': 8,
        'LGastrocnemiusMedialis': 6,
        'LBicepsFemoris': 4,
        'LSemitendinosus': 5,
        'LRectusFemoris': 1,
        'LVastusLateralis': 0,
        'LGluteusMaximus': 12
    }

    RightEMGDict = {
        'RTibialisAnterior': 3,
        'RSoleus': 8,
        'RGastrocnemiusMedialis': 6,
        'RBicepsFemoris': 4,
        'RSemitendinosus': 5,
        'RRectusFemoris': 1,
        'RVastusLateralis': 0,
        'RGluteusMaximus': 12
    }

    # Right leg kinematics
    right_kinematics = [
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", 
        "ankle_angle_r", "subtalar_angle_r", "mtp_angle_r"
    ]
    
    # Left leg kinematics
    left_kinematics = [
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l",
        "ankle_angle_l", "subtalar_angle_l", "mtp_angle_l"
    ]
    
    # Right leg kinetics
    right_kinetics = [
        "hip_flexion_r_moment", "hip_adduction_r_moment", "hip_rotation_r_moment",
        "knee_angle_r_moment", "ankle_angle_r_moment", "subtalar_angle_r_moment",
        "mtp_angle_r_moment"
    ]
    
    # Left leg kinetics
    left_kinetics = [
        "hip_flexion_l_moment", "hip_adduction_l_moment", "hip_rotation_l_moment",
        "knee_angle_l_moment", "ankle_angle_l_moment", "subtalar_angle_l_moment",
        "mtp_angle_l_moment"
    ]
    
    # Initialize storage for both legs
    right_patientTorques = []
    right_patientAngles = []
    right_patientEMGs = []
    
    left_patientTorques = []
    left_patientAngles = []
    left_patientEMGs = []
    
    for patient in sorted(os.listdir(Path)):
        currPath = f'{Path}/{patient}'
        
        right_trialTorques = []
        right_trialAngles = []
        right_trialEMGs = []
        
        left_trialTorques = []
        left_trialAngles = []
        left_trialEMGs = []

        maxEMGs_Right = [0] * len(RightEMGs)
        maxEMGs_Left  = [0] * len(LeftEMGs)
        #EMG NORM CALCULATION
        for EMG in sorted(os.listdir(currPath+'/EMG')):

            headCount = 0
            currEMG = pd.read_csv(
                f'{currPath}/EMG/{EMG}',
                sep=',',
                skiprows=headCount,
                engine='python'
            )
            skipDex=0
            for e, data in enumerate((RightEMGs)):
                if data==0:
                    skipDex+=1 
                    continue
                if data in currEMG.columns:
                    currEMGData = np.array(currEMG[data].iloc[:])

                else: 
                    currEMGData = np.array(currEMG.iloc[:, e+1-skipDex])

                maxEMGs_Right[RightEMGDict[OriginalRightEMGs[e-skipDex]]]=max(maxEMGs_Right[RightEMGDict[OriginalRightEMGs[e-skipDex]]],np.percentile(np.abs(currEMGData),99.5))
            # Process LEFT leg EMGs
            #bad, assumes that the order of the data is inline with the order of the full 
            #index data with skipDex, then access correct FullIndex with dictionary of corresponding skipDex index
            skipDex = 0
            for e, data in enumerate(LeftEMGs):
                if data==0: 
                    skipDex+=1
                    continue
                if data in currEMG.columns:
                    currEMGData = np.array(currEMG[data].iloc[:])

                else: 
                    currEMGData = np.array(currEMG.iloc[:, e+1+len(OriginalRightEMGs)-skipDex])

                maxEMGs_Left[LeftEMGDict[OriginalLeftEMGs[e-skipDex]]]=max(maxEMGs_Left[LeftEMGDict[OriginalLeftEMGs[e-skipDex]]],np.percentile(np.abs(currEMGData),99.5))
        #EMG NORM CALCULATION 
        
        for EMG, Kinematics, moments, IMUs in zip(sorted(os.listdir(currPath+'/EMG')),
                                sorted(os.listdir(currPath+'/Joints_Kinematics')),
                                sorted(os.listdir(currPath+'/Joints_Kinetics')),
                                sorted(os.listdir(currPath+'/IMU'))):
            assert EMG[:-4]==Kinematics[:-4]==moments[:-4]==IMUs[:-4]
            
            # Read EMG data
            headCount = 0
            currEMG = pd.read_csv(
                f'{currPath}/EMG/{EMG}',
                sep=',',
                skiprows=headCount,
                engine='python'
            )

            print(f'{currPath}/IMU/{IMUs}')
            if f'{currPath}/IMU/{IMUs}' == 'C:/EMG/datasets/Moghadam/Surrogate_modelling_to_predict_gait_time_series-main/Gait_Time_Series/P4Data/IMU/12.csv':
                continue
            elif f'{currPath}/IMU/{IMUs}' == 'C:/EMG/datasets/Moghadam/Surrogate_modelling_to_predict_gait_time_series-main/Gait_Time_Series/P4Data/IMU/13.csv':
                continue
            elif f'{currPath}/IMU/{IMUs}' == 'C:/EMG/datasets/Moghadam/Surrogate_modelling_to_predict_gait_time_series-main/Gait_Time_Series/P4Data/IMU/14.csv':
                continue
            elif f'{currPath}/IMU/{IMUs}' == 'C:/EMG/datasets/Moghadam/Surrogate_modelling_to_predict_gait_time_series-main/Gait_Time_Series/P4Data/IMU/15.csv':
                continue
            elif f'{currPath}/IMU/{IMUs}' == 'C:/EMG/datasets/Moghadam/Surrogate_modelling_to_predict_gait_time_series-main/Gait_Time_Series/P4Data/IMU/16.csv':
                continue

            currIMU = pd.read_csv(
                f'{currPath}/IMU/{IMUs}',
                sep=None,
                skiprows=headCount,
                engine='python'
            )
            currIMU.columns = currIMU.columns.str.strip()  # Add this

            # Read Kinematics data
            with open(f'{currPath}/Joints_Kinematics/{Kinematics}') as f:
                for line in f:
                    headCount += 1
                    if 'endheader' in line:
                        break
            
            currKinematic = pd.read_csv(
                f'{currPath}/Joints_Kinematics/{Kinematics}',
                sep=r'\s+',
                skiprows=headCount,
                engine='python'
            )

            headCount=0

            # Read Kinetics data
            with open(f'{currPath+'/Joints_Kinetics/'+moments}') as f:
                for line in f:
                    headCount += 1
                    if 'endheader' in line:
                        break
            currKinetic = pd.read_csv(
                f'{currPath}/Joints_Kinetics/{moments}',
                sep=r'\s+',
                skiprows=headCount,
                engine='python'
            )
            
            # Find time alignment
            data = [currEMG, currKinematic, currKinetic,currIMU]
            maxtime = 1e-9
            minTime = 1e9
            timeType = []
            for x, curr in enumerate(data):

                if 'time' in curr.columns: 
                    timeType.append('time')
                elif 'Time' in curr.columns: 
                    timeType.append('Time')
                else: 
                    input(f'2 failure,{curr.columns} {x}')
                maxtime = max(curr[timeType[x]].iloc[0], maxtime)
                minTime = min(curr[timeType[x]].iloc[-1], minTime)

            # Find the row that corresponds to that time
            row_indexE = (currEMG[timeType[0]] - maxtime).abs().idxmin()
            row_indexA = (currKinematic[timeType[1]] - maxtime).abs().idxmin()
            row_indexM = (currKinetic[timeType[2]] - maxtime).abs().idxmin()
            row_indexI = (currIMU[timeType[-1]] - maxtime).abs().idxmin()

            Mrow_indexE = (currEMG[timeType[0]] - minTime).abs().idxmin()
            Mrow_indexA = (currKinematic[timeType[1]] - minTime).abs().idxmin()
            Mrow_indexM = (currKinetic[timeType[2]] - minTime).abs().idxmin()
            Mrow_indexI = (currIMU[timeType[-1]] - minTime).abs().idxmin()

            assert currEMG[timeType[0]].iloc[row_indexE] == maxtime and \
                currKinematic[timeType[1]].iloc[row_indexA] == maxtime and \
                currKinetic[timeType[2]].iloc[row_indexM] == maxtime and \
                currEMG[timeType[0]].iloc[Mrow_indexE] >= minTime and \
                currKinematic[timeType[1]].iloc[Mrow_indexA] >= minTime and \
                currKinetic[timeType[2]].iloc[Mrow_indexM] >= minTime and \
                f"Time alignment failed:\n" \
                f"MaxTime - EMG={currEMG[timeType[0]].iloc[row_indexE]}, " \
                f"Kinematic={currKinematic[timeType[1]].iloc[row_indexA]}, " \
                f"Kinetic={currKinetic[timeType[2]].iloc[row_indexM]}, " \
                f"IMU={currIMU[timeType[-1]].iloc[row_indexI]}, Expected={maxtime}\n" \
                f"MinTime - EMG={currEMG[timeType[0]].iloc[Mrow_indexE]}, " \
                f"Kinematic={currKinematic[timeType[1]].iloc[Mrow_indexA]}, " \
                f"Kinetic={currKinetic[timeType[2]].iloc[Mrow_indexM]}, " \
                f"IMU={currIMU[timeType[-1]].iloc[Mrow_indexI]}, Expected={minTime}"
            ##find segmentations
            segmentIndices=detect_strides_vertical(currIMU.iloc[row_indexI:Mrow_indexI])
            # Add comprehensive diagnostics

            ##RIGHT^^
            stridesTorqueRight = []
            stridesTorqueLeft = []
            stridesAngleRight = []
            stridesAngleLeft = []
            stridesEMGRight = []
            stridesEMGLeft = []
            ##LEFT^^

            # Process RIGHT leg EMGs
            #return [stridesR, acc_filteredR, peaksR],[stridesL, acc_filteredL, peaksL]

            for strNum in range(len(segmentIndices[0][0])):
                firstTime = segmentIndices[0][0][strNum]['startTime']
                nextTime = segmentIndices[0][0][strNum]['endTime']
                strideRange = nextTime-firstTime

                time_diff_first = (currEMG[timeType[0]] - firstTime).abs()
                firstDex = time_diff_first.idxmin()                
                time_diff_next = (currEMG[timeType[0]] - nextTime).abs()
                nextDex = time_diff_next.idxmin()
                
                if time_diff_first.min() > 0.01 or time_diff_next.min() > 0.01:
                    print('wut (chris pratt)')
                    continue
                if firstTime < currEMG[timeType[0]].iloc[0] or nextTime > currEMG[timeType[0]].iloc[-1]:
                    print('wut (chris )')

                    continue
                if firstTime < currKinematic[timeType[1]].iloc[0] or nextTime > currKinematic[timeType[1]].iloc[-1]:
                    print('wut (chris )')

                    continue
                if firstTime < currKinetic[timeType[2]].iloc[0] or nextTime > currKinetic[timeType[2]].iloc[-1]:
                    print('wut (chris )')

                    continue
                skipDex = 0
                for e, data in enumerate(RightEMGs):
                    if data==0: 
                        skipDex+=1
                        continue
                    if data in currEMG.columns:
                        currEMGData = np.array(currEMG[data].iloc[firstDex:nextDex])

                    else: 
                        currEMGData = np.array(currEMG.iloc[firstDex:nextDex, e+1-skipDex])

                    if e == 0:
                        right_currEMGs = np.zeros((len(RightEMGs), currEMGData.shape[0]))
                    if maxEMGs_Right[RightEMGDict[OriginalRightEMGs[e-skipDex]]]>0:
                        right_currEMGs[RightEMGDict[OriginalRightEMGs[e-skipDex]]] = (np.abs((currEMGData/maxEMGs_Right[RightEMGDict[OriginalRightEMGs[e-skipDex]]]))).clip(max=1.0)
                    else: 
                        right_currEMGs[RightEMGDict[OriginalRightEMGs[e-skipDex]]] = (np.abs(currEMGData)).clip(max=1.0)

                time_diff_first = (currKinematic[timeType[1]] - firstTime).abs()
                firstDex = time_diff_first.idxmin()                
                time_diff_next = (currKinematic[timeType[1]] - nextTime).abs()
                nextDex = time_diff_next.idxmin()
                
                if time_diff_first.min() > 0.01 or time_diff_next.min() > 0.01:
                    print('wut (chris pratt)')
                    continue

                for a, data in enumerate(right_kinematics):
                    currAngle = np.array(currKinematic[data].iloc[firstDex:nextDex])
                    if a == 0:
                        right_currAngles = np.zeros((len(right_kinematics), nextDex-firstDex))
                        right_currAnglesFull = np.zeros((len(joints),len(axis),nextDex-firstDex))
                    if 'hip' in data.lower():
                        if 'adduction' in data.lower():
                            right_currAnglesFull[0,0] = currAngle
                        elif 'flexion' in data.lower():
                            right_currAnglesFull[0,-1] = currAngle
                        elif 'rotation' in data.lower():
                            right_currAnglesFull[0,1] = currAngle

                    elif 'knee' in data.lower():
                        right_currAnglesFull[1,-1] = currAngle
                    elif 'ankle' in data.lower():
                        right_currAnglesFull[-1,-1]=currAngle
                                    
                    right_currAngles[a] = currAngle

                time_diff_first = (currKinetic[timeType[2]] - firstTime).abs()
                firstDex = time_diff_first.idxmin()                
                time_diff_next = (currKinetic[timeType[2]] - nextTime).abs()
                nextDex = time_diff_next.idxmin()
                
                if time_diff_first.min() > 0.01 or time_diff_next.min() > 0.01:
                    print('wut (chris pratt)')
                    continue

                for k, data in enumerate(right_kinetics):
                    currTorque = np.array(currKinetic[data].iloc[firstDex:nextDex])
                    if k == 0:
                        right_currTorques = np.zeros((len(right_kinematics), nextDex-firstDex))
                        right_currTorquesFull = np.zeros((len(joints),len(axis),nextDex-firstDex))
                    if 'hip' in data.lower():
                        if 'adduction' in data.lower():
                            right_currTorquesFull[0,0] = currTorque
                        elif 'flexion' in data.lower():
                            right_currTorquesFull[0,-1] = currTorque
                        elif 'rotation' in data.lower():
                            right_currTorquesFull[0,1] = currTorque

                    elif 'knee' in data.lower():
                        right_currTorquesFull[1,-1] = currTorque
                    elif 'ankle' in data.lower():
                        right_currTorquesFull[-1,-1]=currTorque
                                    
                    right_currTorques[k] = currTorque

                stridesEMGRight.append(right_currEMGs)
                stridesTorqueRight.append(right_currTorquesFull)
                stridesAngleRight.append(right_currAnglesFull)
                
            for strNum in range(len(segmentIndices[1][0])):
                firstTime = segmentIndices[1][0][strNum]['startTime']
                nextTime = segmentIndices[1][0][strNum]['endTime']
                strideRange = nextTime-firstTime

                # Process LEFT leg EMGs
                time_diff_first = (currEMG[timeType[0]] - firstTime).abs()
                firstDex = time_diff_first.idxmin()                
                time_diff_next = (currEMG[timeType[0]] - nextTime).abs()
                nextDex = time_diff_next.idxmin()


                if time_diff_first.min() > 0.01 or time_diff_next.min() > 0.01:
                    print('wut (chris pratt)')
                    continue
                if firstTime < currEMG[timeType[0]].iloc[0] or nextTime > currEMG[timeType[0]].iloc[-1]:
                    print('wut (chris )')

                    continue
                if firstTime < currKinematic[timeType[1]].iloc[0] or nextTime > currKinematic[timeType[1]].iloc[-1]:
                    print('wut (chris )')

                    continue
                if firstTime < currKinetic[timeType[2]].iloc[0] or nextTime > currKinetic[timeType[2]].iloc[-1]:
                    print('wut (chris )')

                    continue

                skipDex = 0
                for e, data in enumerate(LeftEMGs):
                    if data==0:
                        skipDex+=1 
                        continue
                    if data in currEMG.columns:
                        currEMGData = np.array(currEMG[data].iloc[firstDex:nextDex])
                    else: 
                        currEMGData = np.array(currEMG.iloc[firstDex:nextDex, e+1+len(OriginalRightEMGs)-skipDex])

                    if e == 0:
                        left_currEMGs = np.zeros((len(LeftEMGs), currEMGData.shape[0]))
                    if maxEMGs_Left[LeftEMGDict[OriginalLeftEMGs[e-skipDex]]]>0:
                        left_currEMGs[LeftEMGDict[OriginalLeftEMGs[e-skipDex]]] = (np.abs((currEMGData/maxEMGs_Left[LeftEMGDict[OriginalLeftEMGs[e-skipDex]]]))).clip(max=1.0)
                    else:
                        left_currEMGs[LeftEMGDict[OriginalLeftEMGs[e-skipDex]]] = (np.abs(currEMGData)).clip(max=1.0)

                time_diff_first = (currKinematic[timeType[1]] - firstTime).abs()
                firstDex = time_diff_first.idxmin()                
                time_diff_next = (currKinematic[timeType[1]] - nextTime).abs()
                nextDex = time_diff_next.idxmin()
                
                if time_diff_first.min() > 0.01 or time_diff_next.min() > 0.01:
                    print('wut (chris pratt)')
                    continue

                # Process LEFT leg Kinematics
                for a, data in enumerate(left_kinematics):
                    currAngle = np.array(currKinematic[data].iloc[firstDex:nextDex])
                    if a == 0:
                        left_currAngles = np.zeros((len(left_kinematics), currAngle.shape[0]))
                        left_currAnglesFull = np.zeros((len(joints),len(axis),currAngle.shape[0]))
                    if 'hip' in data.lower():
                        if 'adduction' in data.lower():
                            left_currAnglesFull[0,0] = currAngle
                        elif 'flexion' in data.lower():
                            left_currAnglesFull[0,-1] = currAngle
                        elif 'rotation' in data.lower():
                            left_currAnglesFull[0,1] = currAngle

                    elif 'knee' in data.lower():
                        left_currAnglesFull[1,-1] = currAngle
                    elif 'ankle' in data.lower():
                        left_currAnglesFull[-1,-1]=currAngle
                                    
                    left_currAngles[a] = currAngle

                time_diff_first = (currKinetic[timeType[2]] - firstTime).abs()
                firstDex = time_diff_first.idxmin()                
                time_diff_next = (currKinetic[timeType[2]] - nextTime).abs()
                nextDex = time_diff_next.idxmin()

                # Process LEFT leg Kinetics
                for k, data in enumerate(left_kinetics):
                    currTorque = np.array(currKinetic[data].iloc[firstDex:nextDex])
                    if k == 0:
                        left_currTorques = np.zeros((len(left_kinematics), currTorque.shape[0]))
                        left_currTorquesFull = np.zeros((len(joints),len(axis),currTorque.shape[0]))
                    if 'hip' in data.lower():
                        if 'adduction' in data.lower():
                            left_currTorquesFull[0,0] = currTorque
                        elif 'flexion' in data.lower():
                            left_currTorquesFull[0,-1] = currTorque
                        elif 'rotation' in data.lower():
                            left_currTorquesFull[0,1] = currTorque

                    elif 'knee' in data.lower():
                        left_currTorquesFull[1,-1] = currTorque
                    elif 'ankle' in data.lower():
                        left_currTorquesFull[-1,-1]=currTorque
                                    
                    left_currTorques[k] = currTorque
                stridesAngleLeft.append(left_currAnglesFull)
                stridesTorqueLeft.append(left_currTorquesFull)
                stridesEMGLeft.append(left_currEMGs)
        
            # Append trial data for both legs
            right_trialTorques.append(stridesTorqueRight)
            right_trialAngles.append(stridesAngleRight)
            right_trialEMGs.append(stridesEMGRight)
            
            left_trialTorques.append(stridesTorqueLeft)
            left_trialAngles.append(stridesAngleLeft)
            left_trialEMGs.append(stridesEMGLeft)
        input(len(right_trialAngles[0]))
        
        # Append patient data for both legs
        right_patientAngles.append(right_trialAngles)
        right_patientEMGs.append(right_trialEMGs)
        right_patientTorques.append(right_trialTorques)
        
        left_patientAngles.append(left_trialAngles)
        left_patientEMGs.append(left_trialEMGs)
        left_patientTorques.append(left_trialTorques)

    leftEMGMask=[1 if x != 0 else 0 for x in LeftEMGs]
    rightEMGMask=[1 if x != 0 else 0 for x in RightEMGs]
    rightKineticMask = np.zeros((len(joints),len(axis)))
    leftKineticMask = np.zeros((len(joints),len(axis)))
    rightKinematicMask = np.zeros((len(joints),len(axis)))
    leftKinematicMask = np.zeros((len(joints),len(axis)))

    rightKinematicMask[0,:] = 1
    rightKinematicMask[1,-1] = 1
    rightKinematicMask[-1,-1] = 1

    leftKinematicMask[0,:] = 1
    leftKinematicMask[1,-1] = 1
    leftKinematicMask[-1,-1] = 1

    rightKineticMask[0,:] = 1
    rightKineticMask[1,-1] = 1
    rightKineticMask[-1,-1] = 1

    leftKineticMask[0,:] = 1
    leftKineticMask[1,-1] = 1
    leftKineticMask[-1,-1] = 1

    return {
        'mask':{'left':{'emg':leftEMGMask,'kinetic':leftKineticMask,'kinematic':leftKinematicMask},
                'right':{'emg':rightEMGMask,'kinetic':rightKineticMask,'kinematic':rightKinematicMask}
                },
        'walk': {
            'right': {
                'emg': right_patientEMGs, 
                'kinetic': right_patientTorques, 
                'kinematic': right_patientAngles
            },
            'left': {
                'emg': left_patientEMGs, 
                'kinetic': left_patientTorques, 
                'kinematic': left_patientAngles
            }
        }
    }


def parseEmbry(currPath = "C:/EMG/datasets/Embry/InclineExperiment.mat",emgSampleHz=1000):
    #set incline types to ramp up or ramp down
    OriginalEMGTypes = ['RF','TA','BF','GC']
    EMGTypes = [0,'RF',0,'TA','BF',0,0,'GC',0,0,0,0,0]
    

    KineticTypes = ['hip','knee','ankle']
    KinematicTypes = ['hip','knee','ankle']
    axis = ['x','z','y']
    rotation = ['roll','yaw','pitch']

    whichLeg = ['right','left']

    # Separate storage for each trial type
    RightPatientEMG_walk = []
    RightPatientKinetic_walk = []
    RightPatientKinematic_walk = []
    LeftPatientEMG_walk = []
    LeftPatientKinetic_walk = []
    LeftPatientKinematic_walk = []

    RightPatientEMG_rampup = []
    RightPatientKinetic_rampup = []
    RightPatientKinematic_rampup = []
    LeftPatientEMG_rampup = []
    LeftPatientKinetic_rampup = []
    LeftPatientKinematic_rampup = []

    RightPatientEMG_rampdown = []
    RightPatientKinetic_rampdown = []
    RightPatientKinematic_rampdown = []
    LeftPatientEMG_rampdown = []
    LeftPatientKinetic_rampdown = []
    LeftPatientKinematic_rampdown = []

    walk_tasks = [
        's0x8i0',
        's1i0',
        's1x2i0'
    ]

    rampup_tasks = [
        's0x8i2x5', 's1i2x5', 's1x2i2x5',  # 2.5 degrees
        's0x8i5',   's1i5',   's1x2i5',    # 5.0 degrees
        's0x8i7x5', 's1i7x5', 's1x2i7x5',  # 7.5 degrees
        's0x8i10',  's1i10',  's1x2i10'    # 10.0 degrees
    ]

    rampdown_tasks = [
        's0x8d2x5', 's1d2x5', 's1x2d2x5',  # -2.5 degrees
        's0x8d5',   's1d5',   's1x2d5',    # -5.0 degrees
        's0x8d7x5', 's1d7x5', 's1x2d7x5',  # -7.5 degrees
        's0x8d10',  's1d10',  's1x2d10'    # -10.0 degrees
    ]    

    eng.eval(f"data=load('{currPath}')",nargout=0)
    patients=eng.eval("fieldnames(data.Continuous)",nargout=1)
    
    for patient in patients:
        patient = patient.strip()
        trialType =eng.eval(f"fieldnames(data.Continuous.{patient})",nargout=1)

        # Separate trial lists for each type
        RightTrialEMG_walk = []
        RightTrialKinetic_walk = []
        RightTrialKinematic_walk = []
        LeftTrialEMG_walk = []
        LeftTrialKinetic_walk = []
        LeftTrialKinematic_walk = []

        RightTrialEMG_rampup = []
        RightTrialKinetic_rampup = []
        RightTrialKinematic_rampup = []
        LeftTrialEMG_rampup = []
        LeftTrialKinetic_rampup = []
        LeftTrialKinematic_rampup = []

        RightTrialEMG_rampdown = []
        RightTrialKinetic_rampdown = []
        RightTrialKinematic_rampdown = []
        LeftTrialEMG_rampdown = []
        LeftTrialKinetic_rampdown = []
        LeftTrialKinematic_rampdown = []

        #MVC Normalization
        LeftEMGMax = [0] * len(EMGTypes)
        RightEMGMax = [0] * len(EMGTypes)

        for i, currLeg in enumerate(whichLeg):
            for trial in trialType:
                trial = trial.strip()
                if 'subjectdetails' in trial or ('s0x8i2x5' in trial and 'AB09' in patient): continue
                
                # Determine trial type
                if trial in walk_tasks:
                    trial_category = 'walk'
                elif trial in rampup_tasks:
                    trial_category = 'rampup'
                elif trial in rampdown_tasks:
                    trial_category = 'rampdown'
                else:
                    continue  # Skip unknown trial types

                for e,currEMG in enumerate(EMGTypes):
                    if currEMG==0: continue
                    currData =np.array(eng.eval(f"data.Gaitcycle.{patient}.{trial}.emgdata.emg.{currLeg}.{currEMG}",nargout=1)).T

                    if i==0: RightEMGMax[e]=max(np.abs(np.percentile(currData,99.5)),RightEMGMax[e])
                    else: LeftEMGMax[e]=max(np.abs(np.percentile(currData,99.5)),LeftEMGMax[e])
        #MVC Normalization

        for i,currLeg in enumerate(whichLeg):
            
            for trial in trialType:
                trial = trial.strip()
                if 'subjectdetails' in trial or ('s0x8i2x5' in trial and 'AB09' in patient): continue
                
                # Determine trial type
                if trial in walk_tasks:
                    trial_category = 'walk'
                elif trial in rampup_tasks:
                    trial_category = 'rampup'
                elif trial in rampdown_tasks:
                    trial_category = 'rampdown'
                else:
                    continue  # Skip unknown trial types

                #FIND MIN Cycle
                miner =1e9

                for e,currEMG in enumerate(EMGTypes):
                    if currEMG==0: continue
                    currData =np.array(eng.eval(f"data.Gaitcycle.{patient}.{trial}.emgdata.emg.{currLeg}.{currEMG}",nargout=1)).T
                    miner=min(miner,currData.shape[0])

                strideEMG = []                
                for strideIdx in range(miner):
                    currTime =np.array(eng.eval(f"data.Gaitcycle.{patient}.{trial}.cycles.{currLeg}.time(:,{strideIdx+1})",nargout=1)).flatten()
                    duration = currTime[-1] - currTime[0]
                    new_count = int(duration * emgSampleHz)
                    for e,currEMG in enumerate(EMGTypes):

                        if currEMG==0:
                            continue
                        currData =np.array(eng.eval(f"data.Gaitcycle.{patient}.{trial}.emgdata.emg.{currLeg}.{currEMG}(:,{strideIdx+1})",nargout=1)).flatten()

                        # Create new uniform time array
                        new_times = np.linspace(currTime[0], currTime[-1], new_count)
                        interpolator = interp1d(currTime, currData, kind='linear')
                        resampled_emg = interpolator(new_times)

                    #TODO storage
                        if e==1: 
                            EMGData = np.zeros((len(EMGTypes),new_count))
                        
                        if i==0:
                            if RightEMGMax[e]>0:
                                EMGData[e]=(np.abs(resampled_emg/RightEMGMax[e])).clip(max=1.0)
                            else: EMGData[e]=(np.abs(resampled_emg)).clip(max=1.0)
                        else:
                            if LeftEMGMax[e]>0: 
                                EMGData[e]=(np.abs(resampled_emg/LeftEMGMax[e])).clip(max=1.0)
                            else: 
                                EMGData[e]=(np.abs(resampled_emg)).clip(max=1.0)
                    strideEMG.append(EMGData)

                for a,currKinematic in enumerate(KinematicTypes):
                    for ca,currAxis in enumerate(axis):
                        currData =np.array(eng.eval(f"data.Gaitcycle.{patient}.{trial}.kinematics.jointangles.{currLeg}.{currKinematic}.{currAxis}",nargout=1)).T
                        if a ==0 and ca==0: 
                            AngleData = np.zeros((len(KinematicTypes),3,miner,currData.shape[1]))
                        
                        #x=roll, y=pitch, z=yaw
                        if 'x'==currAxis.lower():
                            AngleData[a,0]=currData[:miner]

                        elif 'y' == currAxis.lower():
                            AngleData[a,-1] = currData[:miner]

                        else:
                            AngleData[a,1] = currData[:miner]

                for m,currKinetic in enumerate(KineticTypes):
                    for ca,currAxis in enumerate(axis):
                        currData =np.array(eng.eval(f"data.Gaitcycle.{patient}.{trial}.kinetics.jointforce.{currLeg}.{currKinetic}.{currAxis}",nargout=1)).T
                        if m ==0 and ca==0: KineticData = np.zeros((len(KineticTypes),3,miner,currData.shape[1]))
                        if 'x'==currAxis.lower():
                            KineticData[m,0]=currData[:miner]

                        elif 'y' == currAxis.lower():
                            KineticData[m,-1] = currData[:miner]

                        else:
                            KineticData[m,1] = currData[:miner]
                KineticData = KineticData.transpose(2,0,1,3)
                AngleData = AngleData.transpose(2,0,1,3)

                # Append to appropriate lists based on leg and trial category
                if i==0:  # Right leg
                    if trial_category == 'walk':
                        RightTrialEMG_walk.append(strideEMG)
                        RightTrialKinetic_walk.append(KineticData)
                        RightTrialKinematic_walk.append(AngleData)
                    elif trial_category == 'rampup':
                        RightTrialEMG_rampup.append(strideEMG)
                        RightTrialKinetic_rampup.append(KineticData)
                        RightTrialKinematic_rampup.append(AngleData)
                    elif trial_category == 'rampdown':
                        RightTrialEMG_rampdown.append(strideEMG)
                        RightTrialKinetic_rampdown.append(KineticData)
                        RightTrialKinematic_rampdown.append(AngleData)
                else:  # Left leg
                    if trial_category == 'walk':
                        LeftTrialEMG_walk.append(strideEMG)
                        LeftTrialKinetic_walk.append(KineticData)
                        LeftTrialKinematic_walk.append(AngleData)
                    elif trial_category == 'rampup':
                        LeftTrialEMG_rampup.append(strideEMG)
                        LeftTrialKinetic_rampup.append(KineticData)
                        LeftTrialKinematic_rampup.append(AngleData)
                    elif trial_category == 'rampdown':
                        LeftTrialEMG_rampdown.append(strideEMG)
                        LeftTrialKinetic_rampdown.append(KineticData)
                        LeftTrialKinematic_rampdown.append(AngleData)

        # Append patient data for each trial type
        RightPatientEMG_walk.append(RightTrialEMG_walk)
        RightPatientKinetic_walk.append(RightTrialKinetic_walk)
        RightPatientKinematic_walk.append(RightTrialKinematic_walk)
        LeftPatientEMG_walk.append(LeftTrialEMG_walk)
        LeftPatientKinetic_walk.append(LeftTrialKinetic_walk)
        LeftPatientKinematic_walk.append(LeftTrialKinematic_walk)

        RightPatientEMG_rampup.append(RightTrialEMG_rampup)
        RightPatientKinetic_rampup.append(RightTrialKinetic_rampup)
        RightPatientKinematic_rampup.append(RightTrialKinematic_rampup)
        LeftPatientEMG_rampup.append(LeftTrialEMG_rampup)
        LeftPatientKinetic_rampup.append(LeftTrialKinetic_rampup)
        LeftPatientKinematic_rampup.append(LeftTrialKinematic_rampup)

        RightPatientEMG_rampdown.append(RightTrialEMG_rampdown)
        RightPatientKinetic_rampdown.append(RightTrialKinetic_rampdown)
        RightPatientKinematic_rampdown.append(RightTrialKinematic_rampdown)
        LeftPatientEMG_rampdown.append(LeftTrialEMG_rampdown)
        LeftPatientKinetic_rampdown.append(LeftTrialKinetic_rampdown)
        LeftPatientKinematic_rampdown.append(LeftTrialKinematic_rampdown)

    emgMask=[1 if x != 0 else 0 for x in EMGTypes]
    jointMask = np.ones((3,3))

    return {
        'mask':{'right':{'emg':emgMask,'kinetic':jointMask,'kinematic':jointMask},'left':{'emg':emgMask,'kinetic':jointMask,'kinematic':jointMask}},
        'walk': {
            'right': {'emg': RightPatientEMG_walk, 'kinetic': RightPatientKinetic_walk, 'kinematic': RightPatientKinematic_walk},
            'left': {'emg': LeftPatientEMG_walk, 'kinetic': LeftPatientKinetic_walk, 'kinematic': LeftPatientKinematic_walk}
        },
        'rampup': {
            'right': {'emg': RightPatientEMG_rampup, 'kinetic': RightPatientKinetic_rampup, 'kinematic': RightPatientKinematic_rampup},
            'left': {'emg': LeftPatientEMG_rampup, 'kinetic': LeftPatientKinetic_rampup, 'kinematic': LeftPatientKinematic_rampup}
        },
        'rampdown': {
            'right': {'emg': RightPatientEMG_rampdown, 'kinetic': RightPatientKinetic_rampdown, 'kinematic': RightPatientKinematic_rampdown},
            'left': {'emg': LeftPatientEMG_rampdown, 'kinetic': LeftPatientKinetic_rampdown, 'kinematic': LeftPatientKinematic_rampdown}
        }
    }

    #NOTE R_TA for AB04-AB08 sensor malfunction??

def parseGrimmer(currPath = 'C:/EMG/datasets/Grimmer'):
    #6 stair types, NOTE stairup and stairdown are merged
    OriginalrightEMGs = ['bcf_r', 'foo_r', 'gas_r', 'rcf_r', 'sha_r', 'sol_r', 'tib_r', 'vas_r']
    OriginalleftEMGs = ['bcf_l', 'foo_l', 'gas_l', 'rcf_l', 'sha_l', 'sol_l', 'tib_l', 'vas_l']
    rightEMGs = ['vas_r','rcf_r',0,'tib_r','bcf_r',0,0,'gas_r','sol_r', 0, 0, 0,0]
    leftEMGs = ['vas_l','rcf_l',0,'tib_l','bcf_l',0,0,'gas_l','sol_l', 0, 0, 0,0]
    
    rightMoments = ['hip_moment_flexion_r','knee_moment_flexion_r', 'ankle_moment_flexion_r','hip_moment_adduction_r', 'hip_moment_rotation_r']
    leftMoments = ['hip_moment_flexion_l', 'knee_moment_flexion_l', 'ankle_moment_flexion_l','hip_moment_adduction_l', 'hip_moment_rotation_l']

    rightAngles = ['hip_angle_flexion_r', 'knee_angle_flexion_r', 'ankle_angle_flexion_r','hip_angle_adduction_r', 'hip_angle_rotation_r']
    leftAngles = ['hip_angle_flexion_l','knee_angle_flexion_l', 'ankle_angle_flexion_l','hip_angle_adduction_l', 'hip_angle_rotation_l'] 
    
    joints = ['hip','knee','ankle']
    axis = ['roll','yaw','pitch']
    ascentTypes = ['TD_ascent','TD_descent']

    angleTorquePath = f'{currPath}/Processed'
    EMGPath = f'{currPath}/Preprocessed'

    StairUpPatientLeftEMGs = []
    StairUpPatientLeftAngles = []
    StairUpPatientLeftMoments = []
    StairUpPatientRightMoments = []
    StairUpPatientRightAngles = []
    StairUpPatientRightEMGs = []
    StairDownPatientLeftEMGs = []
    StairDownPatientLeftAngles = []
    StairDownPatientLeftMoments = []
    StairDownPatientRightMoments = []
    StairDownPatientRightAngles = []
    StairDownPatientRightEMGs = []

    for currEMG,currAngle,currMoment, currTouchdown in zip(sorted(os.listdir(f'{EMGPath}/EMG')),
                       sorted(os.listdir(f'{angleTorquePath}/Joint_Kinematics')),
                       sorted(os.listdir(f'{angleTorquePath}/Joint_Kinetics')),
                       sorted(os.listdir(f'{angleTorquePath}/Touchdowns'))):
        if currEMG[-6]=='1' or currAngle[-6]=='1' or currMoment[-6]=='1':
            assert currEMG[-6:-4] == currAngle[-6:-4] == currMoment[-6:-4] == currTouchdown[-6:-4]
        else:
            assert currEMG[-5]==currAngle[-5]==currMoment[-5] == currTouchdown[-5]
        eng.eval(f"angleData=load('{angleTorquePath}/Joint_Kinematics/{currAngle}')",nargout=0)
        eng.eval(f"momentData=load('{angleTorquePath}/Joint_Kinetics/{currMoment}')",nargout=0)
        eng.eval(f"emgData=load('{EMGPath}/EMG/{currEMG}')",nargout=0)
        eng.eval(f"touchdownData=load('{angleTorquePath}/Touchdowns/{currTouchdown}')",nargout=0)

        trialCount = np.array(eng.eval(f'size(angleData.Joint_Kinematics)'))
        maxREMG = [0] * len(rightEMGs)
        maxLEMG = [0] * len(rightEMGs)

        #NOTE MVC calculation
        for currActivity in range(int(trialCount[0][0])):
            trialNums=np.array(eng.eval(f'size(angleData.Joint_Kinematics{{{currActivity+1}}})',nargout=1))
            for currTrial in range(int(trialNums[0][0])):
                for i in range(2):
                    if i==0:
                        #right
                        for e,nowEMG in enumerate(rightEMGs):
                            if nowEMG ==0: continue
                            currData=np.array(eng.eval(f'emgData.Emg{{{currActivity+1}}}{{{currTrial+1}}}.{nowEMG}',nargout=1)).flatten()
                            maxREMG[e]=max(np.percentile(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currData,w0=50,fs=1111.111),fs=1111.111))),99.5),maxREMG[e])

                    else:
                        #left
                        for e,nowEMG in enumerate(leftEMGs):
                            if nowEMG==0: continue
                            currData=np.array(eng.eval(f'emgData.Emg{{{currActivity+1}}}{{{currTrial+1}}}.{nowEMG}',nargout=1)).flatten()
                            maxLEMG[e]=max(np.percentile(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currData,w0=50,fs=1111.111),fs=1111.111))),99.5),maxLEMG[e])
    
        #NOTE MVC calculation
        #activities are represented by differing stair heights 
        StairUpTrialLeftEMGs = []
        StairUpTrialLeftAngles = []
        StairUpTrialLeftMoments = []
        StairUpTrialRightMoments = []
        StairUpTrialRightAngles = []
        StairUpTrialRightEMGs = []

        StairDownTrialLeftEMGs = []
        StairDownTrialLeftAngles = []
        StairDownTrialLeftMoments = []
        StairDownTrialRightMoments = []
        StairDownTrialRightAngles = []
        StairDownTrialRightEMGs = []

        for currActivity in range(int(trialCount[0][0])):
            trialNums=np.array(eng.eval(f'size(angleData.Joint_Kinematics{{{currActivity+1}}})',nargout=1))

            for currTrial in range(int(trialNums[0][0])):

                for i in range(2):
                    if i==0:
                        
                        for currAscent in ascentTypes:
                            rightSegIndices=eng.eval(f'touchdownData.{currAscent}{{{currActivity+1}}}{{{currTrial+1}}}.tdR',nargout=1)                            

                            cycleAngle = []
                            cycleMoment = []
                            cycleEMG = []
                            for numba,currCycleIdx in enumerate(rightSegIndices[0]):
                                if numba == len(rightSegIndices[0])-1:
                                    break
                                currCycleIdx = int(currCycleIdx)
                                nxtCycleIdx = int(rightSegIndices[0][numba+1])
                                cycle_length = nxtCycleIdx - currCycleIdx +1

                                for a,currAngle in enumerate(rightAngles):
                                    currData=np.array(eng.eval(f'angleData.Joint_Kinematics{{{currActivity+1}}}{{{currTrial+1}}}.{currAngle}({currCycleIdx}:{nxtCycleIdx})',nargout=1)).flatten()
                                    if a ==0: 
                                        currRightAngles = np.zeros((len(rightAngles),cycle_length))
                                        currRightAnglesFull = np.zeros((len(joints),len(axis),cycle_length))
                                    if 'hip' in currAngle.lower():
                                        if 'adduction' in currAngle.lower():
                                            currRightAnglesFull[0,0]=currData
                                        elif 'rotation' in currAngle.lower():
                                            currRightAnglesFull[0,1]=currData
                                        elif 'flexion' in currAngle.lower():
                                            currRightAnglesFull[0,-1]=currData

                                    elif 'knee' in currAngle.lower():
                                        currRightAnglesFull[1,-1]=currData
                                    
                                    elif 'ankle' in currAngle.lower():
                                        currRightAnglesFull[-1,-1]=currData
                                    currRightAngles[a]=currData

                                for m,currMoment in enumerate(rightMoments):
                                    currData=np.array(eng.eval(f'momentData.Joint_Kinetics{{{currActivity+1}}}{{{currTrial+1}}}.{currMoment}({currCycleIdx}:{nxtCycleIdx})',nargout=1)).flatten()

                                    if m ==0:
                                        currRightMoments = np.zeros((len(rightMoments),currData.shape[0]))
                                        currRightMomentsFull = np.zeros((len(joints),len(axis),currData.shape[0]))
                                    if 'hip' in currMoment.lower():
                                        if 'adduction' in currMoment.lower():
                                            currRightMomentsFull[0,0]=currData
                                        elif 'rotation' in currMoment.lower():
                                            currRightMomentsFull[0,1]=currData
                                        elif 'flexion' in currMoment.lower():
                                            currRightMomentsFull[0,-1]=currData

                                    elif 'knee' in currMoment.lower():
                                        currRightMomentsFull[1,-1]=currData
                                    
                                    elif 'ankle' in currMoment.lower():
                                        currRightMomentsFull[-1,-1]=currData
                                    
                                    currRightMoments[m]=currData
                                currRightMoments = currRightMomentsFull
                                currRightAngles = currRightAnglesFull
                        
                                for e,nowEMG in enumerate(rightEMGs):
                                    if nowEMG==0: continue
                                    currData=np.array(eng.eval(f'emgData.Emg{{{currActivity+1}}}{{{currTrial+1}}}.{nowEMG}({currCycleIdx}:{nxtCycleIdx})')).flatten()

                                    if e ==0: currRightEMGs = np.zeros((len(rightEMGs),cycle_length))

                                    if maxREMG[e] == 0:
                                        currRightEMGs[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currData,w0=50,fs=1111.111),fs=1111.111)))).clip(max=1.0)
                                    else: currRightEMGs[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currData,w0=50,fs=1111.111),fs=1111.111)))/maxREMG[e]).clip(max=1.0)
                                
                                cycleAngle.append(currRightAnglesFull)
                                cycleMoment.append(currRightMomentsFull)
                                cycleEMG.append(currRightEMGs)

                            if currAscent=='TD_ascent':
                                StairUpTrialRightMoments.append(cycleMoment)
                                StairUpTrialRightAngles.append(cycleAngle)
                                StairUpTrialRightEMGs.append(cycleEMG)
                            elif currAscent=='TD_descent':
                                StairDownTrialRightMoments.append(cycleMoment)
                                StairDownTrialRightAngles.append(cycleAngle)
                                StairDownTrialRightEMGs.append(cycleEMG)
                    else:
                        for currAscent in ascentTypes:
                            leftSegIndices=eng.eval(f'touchdownData.{currAscent}{{{currActivity+1}}}{{{currTrial+1}}}.tdL',nargout=1)

                            cycleAngle = []
                            cycleMoment = []
                            cycleEMG = []
                            for numba,currCycleIdx in enumerate(leftSegIndices[0]):
                                if numba == len(leftSegIndices[0])-1:
                                    break
                                currCycleIdx = int(currCycleIdx)
                                nxtCycleIdx = int(leftSegIndices[0][numba+1])
                                cycle_length = nxtCycleIdx - currCycleIdx + 1

                                for a,currAngle in enumerate(leftAngles):
                                    currData=np.array(eng.eval(f'angleData.Joint_Kinematics{{{currActivity+1}}}{{{currTrial+1}}}.{currAngle}({currCycleIdx}:{nxtCycleIdx})',nargout=1)).flatten()
                                    if a ==0: 
                                        currLeftAngles = np.zeros((len(leftAngles),cycle_length))
                                        currLeftAnglesFull = np.zeros((len(joints),len(axis),cycle_length))
                                    if 'hip' in currAngle.lower():
                                        if 'adduction' in currAngle.lower():
                                            currLeftAnglesFull[0,0]=currData
                                        elif 'rotation' in currAngle.lower():
                                            currLeftAnglesFull[0,1]=currData
                                        elif 'flexion' in currAngle.lower():
                                            currLeftAnglesFull[0,-1]=currData

                                    elif 'knee' in currAngle.lower():
                                        currLeftAnglesFull[1,-1]=currData
                                    
                                    elif 'ankle' in currAngle.lower():
                                        currLeftAnglesFull[-1,-1]=currData
                                    currLeftAngles[a]=currData

                                for m,currMoment in enumerate(leftMoments):
                                    currData=np.array(eng.eval(f'momentData.Joint_Kinetics{{{currActivity+1}}}{{{currTrial+1}}}.{currMoment}({currCycleIdx}:{nxtCycleIdx})',nargout=1)).flatten()
                                    if m ==0:
                                        currLeftMoments = np.zeros((len(leftMoments),currData.shape[0]))
                                        currLeftMomentsFull = np.zeros((len(joints),len(axis),currData.shape[0]))
                                    if 'hip' in currMoment.lower():
                                        if 'adduction' in currMoment.lower():
                                            currLeftMomentsFull[0,0]=currData
                                        elif 'rotation' in currMoment.lower():
                                            currLeftMomentsFull[0,1]=currData
                                        elif 'flexion' in currMoment.lower():
                                            currLeftMomentsFull[0,-1]=currData

                                    elif 'knee' in currMoment.lower():
                                        currLeftMomentsFull[1,-1]=currData
                                    
                                    elif 'ankle' in currMoment.lower():
                                        currLeftMomentsFull[-1,-1]=currData
                                    
                                    currLeftMoments[m]=currData
                                currLeftMoments = currLeftMomentsFull
                                currLeftAngles = currLeftAnglesFull
                        
                                for e,nowEMG in enumerate(leftEMGs):
                                    if nowEMG==0: continue
                                    currData=np.array(eng.eval(f'emgData.Emg{{{currActivity+1}}}{{{currTrial+1}}}.{nowEMG}({currCycleIdx}:{nxtCycleIdx})',nargout=1)).flatten()
                                    if e ==0: currLeftEMGs = np.zeros((len(leftEMGs),cycle_length))

                                    if maxLEMG[e] == 0:
                                        currLeftEMGs[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currData,w0=50,fs=1111.111),fs=1111.111)))).clip(max=1.0)
                                    else: currLeftEMGs[e]=(np.abs(apply_wavelet_denoising(apply_bandpass(apply_notch(currData,w0=50,fs=1111.111),fs=1111.111)))/maxLEMG[e]).clip(max=1.0)
                                
                                cycleAngle.append(currLeftAnglesFull)
                                cycleMoment.append(currLeftMomentsFull)
                                cycleEMG.append(currLeftEMGs)

                            if currAscent=='TD_ascent':
                                StairUpTrialLeftMoments.append(cycleMoment)
                                StairUpTrialLeftAngles.append(cycleAngle)
                                StairUpTrialLeftEMGs.append(cycleEMG)
                            elif currAscent=='TD_descent':
                                StairDownTrialLeftMoments.append(cycleMoment)
                                StairDownTrialLeftAngles.append(cycleAngle)
                                StairDownTrialLeftEMGs.append(cycleEMG)
                        
        #
        StairUpPatientRightMoments.append(StairUpTrialRightMoments)
        StairUpPatientLeftMoments.append(StairUpTrialLeftMoments)
        StairUpPatientRightEMGs.append(StairUpTrialRightEMGs)
        StairUpPatientLeftEMGs.append(StairUpTrialLeftEMGs)
        StairUpPatientRightAngles.append(StairUpTrialRightAngles)
        StairUpPatientLeftAngles.append(StairUpTrialLeftAngles)

        StairDownPatientRightMoments.append(StairDownTrialRightMoments)
        StairDownPatientLeftMoments.append(StairDownTrialLeftMoments)
        StairDownPatientRightEMGs.append(StairDownTrialRightEMGs)
        StairDownPatientLeftEMGs.append(StairDownTrialLeftEMGs)
        StairDownPatientRightAngles.append(StairDownTrialRightAngles)
        StairDownPatientLeftAngles.append(StairDownTrialLeftAngles)
    leftEMGMask = [1 if x != 0 else 0 for x in leftEMGs]
    rightEMGMask = [1 if x != 0 else 0 for x in rightEMGs]

    rightKineticMask = np.zeros((len(joints),len(axis)))
    leftKineticMask = np.zeros((len(joints),len(axis)))
    rightKinematicMask = np.zeros((len(joints),len(axis)))
    leftKinematicMask = np.zeros((len(joints),len(axis)))
    rightKineticMask[0,:]=1
    leftKineticMask[0,:]=1
    leftKinematicMask[0,:]=1
    rightKinematicMask[0,:]=1
    rightKineticMask[1:,-1]=1
    leftKineticMask[1:,-1]=1
    leftKinematicMask[1:,-1]=1
    rightKinematicMask[1:,-1]=1

    return{
        'mask': 
            {'left':{'emg':leftEMGMask,'angle':leftKinematicMask,'kinetic':leftKineticMask},
            'right':{'emg':rightEMGMask,'angle':rightKinematicMask,'kinetic':rightKineticMask}},
        'stairUp':
           {'left':{'emg':StairUpPatientLeftEMGs,'angle':StairUpPatientLeftAngles,'kinetic':StairUpPatientLeftMoments},
           'right':{'emg':StairUpPatientRightEMGs,'angle':StairUpPatientRightAngles,'kinetic':StairUpPatientRightMoments}},
        'stairDown':
           {'left':{'emg': StairDownPatientLeftEMGs,'angle': StairDownPatientLeftAngles,'kinetic': StairDownPatientLeftMoments},
           'right':{'emg': StairDownPatientRightEMGs,'angle': StairDownPatientRightAngles,'kinetic': StairDownPatientRightMoments}},
           }

def parseMacaluso(currPath='C:/EMG/datasets/Macaluso/Subject',emgSampleRate=1000):
    # --- DATA STORAGE LISTS ---
    # Downhill
    patientDownRampLeftEMG, patientDownRampRightEMG = [], []
    patientDownRampLeftAngle, patientDownRampRightAngle = [], []
    patientDownRampLeftKinetic, patientDownRampRightKinetic = [], []
    
    # Uphill
    patientUpRampLeftEMG, patientUpRampRightEMG = [], []
    patientUpRampLeftAngle, patientUpRampRightAngle = [], []
    patientUpRampLeftKinetic, patientUpRampRightKinetic = [], []
    
    # Level Walk
    patientWalkLeftEMG, patientWalkRightEMG = [], []
    patientWalkLeftAngle, patientWalkRightAngle = [], []
    patientWalkLeftKinetic, patientWalkRightKinetic = [], []

    tasks = ['downhill', 'uphill', 'level']
    EMGsorginal = ['RF_norm', 'TA_norm', 'BF_norm', 'GC_norm']
    EMGs = [0,'RF_norm',0,'TA_norm','BF_norm',0,'GC_norm', 0, 0, 0,0, 0, 0]
    maxEMGsLeft = [0] * len(EMGs)
    maxEMGsRight = [0] * len(EMGs)

    joints = ['hip', 'knee', 'ankle']
    angles = ['hip', 'knee', 'ankle']
    axis = ['x_norm', 'z_norm', 'y_norm']
        
    for patient in sorted(os.listdir(currPath)):
        patientPath = currPath + '/' + patient
        print(f"Loading {patient}...")
        eng.eval(f"data=load('{patientPath}')", nargout=0)

        # Temporary lists for the current patient's trials
        trialLeftEMG, trialRightEMG = [], []            
        trialLeftMoment, trialRightMoment = [], []
        trialLeftAngle, trialRightAngle = [], []

        #NOTE MVC Calculation

        for taskNum, currTask in enumerate(tasks):
            try:
                trials = eng.eval(f'fieldnames(data.{patient[:-4]}.{currTask})', nargout=1)
            except:
                print('tfed')
                continue

            for t, currTrial in enumerate(trials):
                if 'part' in currTrial:
                    l_emg_ref = f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.left.L{EMGs[1]}"
                    leftEMGShape = np.array(eng.eval(f'size({l_emg_ref})', nargout=1))
                    num_strides_emg_L = int(leftEMGShape[0][1])

                    timeShapeL = np.array(eng.eval(f'size(data.{patient[:-4]}.{currTask}.{currTrial}.cycles.left.time)', nargout=1))
                    num_strides_kin_L = int(timeShapeL[0][1])

                    # --- RIGHT LEG METADATA ---
                    r_emg_ref = f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.right.R{EMGs[1]}"
                    rightEMGShape = np.array(eng.eval(f'size({r_emg_ref})', nargout=1))
                    num_strides_emg_R = int(rightEMGShape[0][1])

                    timeShapeR = np.array(eng.eval(f'size(data.{patient[:-4]}.{currTask}.{currTrial}.cycles.right.time)', nargout=1))
                    num_strides_kin_R = int(timeShapeR[0][1])

                    safe_strides_L = min(num_strides_emg_L, num_strides_kin_L)
                    safe_strides_R = min(num_strides_emg_R, num_strides_kin_R)

                    for e, currEMG in enumerate(EMGs):
                        if currEMG==0: continue
                        l_data = np.array(eng.eval(f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.left.L{currEMG}", nargout=1))
                        r_data = np.array(eng.eval(f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.right.R{currEMG}", nargout=1))
                        
                        # Transpose to get [Strides, Time], then slice to safe limit
    
                        maxEMGsLeft[e]=max(maxEMGsLeft[e],np.max(l_data.T[:safe_strides_L]))
                        maxEMGsRight[e]=max(maxEMGsRight[e],np.max(r_data.T[:safe_strides_R]))

        #NOTE MVC Calculation


        for taskNum, currTask in enumerate(tasks):
            trialLeftEMG = []
            trialRightEMG = []           
            trialLeftMoment = []
            trialRightMoment = []
            trialLeftAngle = []
            trialRightAngle = []

            try:
                trials = eng.eval(f'fieldnames(data.{patient[:-4]}.{currTask})', nargout=1)
            except:
                continue

            for t, currTrial in enumerate(trials):
                if 'part' in currTrial:
                    
                    
                    # =========================================================
                    # 1. FETCH SHAPES INDEPENDENTLY (LEFT vs RIGHT)
                    # =========================================================
                    
                    # --- LEFT LEG METADATA ---
                    l_emg_ref = f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.left.L{EMGs[1]}"
                    leftEMGShape = np.array(eng.eval(f'size({l_emg_ref})', nargout=1))
                    num_strides_emg_L = int(leftEMGShape[0][1])

                    timeShapeL = np.array(eng.eval(f'size(data.{patient[:-4]}.{currTask}.{currTrial}.cycles.left.time)', nargout=1))
                    num_strides_kin_L = int(timeShapeL[0][1])

                    # --- RIGHT LEG METADATA ---
                    r_emg_ref = f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.right.R{EMGs[1]}"
                    rightEMGShape = np.array(eng.eval(f'size({r_emg_ref})', nargout=1))
                    num_strides_emg_R = int(rightEMGShape[0][1])

                    timeShapeR = np.array(eng.eval(f'size(data.{patient[:-4]}.{currTask}.{currTrial}.cycles.right.time)', nargout=1))
                    num_strides_kin_R = int(timeShapeR[0][1])

                    # =========================================================
                    # 2. DETERMINE SAFE STRIDE COUNTS (TRUNCATE MISMATCHES)
                    # =========================================================
                    # We take the minimum to ensure we don't index out of bounds
                    safe_strides_L = min(num_strides_emg_L, num_strides_kin_L)
                    safe_strides_R = min(num_strides_emg_R, num_strides_kin_R)

                    # Debug Warning
                    if num_strides_emg_L != num_strides_kin_L:
                        print(f"  [Warn] {currTrial} Left: EMG({num_strides_emg_L}) vs Kin({num_strides_kin_L}). Using {safe_strides_L}.")
                    
                    if num_strides_emg_R != num_strides_kin_R:
                         print(f"  [Warn] {currTrial} Right: EMG({num_strides_emg_R}) vs Kin({num_strides_kin_R}). Using {safe_strides_R}.")

                    # =========================================================
                    # 3. INITIALIZE NUMPY ARRAYS
                    # =========================================================
                    # Dimensions: [Muscles, Strides, Time]
                    strideLeftEMG, strideRightEMG = [], []            
                    strideLeftMoment, strideRightMoment = [], []
                    strideLeftAngle, strideRightAngle = [], []

                    # =========================================================
                    # 4. LOAD EMG (SLICE TO SAFE LIMIT)
                    # =========================================================
                    
                    for currStride in range(safe_strides_L):
                        timeArray = np.array(eng.eval(f'data.{patient[:-4]}.{currTask}.{currTrial}.cycles.left.time(:,{currStride+1})', nargout=1))
                        time_duration = timeArray[-1] - timeArray[0]
                        new_sampleCount = int(time_duration * emgSampleRate)
                        old_points = np.linspace(0,1,3000)

                        new_points = np.linspace(0,1,new_sampleCount)
                        for e, currEMG in enumerate(EMGs):
                            if currEMG==0: continue
                            if e==1: 
                                size=np.array(eng.eval(f"size(data.{patient[:-4]}.{currTask}.{currTrial}.emg.left.L{currEMG})", nargout=1))
                                leftEMGData = np.zeros((len(EMGs),new_sampleCount))

                            l_data = np.array(eng.eval(f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.left.L{currEMG}(:,{currStride+1})", nargout=1)).flatten()
                            interpolator=interp1d(old_points,l_data)
                            l_data=interpolator(new_points)

                            # Transpose to get [Strides, Time], then slice to safe limit

                            if maxEMGsLeft[e]>0:
                                leftEMGData[e] = l_data/maxEMGsLeft[e]
                            else:
                                leftEMGData[e] = l_data


                        # =========================================================
                        # 5. PROCESS KINETICS (MOMENTS)
                        # =========================================================
                        for j, joint in enumerate(joints):
                            for num, currAxis in enumerate(axis):
                                raw_mom_L = np.array(eng.eval(f'data.{patient[:-4]}.{currTask}.{currTrial}.kinetics.jointmoment.left.{joint}.{currAxis}(:,{currStride+1})', nargout=1)).flatten()

                                if num==0 and j==0:
                                    leftMomentData = np.zeros((3, 3, raw_mom_L.shape[0]))
                                leftMomentData[j,num,:]=raw_mom_L


                        # =========================================================
                        # 6. PROCESS KINEMATICS (ANGLES)
                        # =========================================================
                        for a, angle in enumerate(angles):
                            for num, currAxis in enumerate(axis):
                                # Left Leg Logic
                                raw_ang_L = np.array(eng.eval(f'data.{patient[:-4]}.{currTask}.{currTrial}.kinematics.jointangles.left.{angle}.{currAxis}(:,{currStride+1})', nargout=1)).flatten()

                                if num==0 and a==0:
                                    leftAngleData = np.zeros((3, 3, raw_ang_L.shape[0]))
                                leftAngleData[a,num,:]=raw_ang_L
                                # Right Leg Logic
                        strideLeftEMG.append(leftEMGData)
                        strideLeftMoment.append(leftMomentData)
                        strideLeftAngle.append(leftAngleData)

                    
                    for currStride in range(safe_strides_R):
                        timeArray = np.array(eng.eval(f'data.{patient[:-4]}.{currTask}.{currTrial}.cycles.right.time(:,{currStride+1})', nargout=1))
                        time_duration = timeArray[-1] - timeArray[0]
                        new_sampleCount = int(time_duration * emgSampleRate)
                        new_points = np.linspace(0,1,new_sampleCount)
                        old_points = np.linspace(0,1,3000)

                        for e, currEMG in enumerate(EMGs):
                            if currEMG==0: continue
                            if e==1: 
                                size=np.array(eng.eval(f"size(data.{patient[:-4]}.{currTask}.{currTrial}.emg.right.R{currEMG})", nargout=1))
                                rightEMGData = np.zeros((len(EMGs), new_sampleCount))

                            r_data = np.array(eng.eval(f"data.{patient[:-4]}.{currTask}.{currTrial}.emg.right.R{currEMG}(:,{currStride+1})", nargout=1)).flatten()
                            interpolator = interp1d(old_points,r_data)
                            r_data = interpolator(new_points)
                            # Transpose to get [Strides, Time], then slice to safe limit

                            if maxEMGsRight[e]>0:
                                rightEMGData[e] = r_data/maxEMGsRight[e]
                            else:
                                rightEMGData[e] = r_data


                        # =========================================================
                        # 5. PROCESS KINETICS (MOMENTS)
                        # =========================================================
                        for j, joint in enumerate(joints):
                            for num, currAxis in enumerate(axis):
                                raw_mom_R = np.array(eng.eval(f'data.{patient[:-4]}.{currTask}.{currTrial}.kinetics.jointmoment.right.{joint}.{currAxis}(:,{currStride+1})', nargout=1)).flatten()

                                if num==0 and j==0:
                                    rightMomentData = np.zeros((3, 3, raw_mom_R.shape[0]))
                                rightMomentData[j,num,:]=raw_mom_R


                        # =========================================================
                        # 6. PROCESS KINEMATICS (ANGLES)
                        # =========================================================
                        for a, angle in enumerate(angles):
                            for num, currAxis in enumerate(axis):
                                # Left Leg Logic
                                raw_ang_R = np.array(eng.eval(f'data.{patient[:-4]}.{currTask}.{currTrial}.kinematics.jointangles.right.{angle}.{currAxis}(:,{currStride+1})', nargout=1)).flatten()

                                if num==0 and a==0:
                                    rightAngleData = np.zeros((3, 3, raw_ang_R.shape[0]))
                                rightAngleData[a,num,:]=raw_ang_R
                                # Right Leg Logic
                        strideRightMoment.append(rightMomentData)
                        strideRightEMG.append(rightEMGData)            
                        strideRightAngle.append(rightAngleData)

                    # Append trial data

                    trialLeftEMG.append(strideLeftEMG)
                    trialRightEMG.append(strideRightEMG)            
                    trialLeftMoment.append(strideLeftMoment)
                    trialRightMoment.append(strideRightMoment)
                    trialLeftAngle.append(strideLeftAngle)
                    trialRightAngle.append(strideRightAngle)

        # Append to Task Buckets (Aggregating all trials for this patient/task)
            if taskNum == 0: # Downhill
                patientDownRampLeftEMG.append(trialLeftEMG)
                patientDownRampLeftAngle.append(trialLeftAngle)
                patientDownRampLeftKinetic.append(trialLeftMoment)
                patientDownRampRightEMG.append(trialRightEMG)
                patientDownRampRightAngle.append(trialRightAngle)
                patientDownRampRightKinetic.append(trialRightMoment)

            elif taskNum == 1: # Uphill
                patientUpRampLeftEMG.append(trialLeftEMG)
                patientUpRampLeftAngle.append(trialLeftAngle)
                patientUpRampLeftKinetic.append(trialLeftMoment)
                patientUpRampRightEMG.append(trialRightEMG)
                patientUpRampRightAngle.append(trialRightAngle)
                patientUpRampRightKinetic.append(trialRightMoment)

            elif taskNum == 2: # Level
                patientWalkLeftEMG.append(trialLeftEMG)
                patientWalkLeftAngle.append(trialLeftAngle)
                patientWalkLeftKinetic.append(trialLeftMoment)
                patientWalkRightEMG.append(trialRightEMG)
                patientWalkRightAngle.append(trialRightAngle)
                patientWalkRightKinetic.append(trialRightMoment)

    # Return structured dictionary
    EMGMask = [1 if x != 0 else 0 for x in EMGs]
    jointMask = np.ones((3,3))
    return {
        'mask':{'right':{'emg':EMGMask,'kinetic':jointMask,'kinematic':jointMask},'left':{'emg':EMGMask,'kinetic':jointMask,'kinematic':jointMask}},
        'walk': {
            'right': { 'emg': patientWalkRightEMG, 'kinetic': patientWalkRightKinetic, 'kinematic': patientWalkRightAngle },
            'left':  { 'emg': patientWalkLeftEMG,  'kinetic': patientWalkLeftKinetic,  'kinematic': patientWalkLeftAngle }
        },
        'rampup': {
            'right': { 'emg': patientUpRampRightEMG, 'kinetic': patientUpRampRightKinetic, 'kinematic': patientUpRampRightAngle },
            'left':  { 'emg': patientUpRampLeftEMG,  'kinetic': patientUpRampLeftKinetic,  'kinematic': patientUpRampLeftAngle }
        },
        'rampdown': {
            'right': { 'emg': patientDownRampRightEMG, 'kinetic': patientDownRampRightKinetic, 'kinematic': patientDownRampRightAngle },
            'left':  { 'emg': patientDownRampLeftEMG,  'kinetic': patientDownRampLeftKinetic,  'kinematic': patientDownRampLeftAngle }
        }
    }

#NOTE EMGNet is a metadataset that uses
# which we have used: Bovi,Hu,Camargo,Lencioni
# not used or previously discard: Embry, Wang,Schulte

def main():
    print('hello')
    #processAngelidou()
    #moghadamDict=parseMoghadam()
    #print('problem 1 done')
    #siatDict=parseSIAT()

    #grimmerDict=parseGrimmer()#TODO
    #embryDict=parseEmbry()

    #huDict=parseHu()#DONE
    #bacekDict=parseBacek() #NOTE is there other than walking here?

    #camargoReturnDict=parseCamargo()
    angelidouDict=parseAngelidou()#TODO
    #UCIrvineDict=parseUCIrvine()
    #dictk2muse=parseK2Muse()
    #lencioniDict=parseLencioni()


    #returnMoreira=parseMoreira()

    #gait120Dict=parseGait120()
    #CriekingeDict=parseCriekinge()#check
    #macaDict=parseMacaluso()#check


    print("go time")

    save_dir = Path('D:/EMG/processed_datasets')
    save_dir.mkdir(exist_ok=True)

    datasets = {
        #'macaluso': macaDict,
        #'embry': embryDict,
        #'moghadam': moghadamDict,
        #'siat': siatDict,
        #'hu': huDict,
        #'grimmer': grimmerDict,

        #'bacek': bacekDict,

        #'camargo': camargoReturnDict,

        'angelidou': angelidouDict,
        #'ucirvine': UCIrvineDict,
        #'k2muse': dictk2muse,
        #'lencioni': lencioniDict,
        #'moreira': returnMoreira,
        #'gait120': gait120Dict,
        #'criekinge': CriekingeDict,
    }

    print("Saving datasets to D:/EMG/processed_datasets/\n")

    for name, data in datasets.items():
        pkl_path = save_dir / f'{name}.pkl'
        print(f"Saving {name}...", end=' ')
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
        
        file_size_mb = pkl_path.stat().st_size / (1024 * 1024)
        print(f"✓ ({file_size_mb:.2f} MB)")

    print(f"\nDone! All datasets saved to {save_dir}")

    #parseHunt()TODO

if __name__ == '__main__':
    main()
        