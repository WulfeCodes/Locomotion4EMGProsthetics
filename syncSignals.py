import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from scipy import signal
import pickle
import numpy as np
from collections import Counter


def resample_stride(stride_data, mask, target_points=200):
    """
    Resample stride kinematic/kinetic data to a fixed number of points with masking.
    
    Parameters:
    -----------
    stride_data : numpy.ndarray
        Input stride data with shape (3, 3, n_timepoints)
        Shape: (joint, dimension, time)
    mask : numpy.ndarray
        Integer mask with shape (3, 3) indicating which channels to process
        1 = process this channel, 0 = skip
    target_points : int, default=200
        Target number of points for the resampled stride
        
    Returns:
    --------
    resampled_data : numpy.ndarray
        Resampled data with shape (3, 3, target_points)
        Masked channels are filled with zeros
    """
    # Get original number of time points
    original_points = stride_data.shape[-1]
    
    # Initialize output array
    resampled_data = np.zeros((3, 3, target_points))
    
    # If already at target length and all channels masked, return zeros
    if original_points == target_points:
        for i in range(3):
            for j in range(3):
                if mask[i, j]:
                    resampled_data[i, j, :] = stride_data[i, j, :]
        return resampled_data
    
    # Create original and target time arrays
    original_time = np.linspace(0, 1, original_points)
    target_time = np.linspace(0, 1, target_points)
    
    # Process each channel according to mask
    for i in range(3):
        for j in range(3):
            if mask[i, j] == 1:
                # Use cubic interpolation for smooth resampling
                interpolator = interp1d(original_time, stride_data[i, j, :], 
                                       kind='cubic', fill_value='extrapolate')
                resampled_data[i, j, :] = interpolator(target_time)
            # else: remains zeros (already initialized)
    
    return np.array(resampled_data)

def resample_emg(emg_data, original_hz, target_hz=1000):
    """
    Resample EMG data from original frequency to target frequency.
    
    Parameters:
    -----------
    emg_data : np.array
        EMG data array (samples x channels)
    original_hz : float
        Original sampling frequency
    target_hz : float
        Target sampling frequency (default: 1000)
    
    Returns:
    --------
    np.array : Resampled EMG data
    """
    if original_hz == target_hz:
        return emg_data
    
    num_samples = int(len(emg_data) * target_hz / original_hz)
    resampled = signal.resample(emg_data, num_samples, axis=0)
    return np.array(resampled)

def create_gait_percentage_vector(emg_length):
    """
    Create a gait cycle percentage vector for EMG data.
    
    Parameters:
    -----------
    emg_length : int
        Number of samples in the EMG data
    
    Returns:
    --------
    np.array : Vector of percentages from 0 to 100
    """
    return np.array(np.linspace(0, 100, emg_length))

def resample_all_datasets(target_emgHz=1000, target_points=200, output_folder="D:/EMG/postprocessed_datasets"):
    """
    Resample kinematic/kinetic/EMG data in all dataset pickle files and save.
    
    Parameters:
    -----------
    target_emgHz : int
        Target EMG sampling frequency (default: 1000)
    target_points : int
        Number of points to resample to (default: 200)
    output_folder : str
        Folder to save resampled pickle files
    """
    
    os.makedirs(output_folder, exist_ok=True)
    
    def resample_criekinge(input_path="D:/EMG/processed_datasets/criekinge.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        directions = ['left', 'right', 'stroke']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetics']
        
        for currLeg in directions:
            new_angles = []
            new_kinetics = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle['walk'][currLeg]['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for stride_idx in range(len(currPickle['walk'][currLeg]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle['walk'][currLeg]['angle'][patient_idx][stride_idx])
                    patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                    
                    stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetics'][patient_idx][stride_idx])
                    patient_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                    
                    stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][stride_idx])
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(resampled_emg)
                    patient_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle['walk'][currLeg]['angle'] = new_angles
            currPickle['walk'][currLeg]['kinetics'] = new_kinetics
            currPickle['walk'][currLeg]['emg'] = new_emgs
            currPickle['walk'][currLeg]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "criekinge.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_moghadam(input_path="D:/EMG/processed_datasets/moghadam.pkl"):
        ORIGINAL_EMG_HZ = 100
        directions = ['left', 'right']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currLeg in directions:
            kinematicMask = currPickle['mask'][currLeg]['kinematic']
            kineticMask = currPickle['mask'][currLeg]['kinetic']
            
            new_kinematics = []
            new_kinetics = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle['walk'][currLeg]['kinematic'])):
                patient_kinematics = []
                patient_kinetics = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for trial_idx in range(len(currPickle['walk'][currLeg]['kinematic'][patient_idx])):
                    if len(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx]) == 0:
                        patient_kinematics.append([])
                        patient_kinetics.append([])
                        patient_emgs.append([])
                        patient_gait_percentages.append([])
                        continue
                    
                    trial_kinematics = []
                    trial_kinetics = []
                    trial_emgs = []
                    trial_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx])):
                        stride_kinematic = np.array(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx][stride_idx])
                        trial_kinematics.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetic'][patient_idx][trial_idx][stride_idx])
                        trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][trial_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        trial_emgs.append(resampled_emg)
                        trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                    
                    patient_kinematics.append(trial_kinematics)
                    patient_kinetics.append(trial_kinetics)
                    patient_emgs.append(trial_emgs)
                    patient_gait_percentages.append(trial_gait_percentages)
                
                new_kinematics.append(patient_kinematics)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle['walk'][currLeg]['kinematic'] = new_kinematics
            currPickle['walk'][currLeg]['kinetic'] = new_kinetics
            currPickle['walk'][currLeg]['emg'] = new_emgs
            currPickle['walk'][currLeg]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "moghadam.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_lencioni(input_path="D:/EMG/processed_datasets/lencioni.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        activities = ['step up', 'step down', 'walk']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetic']
        
        for currActivity in activities:
            new_angles = []
            new_kinetics = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle[currActivity]['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for stride_idx in range(len(currPickle[currActivity]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle[currActivity]['angle'][patient_idx][stride_idx])
                    patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                    
                    stride_kinetic = np.array(currPickle[currActivity]['kinetic'][patient_idx][stride_idx])
                    patient_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                    
                    stride_emg = np.array(currPickle[currActivity]['emg'][patient_idx][stride_idx])
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(resampled_emg)
                    patient_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle[currActivity]['angle'] = new_angles
            currPickle[currActivity]['kinetic'] = new_kinetics
            currPickle[currActivity]['emg'] = new_emgs
            currPickle[currActivity]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "lencioni.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_moreira(input_path="D:/EMG/processed_datasets/moreira.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        directions = ['left', 'right']
        activities = ['walk']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            
            for currActivity in activities:
                new_angles = []
                new_kinetics = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        trial_angles = []
                        trial_kinetics = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(resampled_emg)
                            trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                        
                        patient_angles.append(trial_angles)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "moreira.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_hu(input_path="D:/EMG/processed_datasets/hu.pkl"):
        ORIGINAL_EMG_HZ = 1000
        activities = ['walk', 'ramp_up', 'ramp_down', 'stair_up', 'stair_down']
        directions = ['left', 'right']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['masks'][currDirection]['angles']
            
            for currActivity in activities:
                new_angles = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])
                        patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        patient_emgs.append(resampled_emg)
                        patient_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                    
                    new_angles.append(patient_angles)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "hu.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_grimmer(input_path="D:/EMG/processed_datasets/grimmer.pkl"):
        ORIGINAL_EMG_HZ = 1111.1111
        activities = ['stairUp', 'stairDown']
        directions = ['left', 'right']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            
            for currActivity in activities:
                new_angles = []
                new_kinetics = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        trial_angles = []
                        trial_kinetics = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(resampled_emg)
                            trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                        
                        patient_angles.append(trial_angles)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "grimmer.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_siat(input_path="D:/EMG/processed_datasets/siat.pkl"):
        ORIGINAL_EMG_HZ = 1926
        activities = ['walk', 'stair_up', 'stair_down']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['masks']['left']['angle']
        kineticMask = currPickle['masks']['left']['kinetic']
        
        for activityType in activities:
            new_angles = []
            new_kinetics = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle[activityType]['left']['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for session_idx in range(len(currPickle[activityType]['left']['angle'][patient_idx])):
                    session_angles = []
                    session_kinetics = []
                    session_emgs = []
                    session_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle[activityType]['left']['angle'][patient_idx][session_idx])):
                        stride_kinematic = np.array(currPickle[activityType]['left']['angle'][patient_idx][session_idx][stride_idx])
                        session_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle[activityType]['left']['kinetic'][patient_idx][session_idx][stride_idx])
                        session_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        stride_emg = np.array(currPickle[activityType]['left']['emg'][patient_idx][session_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        session_emgs.append(resampled_emg)
                        session_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                    
                    patient_angles.append(session_angles)
                    patient_kinetics.append(session_kinetics)
                    patient_emgs.append(session_emgs)
                    patient_gait_percentages.append(session_gait_percentages)
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle[activityType]['left']['angle'] = new_angles
            currPickle[activityType]['left']['kinetic'] = new_kinetics
            currPickle[activityType]['left']['emg'] = new_emgs
            currPickle[activityType]['left']['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "siat.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_embry(input_path="D:/EMG/processed_datasets/embry.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        directions = ['left', 'right']
        activities = ['walk', 'rampup', 'rampdown']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['kinematic']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            
            for currActivity in activities:
                new_kinematics = []
                new_kinetics = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['kinematic'])):
                    patient_kinematics = []
                    patient_kinetics = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['kinematic'][patient_idx])):
                        trial_kinematics = []
                        trial_kinetics = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])
                            trial_kinematics.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(resampled_emg)
                            trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                        
                        patient_kinematics.append(trial_kinematics)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_kinematics.append(patient_kinematics)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['kinematic'] = new_kinematics
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "embry.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_gait120(input_path="D:/EMG/processed_datasets/gait120.pkl"):
        ORIGINAL_EMG_HZ = 1000
        activities = ['levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 'slopeDescent', 'sitToStand', 'standToSit']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['mask']['angle']
        
        for currActivity in activities:
            new_angles = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle['right'][currActivity]['angle'])):
                patient_angles = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for stride_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle['right'][currActivity]['angle'][patient_idx][stride_idx])
                    patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                    
                    stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][stride_idx])
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(resampled_emg)
                    patient_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                
                new_angles.append(patient_angles)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle['right'][currActivity]['angle'] = new_angles
            currPickle['right'][currActivity]['emg'] = new_emgs
            currPickle['right'][currActivity]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "gait120.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_camargo(input_path="D:/EMG/processed_datasets/camargo.pkl"):
        ORIGINAL_EMG_HZ = 1000
        activities = ['walk', 'stair', 'ramp']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetic']
        
        for currActivity in activities:
            new_angles = []
            new_kinetics = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle['right'][currActivity]['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for trial_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx])):
                    trial_angles = []
                    trial_kinetics = []
                    trial_emgs = []
                    trial_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx])):
                        stride_kinematic = np.array(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx][stride_idx])
                        trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle['right'][currActivity]['kinetic'][patient_idx][trial_idx][stride_idx])
                        trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][trial_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        trial_emgs.append(resampled_emg)
                        trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                    
                    patient_angles.append(trial_angles)
                    patient_kinetics.append(trial_kinetics)
                    patient_emgs.append(trial_emgs)
                    patient_gait_percentages.append(trial_gait_percentages)
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle['right'][currActivity]['angle'] = new_angles
            currPickle['right'][currActivity]['kinetic'] = new_kinetics
            currPickle['right'][currActivity]['emg'] = new_emgs
            currPickle['right'][currActivity]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "camargo.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")


    def resample_k2muse(input_path="D:/EMG/processed_datasets/k2muse.pkl"):
        ORIGINAL_EMG_HZ = 2000
        directions = ['right']
        activities = ['walk', 'up_ramp', 'down_ramp']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            
            for currActivity in activities:
                new_angles = []
                new_kinetics = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currDirection][currActivity]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for trial_idx in range(len(currPickle[currDirection][currActivity]['angle'][patient_idx])):
                        trial_angles = []
                        trial_kinetics = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for subtrial_idx in range(len(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx])):
                            subtrial_angles = []
                            subtrial_kinetics = []
                            subtrial_emgs = []
                            subtrial_gait_percentages = []
                            
                            for stride_idx in range(len(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx])):
                                stride_kinematic = np.array(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                                
                                stride_kinetic = np.array(currPickle[currDirection][currActivity]['kinetic'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                                
                                stride_emg = np.array(currPickle[currDirection][currActivity]['emg'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                                subtrial_emgs.append(resampled_emg)
                                subtrial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                            
                            trial_angles.append(subtrial_angles)
                            trial_kinetics.append(subtrial_kinetics)
                            trial_emgs.append(subtrial_emgs)
                            trial_gait_percentages.append(subtrial_gait_percentages)
                        
                        patient_angles.append(trial_angles)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currDirection][currActivity]['angle'] = new_angles
                currPickle[currDirection][currActivity]['kinetic'] = new_kinetics
                currPickle[currDirection][currActivity]['emg'] = new_emgs
                currPickle[currDirection][currActivity]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder,'k2muse.pkl')
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    def resample_macaluso(input_path="D:/EMG/processed_datasets/macaluso.pkl"):
        ORIGINAL_EMG_HZ = 1000
        activities = ['walk', 'rampup', 'rampdown']
        directions = ['right', 'left']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['kinematic']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            
            for currActivity in activities:
                new_kinematics = []
                new_kinetics = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['kinematic'])):
                    patient_kinematics = []
                    patient_kinetics = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['kinematic'][patient_idx])):
                        trial_kinematics = []
                        trial_kinetics = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])
                            trial_kinematics.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(resampled_emg)
                            trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                        
                        patient_kinematics.append(trial_kinematics)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_kinematics.append(patient_kinematics)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['kinematic'] = new_kinematics
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "macaluso.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    def resample_angelidou(input_path="D:/EMG/processed_datasets/angelidou.pkl"):
        ORIGINAL_EMG_HZ = 2000
        activities = ['walk']
        directions = ['left', 'right']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']  # Angelidou has EMG mask
            
            for currActivity in activities:
                new_angles = []
                new_kinetics = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])
                        patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][stride_idx])
                        patient_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        # EMG for Angelidou needs to be resampled using resample_stride with emgMask
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        patient_emgs.append(resampled_emg)
                        patient_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "angelidou.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    def resample_bacek(input_path="D:/EMG/processed_datasets/bacek.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Update this based on your actual EMG sampling rate
        activities = ['walk']
        directions = ['left', 'right']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        # Check if mask exists in the pickle, if not we'll need to infer it
        # Based on syncBacek, the mask assignment looks wrong - it's pointing to data
        # Let's check if there's a 'mask' key at the root level
        if 'mask' in currPickle:
            # Try to get mask from root level
            try:
                kinematicMask = currPickle['mask']['angle']
            except:
                # If that doesn't work, create a default mask of all 1s
                # We'll determine the shape from the first stride
                kinematicMask = None
        else:
            kinematicMask = None
        
        for currDirection in directions:
            for currActivity in activities:
                new_angles = []
                new_emgs = []
                new_gait_percentages = []
                
                # Iterate using zip like in syncBacek
                for currPatientEMG, currPatientKinematic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                currPickle[currActivity][currDirection]['angle']):
                    patient_angles = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for currTrialEMG, currTrialKinematic in zip(currPatientEMG, currPatientKinematic):
                        trial_angles = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for currStrideEMG, currStrideKinematic in zip(currTrialEMG, currTrialKinematic):
                            # Process kinematic data
                            stride_kinematic = np.array(currStrideKinematic)
                            
                            # Create mask if it doesn't exist (all ones - use all channels)
                            if kinematicMask is None:
                                kinematicMask = np.ones((stride_kinematic.shape[0], stride_kinematic.shape[1]))
                            
                            trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            # Process EMG data
                            stride_emg = np.array(currStrideEMG)
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(resampled_emg)
                            trial_gait_percentages.append(create_gait_percentage_vector(len(resampled_emg)))
                        
                        patient_angles.append(trial_angles)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_angles.append(patient_angles)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        output_path = os.path.join(output_folder, "bacek.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    # # Run all resampling functions
    resample_criekinge()
    resample_moghadam()
    resample_lencioni()
    resample_moreira()
    resample_hu()
    resample_grimmer()
    resample_siat()
    resample_embry()
    resample_gait120()
    resample_camargo()
    resample_k2muse()
    resample_macaluso()
    resample_angelidou()
    resample_bacek()
def main():
    print('hello')
    resample_all_datasets()

if __name__ == '__main__':
    main()