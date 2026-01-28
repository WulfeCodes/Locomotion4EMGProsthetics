import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from scipy import signal
import torch
from collections import Counter

def check_and_log_data_quality(data, data_type, activity, patient_idx, stride_idx, stats):
    """
    Check for NaN/Inf values and track statistics for a data array.
    
    Args:
        data: numpy array to check
        data_type: string identifier ('angle', 'kinetic', 'emg')
        activity: current activity name
        patient_idx: patient index
        stride_idx: stride index
        stats: statistics dictionary to update
    
    Returns:
        Updated stats dictionary
    """
    # Check for NaN
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        stats['nan_count'][data_type] += nan_count
        print(f"WARNING: {nan_count} NaNs in {data_type} - {activity}, patient {patient_idx}, stride {stride_idx}")
    
    # Check for Inf
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        stats['inf_count'][data_type] += inf_count
        print(f"WARNING: {inf_count} Infs in {data_type} - {activity}, patient {patient_idx}, stride {stride_idx}")
    
    # Track min/max statistics
    stats[data_type]['min'] = min(stats[data_type]['min'], np.nanmin(data))
    stats[data_type]['max'] = max(stats[data_type]['max'], np.nanmax(data))
    
    return stats

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
    
    num_samples = int(emg_data.shape[-1] * target_hz / original_hz)
    resampled = signal.resample(emg_data, num_samples, axis=1)
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

    def print_data_statistics(stats, dataset_name=""):
        """
        Print comprehensive statistics summary for all data modalities.
        
        Args:
            stats: statistics dictionary containing min/max/values/nan_count/inf_count
            dataset_name: name of the dataset for the header
        """
        print("\n" + "="*70)
        print(f"{dataset_name} DATA RESAMPLING SUMMARY")
        print("="*70)
        
        # NaN and Inf counts
        print(f"\nNaN Counts:")
        print(f"  Angles:   {stats['nan_count']['angle']}")
        print(f"  Kinetics: {stats['nan_count']['kinetic']}")
        print(f"  EMG:      {stats['nan_count']['emg']}")
        
        print(f"\nInf Counts:")
        print(f"  Angles:   {stats['inf_count']['angle']}")
        print(f"  Kinetics: {stats['inf_count']['kinetic']}")
        print(f"  EMG:      {stats['inf_count']['emg']}")
        
        print(f"\nAll-Zero Array Counts (possibly from aggressive filtering):")
        print(f"  Angles:   {stats['zero_count']['angle']}")
        print(f"  Kinetics: {stats['zero_count']['kinetic']}")
        print(f"  EMG:      {stats['zero_count']['emg']}")
        
        # Compute mean and std for each modality
        angle_all = np.concatenate(stats['angle']['values'])
        kinetic_all = np.concatenate(stats['kinetic']['values'])
        emg_all = np.concatenate(stats['emg']['values'])
        
        print(f"\nAngle Statistics:")
        print(f"  Original range (degrees): [{stats['angle']['min']:.4f}, {stats['angle']['max']:.4f}]")
        print(f"  Converted to radians:")
        print(f"    Min:  {np.nanmin(angle_all):.4f} ({np.rad2deg(np.nanmin(angle_all)):.4f}°)")
        print(f"    Max:  {np.nanmax(angle_all):.4f} ({np.rad2deg(np.nanmax(angle_all)):.4f}°)")
        print(f"    Mean: {np.nanmean(angle_all):.4f} ({np.rad2deg(np.nanmean(angle_all)):.4f}°)")
        print(f"    Std:  {np.nanstd(angle_all):.4f} ({np.rad2deg(np.nanstd(angle_all)):.4f}°)")
        
        print(f"\nKinetic (Force/Moment) Statistics:")
        print(f"  Min:  {stats['kinetic']['min']:.4f}")
        print(f"  Max:  {stats['kinetic']['max']:.4f}")
        print(f"  Mean: {np.nanmean(kinetic_all):.4f}")
        print(f"  Std:  {np.nanstd(kinetic_all):.4f}")
        
        print(f"\nEMG Statistics:")
        print(f"  Min:  {stats['emg']['min']:.6f}")
        print(f"  Max:  {stats['emg']['max']:.6f}")
        print(f"  Mean: {np.nanmean(emg_all):.6f}")
        print(f"  Std:  {np.nanstd(emg_all):.6f}")
        
        # Warnings
        if stats['kinetic']['max'] > 1000 or stats['kinetic']['min'] < -1000:
            print(f"\n⚠️  WARNING: Kinetic values are very large (range: {stats['kinetic']['min']:.1f} to {stats['kinetic']['max']:.1f})")
            print("   This will cause training instability. Consider normalization!")
        
        print(f"\n✓ Angles converted from degrees to radians")
        
        total_nans = sum(stats['nan_count'].values())
        total_infs = sum(stats['inf_count'].values())
        total_zeros = sum(stats['zero_count'].values())
        
        if total_nans > 0 or total_infs > 0:
            print(f"\n⚠️  WARNING: Found {total_nans} NaNs and {total_infs} Infs in data!")
            print("   These will cause training failures.")
        
        if total_zeros > 0:
            print(f"\n⚠️  WARNING: Found {total_zeros} all-zero arrays in data!")
            print("   These may indicate aggressive filtering or data collection issues.")
        
        print("="*70 + "\n")

    def is_data_in_degrees(data, threshold=10):
        """
        Heuristic to detect if angle data is in degrees or radians.
        
        Args:
            data: numpy array of angle values
            threshold: threshold in radians (default 10, ~572 degrees)
        
        Returns:
            True if data appears to be in degrees, False if in radians
        
        Logic:
            - Radians for joint angles typically range from -π to π (about -3.14 to 3.14)
            - Degrees for joint angles typically range from -180 to 180
            - If max absolute value > threshold (e.g., 10), likely degrees
        """
        max_abs_value = np.max(np.abs(data))
        return max_abs_value > threshold


    def resample_lencioni(input_path="D:/EMG/processed_datasets/lencioni.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        activities = ['step up', 'step down', 'walk']
        is_degree = False
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetic']
        emgMask = currPickle['mask']['emg']
        
        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }
        
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

                if patient_idx ==0:
                    for stride_idx in range(len(currPickle[currActivity]['angle'][patient_idx])):

                        if is_degree: break
                        is_degree=is_data_in_degrees(currPickle[currActivity]['angle'][patient_idx][stride_idx])

                for stride_idx in range(len(currPickle[currActivity]['angle'][patient_idx])):
                    # Process angles
                    stride_kinematic = np.array(currPickle[currActivity]['angle'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_kinematic, 'angle', currActivity, patient_idx, stride_idx, stats)
                    
                    # Convert degrees to radians
                    stride_kinematic_rad = np.deg2rad(stride_kinematic)
                    stats['angle']['values'].append(stride_kinematic_rad.flatten())
                    
                    resampled_angle = resample_stride(stride_kinematic_rad, kinematicMask, target_points)
                    patient_angles.append(torch.Tensor(resampled_angle).share_memory_())
                    
                    # Process kinetics
                    stride_kinetic = np.array(currPickle[currActivity]['kinetic'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_kinetic, 'kinetic', currActivity, patient_idx, stride_idx, stats)
                    stats['kinetic']['values'].append(stride_kinetic.flatten())
                    
                    resampled_kinetic = resample_stride(stride_kinetic, kineticMask, target_points)
                    patient_kinetics.append(torch.Tensor(resampled_kinetic).share_memory_())
                    
                    # Process EMG
                    stride_emg = np.array(currPickle[currActivity]['emg'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_emg, 'emg', currActivity, patient_idx, stride_idx, stats)
                    stats['emg']['values'].append(stride_emg.flatten())
                    
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                    patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle[currActivity]['angle'] = new_angles
            currPickle[currActivity]['kinetic'] = new_kinetics
            currPickle[currActivity]['emg'] = new_emgs
            currPickle[currActivity]['emg_gait_percentage'] = new_gait_percentages
        
        # Print comprehensive statistics
        print_data_statistics(stats, "LENCIONI")
        
        output_path = os.path.join(output_folder, "lencioni.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    def resample_grimmer(input_path="D:/EMG/processed_datasets/grimmer.pkl"):
        ORIGINAL_EMG_HZ = 1111.1111
        activities = ['stairUp', 'stairDown']
        directions = ['left', 'right']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }
        
        # Check if data is in degrees (sample first stride from first activity/direction)
        first_stride = np.array(currPickle[activities[0]][directions[0]]['angle'][0][0][0])
        data_is_degrees = is_data_in_degrees(first_stride)
        
        if data_is_degrees:
            print(f"✓GRIMMER Detected angles in DEGREES (max abs value: {np.max(np.abs(first_stride)):.2f})")
            print("  Converting to radians...")
        else:
            print(f"GRIMMER ✓ Detected angles in RADIANS (max abs value: {np.max(np.abs(first_stride)):.2f})")
            print("  No conversion needed.")
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']
            
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
                            # Process angles
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinematic, 'angle', currActivity, patient_idx, stride_idx, stats)
                            
                            # Convert to radians if needed
                            if data_is_degrees:
                                stride_kinematic = np.deg2rad(stride_kinematic)
                            
                            stats['angle']['values'].append(stride_kinematic.flatten())
                            trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                            
                            # Process kinetics
                            stride_kinetic = (currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinetic, 'kinetic', currActivity, patient_idx, stride_idx, stats)
                            stats['kinetic']['values'].append(stride_kinetic.flatten())
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                            
                            # Process EMG
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_emg, 'emg', currActivity, patient_idx, stride_idx, stats)
                            stats['emg']['values'].append(stride_emg.flatten())
                            
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor((resampled_emg)).share_memory_())
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                        
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
        
        # Print comprehensive statistics
        print_data_statistics(stats, "GRIMMER")
        
        output_path = os.path.join(output_folder, "grimmer.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    
    def resample_criekinge(input_path="D:/EMG/processed_datasets/criekinge.pkl"):
        ORIGINAL_EMG_HZ = 1000
        directions = ['left', 'right', 'stroke']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        emgMask = currPickle['mask']['emg']
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetics']
        
        nan_count = 0

        data_is_degrees = False
        
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

                if patient_idx==0:
                    for stride_idx in range(len(currPickle['walk'][currLeg]['angle'][patient_idx])):

                            if data_is_degrees: break
                            
                            first_stride = np.array(currPickle['walk'][currLeg]['angle'][patient_idx][stride_idx])
                            data_is_degrees = is_data_in_degrees(first_stride)
                
                for stride_idx in range(len(currPickle['walk'][currLeg]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle['walk'][currLeg]['angle'][patient_idx][stride_idx])
                    stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetics'][patient_idx][stride_idx])
                    stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][stride_idx])
                    
                    # Check for NaN
                    if np.isnan(stride_kinematic).any():
                        print(f"NaN in {currLeg} patient {patient_idx} stride {stride_idx}: kinematic")
                        nan_count += 1
                    if np.isnan(stride_kinetic).any():
                        print(f"NaN in {currLeg} patient {patient_idx} stride {stride_idx}: kinetic")
                        nan_count += 1
                    if np.isnan(stride_emg).any():
                        print(f"NaN in {currLeg} patient {patient_idx} stride {stride_idx}: EMG")
                        nan_count += 1
                    
                    # Clean and resample
                    stride_kinematic = np.nan_to_num(stride_kinematic, nan=0.0)
                    stride_kinetic = np.nan_to_num(stride_kinetic, nan=0.0)
                    stride_emg = np.nan_to_num(stride_emg, nan=0.0)
                    
                    if data_is_degrees:
                        patient_angles.append(torch.Tensor(resample_stride(np.deg2rad(stride_kinematic), kinematicMask, target_points)).share_memory_())
                    else:
                        patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())

                    patient_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                    patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle['walk'][currLeg]['angle'] = new_angles
            currPickle['walk'][currLeg]['kinetics'] = new_kinetics
            currPickle['walk'][currLeg]['emg'] = new_emgs
            currPickle['walk'][currLeg]['emg_gait_percentage'] = new_gait_percentages

        print(f"\nTotal NaN strides found: {nan_count}")
        
        output_path = os.path.join(output_folder, "criekinge.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_moghadam(input_path="D:/EMG/processed_datasets/moghadam.pkl"):
        ORIGINAL_EMG_HZ = 100
        directions = ['left', 'right']
        is_degree = False
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)

        is_degree = is_data_in_degrees(currPickle['walk'][directions[0]]['kinematic'][0][0][0])
        
        for currLeg in directions:
            kinematicMask = currPickle['mask'][currLeg]['kinematic']
            kineticMask = currPickle['mask'][currLeg]['kinetic']
            emgMask = currPickle['mask'][currLeg]['emg']
            
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
                        continue
                    
                    trial_kinematics = []
                    trial_kinetics = []
                    trial_emgs = []
                    trial_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx])):
                        stride_kinematic = np.array(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx][stride_idx])
                        
                        stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetic'][patient_idx][trial_idx][stride_idx])
                        trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())

                        if is_degree:
                            trial_kinematics.append(torch.Tensor(resample_stride(np.deg2rad(stride_kinematic), kinematicMask, target_points)).share_memory_())
                        else: 
                            trial_kinematics.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                        
                        stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][trial_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        trial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                        trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                    
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
    

    def resample_moreira(input_path="D:/EMG/processed_datasets/moreira.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        directions = ['left', 'right']
        activities = ['walk']
        is_degree = False
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']
            
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

                        if trial_idx==0:
                            for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                                if is_degree:
                                    break
                                is_degree=is_data_in_degrees(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                            if is_degree:
                                stride_kinematic = np.deg2rad(np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx]))
                            else: stride_kinematic =np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])

                            trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                        
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

        first_stride = currPickle[activities[0]][directions[0]]['angle'][0][0]

        is_degree=is_data_in_degrees(first_stride)
        
        for currDirection in directions:
            kinematicMask = currPickle['masks'][currDirection]['angles']
            emgMask = currPickle['masks'][currDirection]['emg']
            
            for currActivity in activities:
                new_angles = []
                new_emgs = []
                new_gait_percentages = []
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        if is_degree:
                            stride_kinematic = np.deg2rad(np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx]))
                        else: stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])

                        patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                        
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        patient_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                        patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                    
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
    

    
    def resample_siat(input_path="D:/EMG/processed_datasets/siat.pkl"):
        ORIGINAL_EMG_HZ = 1926
        activities = ['walk', 'stair_up', 'stair_down']
        
        with open(input_path, 'rb') as file:
            currPickle = pickle.load(file)
        
        kinematicMask = currPickle['masks']['left']['angle']
        kineticMask = currPickle['masks']['left']['kinetic']
        emgMask = currPickle['masks']['left']['emg']

        if is_data_in_degrees(currPickle[activities[0]]['left']['angle'][0][0][0]):
            is_degree = True
        else: is_degree = False
        
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
                        if is_degree: stride_kinematic = np.deg2rad(np.array(currPickle[activityType]['left']['angle'][patient_idx][session_idx][stride_idx]))
                        else:
                            stride_kinematic = np.array(currPickle[activityType]['left']['angle'][patient_idx][session_idx][stride_idx])

                        session_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                        
                        stride_kinetic = np.array(currPickle[activityType]['left']['kinetic'][patient_idx][session_idx][stride_idx])
                        session_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                        
                        stride_emg = np.array(currPickle[activityType]['left']['emg'][patient_idx][session_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        session_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                        session_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                    
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

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['kinematic'][0][0][0])):
            is_degree = True
        else: is_degree=False
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['kinematic']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']
            
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
                            if is_degree: stride_kinematic = np.deg2rad(np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx]))
                            else: 
                                stride_kinematic = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])
                            trial_kinematics.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                        
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
        emgMask = currPickle['mask']['emg']

        if is_data_in_degrees(np.array(currPickle['right'][activities[0]]['angle'][0][0])):
            is_degree = True
        else: is_degree = False
        
        for currActivity in activities:
            new_angles = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle['right'][currActivity]['angle'])):
                patient_angles = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for stride_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx])):
                    if is_degree:
                        stride_kinematic = np.deg2rad(np.array(currPickle['right'][currActivity]['angle'][patient_idx][stride_idx]))
                    else: stride_kinematic = np.array(currPickle['right'][currActivity]['angle'][patient_idx][stride_idx])

                    patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                    
                    stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][stride_idx])
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                    patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                
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
        emgMask = currPickle['mask']['emg']

        if is_data_in_degrees(np.array(currPickle['right'][activities[0]]['angle'][0][0][0])):
            is_degree = True
        else: is_degree = False
        
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
                        if is_degree:
                            stride_kinematic = np.deg2rad(np.array(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx][stride_idx]))
                        else:
                            stride_kinematic = np.array(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx][stride_idx])

                        trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                        
                        stride_kinetic = np.array(currPickle['right'][currActivity]['kinetic'][patient_idx][trial_idx][stride_idx])
                        trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                        
                        stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][trial_idx][stride_idx])
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        trial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                        trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                    
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

        if is_data_in_degrees(np.array(currPickle[directions[0]][activities[0]]['angle'][0][0][0][0])):
            is_degree = True
        
        else: 
            is_degree = False
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']
            
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
                                if is_degree:
                                    stride_kinematic = np.deg2rad(np.array(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx][stride_idx]))

                                else:
                                    stride_kinematic = np.array(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                                
                                stride_kinetic = np.array(currPickle[currDirection][currActivity]['kinetic'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                                
                                stride_emg = np.array(currPickle[currDirection][currActivity]['emg'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                                subtrial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                                subtrial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                            
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


        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['kinematic'][0][0][0])):
            is_degree =  True
        
        else:
            is_degree = False
        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['kinematic']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']
            
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
                            if is_degree:
                                stride_kinematic = np.deg2rad(np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx]))
                            else: stride_kinematic = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])

                            trial_kinematics.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])).share_memory_())
                        
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

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['angle'][0][0])):
            is_degree =  True
        
        else:
            is_degree = False

        
        for currDirection in directions:
            kinematicMask = currPickle['mask'][currDirection]['angle']
            kineticMask = currPickle['mask'][currDirection]['kinetic']
            emgMask = currPickle['mask'][currDirection]['emg']
            
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
                        if is_degree: 
                            stride_kinematic = np.deg2rad(np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx]))
                        else:
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])

                        patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                        
                        stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][stride_idx])
                        patient_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)).share_memory_())
                        
                        # EMG for Angelidou needs to be resampled using resample_stride with emgMask
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])

                        # resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        temp_emg=resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)

                        patient_emgs.append(torch.Tensor(temp_emg).share_memory_())
                        patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(temp_emg.shape[-1])).share_memory_())
                    
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

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['angle'][0][0][0])):
            is_degree = True
        
        else:
            is_degree = False

        
        # Check if mask exists in the pickle, if not we'll need to infer it
        # Based on syncBacek, the mask assignment looks wrong - it's pointing to data
        # Let's check if there's a 'mask' key at the root level

        emgMask = np.array(currPickle['mask']['right']['emg'])
        kinematicMask = np.array(currPickle['mask']['right']['angle'])

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
                            if is_degree:
                                stride_kinematic = np.deg2rad(np.array(currStrideKinematic))
                            else: stride_kinematic = np.array(currStrideKinematic)

                            # Create mask if it doesn't exist (all ones - use all channels)
                            if kinematicMask is None:
                                kinematicMask = np.ones((stride_kinematic.shape[0], stride_kinematic.shape[1]))
                            
                            trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)).share_memory_())
                            
                            # Process EMG data
                            stride_emg = np.array(currStrideEMG)
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)

                            trial_emgs.append(torch.Tensor(resampled_emg).share_memory_())
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[1])).share_memory_())
                            
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
    #Run all resampling functions
    resample_lencioni()
    resample_moreira()
    resample_hu()
    resample_siat()
    resample_embry()
    resample_gait120()
    resample_camargo()
    resample_k2muse()
    resample_macaluso()
    resample_angelidou()
    #resample_moghadam()
    #resample_grimmer()
    #resample_criekinge()
    # resample_bacek()
def main():
    print('hello')
    resample_all_datasets()

if __name__ == '__main__':
    main()