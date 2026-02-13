import pickle
import os
import pandas as pd
import numpy as np
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
from scipy import signal
import torch
import matplotlib.pyplot as plt
from collections import Counter

def check_and_log_data_quality(data, data_type, activity, patient_idx, stride_idx, stats, 
                                 joint_names=None, threshold=3.5):
    """
    Check for NaN/Inf values and track statistics for a data array.
    
    Args:
        data: numpy array to check (shape: [3, 3, timesteps] for kinetic)
        data_type: string identifier ('angle', 'kinetic', 'emg')
        activity: current activity name
        patient_idx: patient index
        stride_idx: stride index
        stats: statistics dictionary to update
        joint_names: list of joint/feature names (optional, for better logging)
        threshold: threshold for flagging extreme values (default 3.5 for kinetics in Nm/kg)
    
    Returns:
        Updated stats dictionary
    """
    # Check for NaN
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        stats['nan_count'][data_type] += nan_count
    
    # Check for Inf
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        stats['inf_count'][data_type] += inf_count

    # Track min/max statistics
    current_min = np.nanmin(data)
    current_max = np.nanmax(data)
    
    stats[data_type]['min'] = min(stats[data_type]['min'], current_min)
    stats[data_type]['max'] = max(stats[data_type]['max'], current_max)
    
    # Track extreme values with their sources
    if data_type == 'kinetic':
        # Check if current max exceeds threshold
        if abs(current_max) > threshold or abs(current_min) > threshold:
            if 'extreme_values' not in stats[data_type]:
                stats[data_type]['extreme_values'] = []
            
            # 3D data: (3, 3, timesteps)
            max_idx = np.unravel_index(np.nanargmax(np.abs(data)), data.shape)
            extreme_val = data[max_idx]
            joint_dim1, joint_dim2, timestep = max_idx
            
            if joint_names and len(joint_names) > joint_dim1:
                feature_name = f"{joint_names[joint_dim1]}_dim{joint_dim2}"
            else:
                feature_name = f"Joint_{joint_dim1}_dim{joint_dim2}"
            
            # Log the extreme value
            stats[data_type]['extreme_values'].append({
                'value': float(extreme_val),
                'patient': patient_idx,
                'activity': activity,
                'stride': stride_idx,
                'feature': feature_name,
                'timestep': int(timestep),
                'threshold_exceeded': abs(extreme_val) > threshold
            })
    
    return stats

def report_extreme_values(stats, top_n=100):
    """
    Generate a report of extreme kinetic values found in the dataset.
    
    Args:
        stats: statistics dictionary from data quality checks
        top_n: number of top extreme values to report
    """
    if 'extreme_values' not in stats.get('kinetic', {}):
        print("No extreme values tracked.")
        return
    
    extreme_vals = stats['kinetic']['extreme_values']
    
    if not extreme_vals:
        print("No values exceeded the threshold.")
        return
    
    # Sort by absolute value
    sorted_extremes = sorted(extreme_vals, key=lambda x: abs(x['value']), reverse=True)
    
    print(f"\n{'='*80}")
    print(f"EXTREME KINETIC VALUES REPORT (Threshold: 3.5 Nm/kg)")
    print(f"{'='*80}")
    print(f"Total extreme values found: {len(sorted_extremes)}")
    print(f"\nTop {min(top_n, len(sorted_extremes))} extreme values:\n")
    
    for i, record in enumerate(sorted_extremes[:top_n], 1):
        print(f"{i}. Value: {record['value']:.4f} Nm/kg")
        print(f"   Patient: {record['patient']} | Activity: {record['activity']} | Stride: {record['stride']}")
        print(f"   Joint/Feature: {record['feature']} | Timestep: {record['timestep']}")
        print()
    
    # NEW: Group by unique (patient, stride, activity) combinations
    print(f"\n{'='*80}")
    print("UNIQUE PATIENT-STRIDE-ACTIVITY COMBINATIONS WITH EXTREME VALUES:")
    print(f"{'='*80}")
    
    # Dictionary to store all instances for each unique combination
    unique_combos = {}
    
    for record in sorted_extremes:
        combo_key = (record['patient'], record['stride'], record['activity'])
        if combo_key not in unique_combos:
            unique_combos[combo_key] = []
        unique_combos[combo_key].append(record)
    
    print(f"\nTotal unique combinations: {len(unique_combos)}\n")
    
    # Sort combinations by the maximum absolute value in each combination
    sorted_combos = sorted(unique_combos.items(), 
                          key=lambda x: max(abs(r['value']) for r in x[1]), 
                          reverse=True)
    
    for combo_idx, ((patient, stride, activity), records) in enumerate(sorted_combos, 1):
        print(f"\n{combo_idx}. Patient: {patient} | Stride: {stride} | Activity: {activity}")
        print(f"   {'─'*76}")
        print(f"   Number of extreme values in this combination: {len(records)}")
        
        # Sort records within this combination by absolute value
        sorted_records = sorted(records, key=lambda x: abs(x['value']), reverse=True)
        
        # Show all instances for this combination
        for instance_idx, record in enumerate(sorted_records, 1):
            print(f"   {instance_idx}) Value: {record['value']:>8.4f} Nm/kg | "
                  f"Feature: {record['feature']:30s} | Timestep: {record['timestep']:3d}")
        print()
    
    # Summary by activity
    print(f"\n{'='*80}")
    print("BREAKDOWN BY ACTIVITY:")
    print(f"{'='*80}")
    activity_counts = {}
    for record in sorted_extremes:
        activity = record['activity']
        activity_counts[activity] = activity_counts.get(activity, 0) + 1
    
    for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{activity}: {count} extreme values")
    
    # Summary by feature/joint
    print(f"\n{'='*80}")
    print("BREAKDOWN BY JOINT/FEATURE:")
    print(f"{'='*80}")
    feature_counts = {}
    for record in sorted_extremes:
        feature = record['feature']
        feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {count} extreme values")
    
    # NEW: Additional summary - combinations per activity
    print(f"\n{'='*80}")
    print("UNIQUE COMBINATIONS PER ACTIVITY:")
    print(f"{'='*80}")
    combos_per_activity = {}
    for (patient, stride, activity), records in unique_combos.items():
        if activity not in combos_per_activity:
            combos_per_activity[activity] = set()
        combos_per_activity[activity].add((patient, stride))
    
    for activity, combo_set in sorted(combos_per_activity.items(), 
                                     key=lambda x: len(x[1]), reverse=True):
        print(f"{activity}: {len(combo_set)} unique patient-stride combinations")

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
        if dataset_name.lower() != 'hu' and dataset_name.lower() != 'siat' and dataset_name.lower() != 'bacek' and dataset_name.lower() != 'gait120':
            kinetic_all = np.concatenate(stats['kinetic']['values'])
        emg_all = np.concatenate(stats['emg']['values'])
        
        print(f"\nAngle Statistics:")
        print(f"  Original range (degrees): [{stats['angle']['min']:.4f}, {stats['angle']['max']:.4f}]")
        print(f"  Converted to radians:")
        print(f"    Min:  {np.nanmin(angle_all):.4f} ({np.rad2deg(np.nanmin(angle_all)):.4f}°)")
        print(f"    Max:  {np.nanmax(angle_all):.4f} ({np.rad2deg(np.nanmax(angle_all)):.4f}°)")
        print(f"    Mean: {np.nanmean(angle_all):.4f} ({np.rad2deg(np.nanmean(angle_all)):.4f}°)")
        print(f"    Std:  {np.nanstd(angle_all):.4f} ({np.rad2deg(np.nanstd(angle_all)):.4f}°)")

        if dataset_name.lower() != 'hu' and dataset_name.lower() != 'siat' and dataset_name.lower() != 'bacek' and dataset_name.lower() != 'gait120':
            print(f"\nKinetic (Force/Moment) Statistics:", dataset_name,dataset_name.lower() != 'hu')
            print(f"  Min:  {stats['kinetic']['min']:.4f}")
            print(f"  Max:  {stats['kinetic']['max']:.4f}")
            print(f"  Mean: {np.nanmean(kinetic_all):.4f}")
            print(f"  Std:  {np.nanstd(kinetic_all):.4f}")
        
        print(f"\nEMG Statistics:")
        print(f"  Min:  {stats['emg']['min']:.6f}")
        print(f"  Max:  {stats['emg']['max']:.6f}")
        if dataset_name.lower() != 'bacek:':
            print(f"  Mean: {np.nanmean(emg_all):.6f}")
            print(f"  Std:  {np.nanstd(emg_all):.6f}")
        
        # Warnings
        if stats['kinetic']['max'] > 1000 or stats['kinetic']['min'] < -1000:
            print(f"\n WARNING: Kinetic values are very large (range: {stats['kinetic']['min']:.1f} to {stats['kinetic']['max']:.1f})")
            print("   This will cause training instability. Consider normalization!")
        
        print(f"\n✓ Angles converted from degrees to radians")
        
        total_nans = sum(stats['nan_count'].values())
        total_infs = sum(stats['inf_count'].values())
        total_zeros = sum(stats['zero_count'].values())
        
        if total_nans > 0 or total_infs > 0:
            print(f"\n WARNING: Found {total_nans} NaNs and {total_infs} Infs in data!")
            print("   These will cause training failures.")
        
        if total_zeros > 0:
            print(f"\nWARNING: Found {total_zeros} all-zero arrays in data!")
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
                    patient_angles.append(torch.Tensor(resampled_angle))
                    
                    # Process kinetics
                    stride_kinetic = np.array(currPickle[currActivity]['kinetic'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_kinetic, 'kinetic', currActivity, patient_idx, stride_idx, stats)
                    stats['kinetic']['values'].append(stride_kinetic.flatten())
                    
                    resampled_kinetic = resample_stride(stride_kinetic, kineticMask, target_points)
                    patient_kinetics.append(torch.Tensor(resampled_kinetic))
                    
                    # Process EMG
                    stride_emg = np.array(currPickle[currActivity]['emg'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_emg, 'emg', currActivity, patient_idx, stride_idx, stats)
                    stats['emg']['values'].append(stride_emg.flatten())
                    
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(torch.Tensor(resampled_emg))
                    patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                
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
                            trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                            
                            # Process kinetics
                            stride_kinetic = (currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            if np.max(stride_kinetic)>3:
                                print('max found',np.max(stride_kinetic),patient_idx,stride_idx)
                                print('patient',patient_idx,'stride',stride_idx)
                                continue
                                # for z in range(3):
                                #     for y in range(3):
                                #         if stride_kinetic[z,y].any() and np.max(stride_kinetic[z,y]>3.5):
                                #             print(z,y,patient_idx,stride_idx,trial_idx,currDirection,currActivity)
                                #             plt.plot(stride_kinetic[z,y])
                                #             plt.show()
                            check_and_log_data_quality(stride_kinetic, 'kinetic', currActivity, patient_idx, stride_idx, stats)
                            stats['kinetic']['values'].append(stride_kinetic.flatten())
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                            
                            # Process EMG
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_emg, 'emg', currActivity, patient_idx, stride_idx, stats)
                            stats['emg']['values'].append(stride_emg.flatten())
                            
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor((resampled_emg)))
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                        
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
        report_extreme_values(stats)
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
                        nan_count += 1
                    if np.isnan(stride_kinetic).any():
                        nan_count += 1
                    if np.isnan(stride_emg).any():
                        nan_count += 1
                    
                    # Clean and resample
                    stride_kinematic = np.nan_to_num(stride_kinematic, nan=0.0)
                    stride_kinetic = np.nan_to_num(stride_kinetic, nan=0.0)
                    stride_emg = np.nan_to_num(stride_emg, nan=0.0)
                    
                    if data_is_degrees:
                        patient_angles.append(torch.Tensor(resample_stride(np.deg2rad(stride_kinematic), kinematicMask, target_points)))
                    else:
                        patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))

                    patient_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(torch.Tensor(resampled_emg))
                    patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                
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

        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }

        # Safely find first available kinematic data to check if degrees or radians
        sample_data = None
        for direction in directions:
            kinematic_data = currPickle['walk'][direction]['kinematic']
            for patient_data in kinematic_data:
                if len(patient_data) == 0:
                    continue
                for trial_data in patient_data:
                    if len(trial_data) == 0:
                        continue
                    for stride_data in trial_data:
                        if len(stride_data) > 0:
                            sample_data = stride_data
                            break
                    if sample_data is not None:
                        break
                if sample_data is not None:
                    break
            if sample_data is not None:
                break
        
        if sample_data is None:
            raise ValueError("No kinematic data found in the dataset!")
        
        is_degree = is_data_in_degrees(sample_data)
        
        if is_degree:
            print(f"✓ MOGHADAM Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            print(f"✓ MOGHADAM Detected angles in RADIANS")
            print("  No conversion needed.")
        
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
                
                for stride_idx in range(len(currPickle['walk'][currLeg]['kinematic'][patient_idx])):
                    if len(currPickle['walk'][currLeg]['kinematic'][patient_idx][stride_idx]) == 0:
                        print('tf??')
                        continue
                    
                    trial_kinematics = []
                    trial_kinetics = []
                    trial_emgs = []
                    trial_gait_percentages = []
                    
                    # Process kinematics (angles)
                    stride_kinematic = np.array(currPickle['walk'][currLeg]['kinematic'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_kinematic, 'angle', f'walk-{currLeg}', patient_idx, stride_idx, stats)
                    
                    # Convert to radians if needed
                    if is_degree:
                        stride_kinematic_rad = np.deg2rad(stride_kinematic)
                    else:
                        stride_kinematic_rad = stride_kinematic
                    
                    stats['angle']['values'].append(stride_kinematic_rad.flatten())
                    trial_kinematics.append(torch.Tensor(resample_stride(stride_kinematic_rad, kinematicMask, target_points)))
                    
                    # Process kinetics
                    stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetic'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_kinetic, 'kinetic', f'walk-{currLeg}', patient_idx, stride_idx, stats)
                    stats['kinetic']['values'].append(stride_kinetic.flatten())
                    trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                    
                    # Process EMG
                    stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_emg, 'emg', f'walk-{currLeg}', patient_idx, stride_idx, stats)
                    stats['emg']['values'].append(stride_emg.flatten())
                    
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    trial_emgs.append(torch.Tensor(resampled_emg))
                    trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                    
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

        # Print comprehensive statistics
        print_data_statistics(stats, "MOGHADAM")
        
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
        
        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }
        
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

                        if trial_idx == 0 and patient_idx == 0:
                            for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                                if is_degree:
                                    break
                                is_degree = is_data_in_degrees(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            
                            if is_degree:
                                print(f"✓ MOREIRA Detected angles in DEGREES")
                                print("  Converting to radians...")
                            else:
                                print(f"✓ MOREIRA Detected angles in RADIANS")
                                print("  No conversion needed.")
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                            # Process angles
                            stride_kinematic_raw = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            
                            if is_degree:
                                stride_kinematic = np.deg2rad(stride_kinematic_raw)
                            else:
                                stride_kinematic = stride_kinematic_raw
                            
                            stats['angle']['values'].append(stride_kinematic.flatten())
                            trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                            
                            # Process kinetics
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinetic, 'kinetic', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['kinetic']['values'].append(stride_kinetic.flatten())
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                            
                            # Process EMG
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['emg']['values'].append(stride_emg.flatten())
                            
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor(resampled_emg))
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                        
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
        print_data_statistics(stats, "MOREIRA")
        
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

        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }

        first_stride = currPickle[activities[0]][directions[0]]['angle'][0][0]
        is_degree = is_data_in_degrees(first_stride)
        
        if is_degree:
            print(f"✓ HU Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            print(f"✓ HU Detected angles in RADIANS")
            print("  No conversion needed.")
        
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
                        # Process angles
                        stride_kinematic_raw = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])
                        check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                        
                        if is_degree:
                            stride_kinematic = np.deg2rad(stride_kinematic_raw)
                        else:
                            stride_kinematic = stride_kinematic_raw
                        
                        stats['angle']['values'].append(stride_kinematic.flatten())
                        patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                        
                        # Process EMG
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])
                        check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                        stats['emg']['values'].append(stride_emg.flatten())
                        
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        patient_emgs.append(torch.Tensor(resampled_emg))
                        patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                    
                    new_angles.append(patient_angles)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages

        # Print comprehensive statistics
        print_data_statistics(stats, "hu")

        output_path = os.path.join(output_folder, "hu.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_siat(input_path="D:/EMG/processed_datasets/siat.pkl"):
        ORIGINAL_EMG_HZ = 1926
        activities = ['walk', 'stair_up', 'stair_down']
        
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
        
        kinematicMask = currPickle['masks']['left']['angle']
        kineticMask = currPickle['masks']['left']['kinetic']
        emgMask = currPickle['masks']['left']['emg']

        if is_data_in_degrees(currPickle[activities[0]]['left']['angle'][0][0][0]):
            is_degree = True
            print(f"✓ SIAT Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ SIAT Detected angles in RADIANS")
            print("  No conversion needed.")
        
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
                        # Process angles
                        stride_kinematic_raw = np.array(currPickle[activityType]['left']['angle'][patient_idx][session_idx][stride_idx])
                        check_and_log_data_quality(stride_kinematic_raw, 'angle', activityType, patient_idx, stride_idx, stats)
                        
                        if is_degree:
                            stride_kinematic = np.deg2rad(stride_kinematic_raw)
                        else:
                            stride_kinematic = stride_kinematic_raw
                        
                        stats['angle']['values'].append(stride_kinematic.flatten())
                        session_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                        
                        # Process kinetics
                        stride_kinetic = np.array(currPickle[activityType]['left']['kinetic'][patient_idx][session_idx][stride_idx])
                        check_and_log_data_quality(stride_kinetic, 'kinetic', activityType, patient_idx, stride_idx, stats)
                        stats['kinetic']['values'].append(stride_kinetic.flatten())
                        session_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                        
                        # Process EMG
                        stride_emg = np.array(currPickle[activityType]['left']['emg'][patient_idx][session_idx][stride_idx])
                        check_and_log_data_quality(stride_emg, 'emg', activityType, patient_idx, stride_idx, stats)
                        stats['emg']['values'].append(stride_emg.flatten())
                        
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        session_emgs.append(torch.Tensor(resampled_emg))
                        session_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                    
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

        # Print comprehensive statistics
        print_data_statistics(stats, "SIAT")

        output_path = os.path.join(output_folder, "siat.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_embry(input_path="D:/EMG/processed_datasets/embry.pkl"):
        ORIGINAL_EMG_HZ = 1000  # Already processed at 1000Hz
        directions = ['left', 'right']
        activities = ['rampup', 'rampdown'] #walk
        
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

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['kinematic'][0][0][0])):
            is_degree = True
            print(f"✓ EMBRY Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ EMBRY Detected angles in RADIANS")
            print("  No conversion needed.")
        
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
                            # Process kinematics (angles)
                            stride_kinematic_raw = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            
                            if is_degree:
                                stride_kinematic = np.deg2rad(stride_kinematic_raw)
                            else:
                                stride_kinematic = stride_kinematic_raw
                            
                            stats['angle']['values'].append(stride_kinematic.flatten())
                            trial_kinematics.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                            
                            # Process kinetics
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinetic, 'kinetic', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['kinetic']['values'].append(stride_kinetic.flatten())
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                            
                            # Process EMG
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['emg']['values'].append(stride_emg.flatten())
                            
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor(resampled_emg))
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                        
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

        # Print comprehensive statistics
        print_data_statistics(stats, "EMBRY")

        output_path = os.path.join(output_folder, "embry.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    
    def resample_gait120(input_path="D:/EMG/processed_datasets/gait120.pkl"):
        ORIGINAL_EMG_HZ = 1000
        activities = ['levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 'slopeDescent', 'sitToStand', 'standToSit']
        
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
        
        kinematicMask = currPickle['mask']['angle']
        emgMask = currPickle['mask']['emg']

        if is_data_in_degrees(np.array(currPickle['right'][activities[0]]['angle'][0][0])):
            is_degree = True
            print(f"✓ GAIT120 Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ GAIT120 Detected angles in RADIANS")
            print("  No conversion needed.")
        
        for currActivity in activities:
            new_angles = []
            new_emgs = []
            new_gait_percentages = []
            
            for patient_idx in range(len(currPickle['right'][currActivity]['angle'])):
                patient_angles = []
                patient_emgs = []
                patient_gait_percentages = []
                
                for stride_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx])):
                    # Process angles
                    stride_kinematic_raw = np.array(currPickle['right'][currActivity]['angle'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_kinematic_raw, 'angle', currActivity, patient_idx, stride_idx, stats)
                    
                    if is_degree:
                        stride_kinematic = np.deg2rad(stride_kinematic_raw)
                    else:
                        stride_kinematic = stride_kinematic_raw
                    
                    stats['angle']['values'].append(stride_kinematic.flatten())
                    patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                    
                    # Process EMG
                    stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][stride_idx])
                    check_and_log_data_quality(stride_emg, 'emg', currActivity, patient_idx, stride_idx, stats)
                    stats['emg']['values'].append(stride_emg.flatten())
                    
                    resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                    patient_emgs.append(torch.Tensor(resampled_emg))
                    patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                
                new_angles.append(patient_angles)
                new_emgs.append(patient_emgs)
                new_gait_percentages.append(patient_gait_percentages)
            
            currPickle['right'][currActivity]['angle'] = new_angles
            currPickle['right'][currActivity]['emg'] = new_emgs
            currPickle['right'][currActivity]['emg_gait_percentage'] = new_gait_percentages

        # Print comprehensive statistics
        print_data_statistics(stats, "GAIT120")

        output_path = os.path.join(output_folder, "gait120.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    
    def resample_camargo(input_path="D:/EMG/processed_datasets/camargo.pkl"):
        ORIGINAL_EMG_HZ = 1000
        activities = ['walk', 'stair', 'ramp']
        
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
        
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetic']
        emgMask = currPickle['mask']['emg']

        if is_data_in_degrees(np.array(currPickle['right'][activities[0]]['angle'][0][0][0])):
            is_degree = True
            print(f"✓ CAMARGO Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ CAMARGO Detected angles in RADIANS")
            print("  No conversion needed.")
        
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
                        # Process angles
                        stride_kinematic_raw = np.array(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx][stride_idx])
                        check_and_log_data_quality(stride_kinematic_raw, 'angle', currActivity, patient_idx, stride_idx, stats)
                        
                        if is_degree:
                            stride_kinematic = np.deg2rad(stride_kinematic_raw)
                        else:
                            stride_kinematic = stride_kinematic_raw
                        
                        stats['angle']['values'].append(stride_kinematic.flatten())
                        trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                        
                        # Process kinetics
                        stride_kinetic = np.array(currPickle['right'][currActivity]['kinetic'][patient_idx][trial_idx][stride_idx])
                        check_and_log_data_quality(stride_kinetic, 'kinetic', currActivity, patient_idx, stride_idx, stats)
                        stats['kinetic']['values'].append(stride_kinetic.flatten())
                        trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                        
                        # Process EMG
                        stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][trial_idx][stride_idx])
                        check_and_log_data_quality(stride_emg, 'emg', currActivity, patient_idx, stride_idx, stats)
                        stats['emg']['values'].append(stride_emg.flatten())
                        
                        resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                        trial_emgs.append(torch.Tensor(resampled_emg))
                        trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                    
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

        # Print comprehensive statistics
        report_extreme_values(stats)
        print_data_statistics(stats, "CAMARGO")

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


        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }

        if is_data_in_degrees(np.array(currPickle[directions[0]][activities[0]]['angle'][0][0][0][0])):
            is_degree = True
            print(f"✓ K2MUSE Detected angles in DEGREES")
            print("  Converting to radians...")
        else: 
            is_degree = False
            print(f"✓ K2MUSE Detected angles in RADIANS")
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
                                # Process angles
                                stride_kinematic_raw = np.array(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                                
                                if is_degree:
                                    stride_kinematic = np.deg2rad(stride_kinematic_raw)
                                else:
                                    stride_kinematic = stride_kinematic_raw
                                

                                # Process kinetics
                                stride_kinetic = np.array(currPickle[currDirection][currActivity]['kinetic'][patient_idx][trial_idx][subtrial_idx][stride_idx])

                                if np.max(stride_kinetic) > 3.7: 
                                    print('OOD example found:',np.max(stride_kinetic))
                                    continue

                                check_and_log_data_quality(stride_kinetic, 'kinetic', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)

                                stats['kinetic']['values'].append(stride_kinetic.flatten())
                                subtrial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                                
                                stats['angle']['values'].append(stride_kinematic.flatten())
                                subtrial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                                
                                # Process EMG
                                stride_emg = np.array(currPickle[currDirection][currActivity]['emg'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                                stats['emg']['values'].append(stride_emg.flatten())
                                
                                resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                                subtrial_emgs.append(torch.Tensor(resampled_emg))
                                subtrial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                            
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

        # Print comprehensive statistics
        report_extreme_values(stats)
        print_data_statistics(stats, "K2MUSE")

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

        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['kinematic'][0][0][0])):
            is_degree = True
            print(f"✓ MACALUSO Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ MACALUSO Detected angles in RADIANS")
            print("  No conversion needed.")
        
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
                            # Process kinematics (angles)
                            stride_kinematic_raw = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            
                            if is_degree:
                                stride_kinematic = np.deg2rad(stride_kinematic_raw)
                            else:
                                stride_kinematic = stride_kinematic_raw
                            
                            stats['angle']['values'].append(stride_kinematic.flatten())
                            trial_kinematics.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                            
                            # Process kinetics
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_kinetic, 'kinetic', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['kinetic']['values'].append(stride_kinetic.flatten())
                            trial_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                            
                            # Process EMG
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['emg']['values'].append(stride_emg.flatten())
                            
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)
                            trial_emgs.append(torch.Tensor(resampled_emg))
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[-1])))
                        
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
        
        # Print comprehensive statistics
        print_data_statistics(stats, "MACALUSO")
        
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

        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['angle'][0][0])):
            is_degree = True
            print(f"✓ ANGELIDOU Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ ANGELIDOU Detected angles in RADIANS")
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
                    
                    for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        # Process angles
                        stride_kinematic_raw = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])
                        check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                        
                        if is_degree: 
                            stride_kinematic = np.deg2rad(stride_kinematic_raw)
                        else:
                            stride_kinematic = stride_kinematic_raw
                        
                        stats['angle']['values'].append(stride_kinematic.flatten())
                        patient_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                        
                        # Process kinetics
                        stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][stride_idx])
                        check_and_log_data_quality(stride_kinetic, 'kinetic', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                        stats['kinetic']['values'].append(stride_kinetic.flatten())
                        patient_kinetics.append(torch.Tensor(resample_stride(stride_kinetic, kineticMask, target_points)))
                        
                        # Process EMG
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])
                        check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                        stats['emg']['values'].append(stride_emg.flatten())
                        
                        # EMG for Angelidou needs to be resampled using resample_stride with emgMask
                        temp_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)

                        patient_emgs.append(torch.Tensor(temp_emg))
                        patient_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(temp_emg.shape[-1])))
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages
        
        # Print comprehensive statistics
        print_data_statistics(stats, "ANGELIDOU")
        report_extreme_values(stats)

        
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

        # Initialize statistics tracking
        stats = {
            'angle': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'kinetic': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'emg': {'min': float('inf'), 'max': float('-inf'), 'values': []},
            'nan_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'inf_count': {'angle': 0, 'kinetic': 0, 'emg': 0},
            'zero_count': {'angle': 0, 'kinetic': 0, 'emg': 0}
        }

        if is_data_in_degrees(np.array(currPickle[activities[0]][directions[0]]['angle'][0][0][0])):
            is_degree = True
            print(f"✓ BACEK Detected angles in DEGREES")
            print("  Converting to radians...")
        else:
            is_degree = False
            print(f"✓ BACEK Detected angles in RADIANS")
            print("  No conversion needed.")

        
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
                
                # Track patient and stride indices for logging
                patient_idx = 0
                
                # Iterate using zip like in syncBacek
                for currPatientEMG, currPatientKinematic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                currPickle[currActivity][currDirection]['angle']):
                    patient_angles = []
                    patient_emgs = []
                    patient_gait_percentages = []
                    
                    stride_idx = 0
                    
                    for currTrialEMG, currTrialKinematic in zip(currPatientEMG, currPatientKinematic):
                        trial_angles = []
                        trial_emgs = []
                        trial_gait_percentages = []
                        
                        for currStrideEMG, currStrideKinematic in zip(currTrialEMG, currTrialKinematic):
                            # Process kinematic data
                            stride_kinematic_raw = np.array(currStrideKinematic)
                            check_and_log_data_quality(stride_kinematic_raw, 'angle', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            
                            if is_degree:
                                stride_kinematic = np.deg2rad(stride_kinematic_raw)
                            else:
                                stride_kinematic = stride_kinematic_raw
                            
                            stats['angle']['values'].append(stride_kinematic.flatten())

                            # Create mask if it doesn't exist (all ones - use all channels)
                            if kinematicMask is None:
                                kinematicMask = np.ones((stride_kinematic.shape[0], stride_kinematic.shape[1]))
                            
                            trial_angles.append(torch.Tensor(resample_stride(stride_kinematic, kinematicMask, target_points)))
                            
                            # Process EMG data
                            stride_emg = np.array(currStrideEMG)
                            check_and_log_data_quality(stride_emg, 'emg', f'{currActivity}-{currDirection}', patient_idx, stride_idx, stats)
                            stats['emg']['values'].append(stride_emg.flatten())
                            
                            resampled_emg = resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz)

                            trial_emgs.append(torch.Tensor(resampled_emg))
                            trial_gait_percentages.append(torch.Tensor(create_gait_percentage_vector(resampled_emg.shape[1])))
                            
                            stride_idx += 1
                            
                        patient_angles.append(trial_angles)
                        patient_emgs.append(trial_emgs)
                        patient_gait_percentages.append(trial_gait_percentages)
                    
                    new_angles.append(patient_angles)
                    new_emgs.append(patient_emgs)
                    new_gait_percentages.append(patient_gait_percentages)
                    
                    patient_idx += 1
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['emg'] = new_emgs
                currPickle[currActivity][currDirection]['emg_gait_percentage'] = new_gait_percentages

        # Print comprehensive statistics
        print_data_statistics(stats, "BACEK")

        output_path = os.path.join(output_folder, "bacek.pkl")
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")
    #Run all resampling functions
    # resample_lencioni()
    #resample_moreira()
    # resample_hu()
    
    # resample_embry()
    # resample_gait120()
    #resample_camargo()
    #resample_k2muse()
    # resample_macaluso()
    #resample_angelidou()
    #resample_grimmer()
    # resample_criekinge()

    #resample_moghadam()
    resample_bacek()
    # resample_siat()




def main():
    print('hello')
    resample_all_datasets()

if __name__ == '__main__':
    main()