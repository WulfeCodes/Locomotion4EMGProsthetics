import pickle
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

#Array Name,Length,"Value at Index i (e.g., 42)",Purpose
# stride_starts,N,10500,Points to the start row in the raw EMG/Kin matrix.
# stride_ends,N,10650,Points to the end row in the raw EMG/Kin matrix.
# stride_metadata/patient_id,N,5,Tells you this stride belongs to Global Patient #5.
# stride_metadata/dataset_id,N,'criekinge',Tells you this stride came from the Criekinge dataset.
# stride_metadata/activity,N,'walk',The label for this specific stride.

class EMGDatasetConverter:
    def __init__(self, input_dir="D:/EMG/postprocessed_datasets", output_path="D:/EMG/combined_dataset.h5"):
        self.input_dir = Path(input_dir)
        self.output_path = output_path
        
        # Initialize storage lists
        self.all_emg = []
        self.all_kinematic = []
        self.all_kinetic = []
        self.all_gait_percentage = []
        
        self.stride_starts = []
        self.stride_ends = []
        
        # Metadata per stride
        self.stride_activity_labels = []
        self.stride_direction_labels = []
        self.stride_patient_ids = []
        self.stride_dataset_ids = []
        self.stride_kinetic_available = []
        
        # Global patient tracking
        self.global_patient_counter = 0
        self.patient_to_dataset = {}  # global_patient_id -> dataset_name
        self.patient_stride_ranges = defaultdict(list)  # global_patient_id -> [stride_indices]
        
        # Dataset masks storage
        self.dataset_masks = {}  # dataset_name -> {emg_mask, kinematic_mask, kinetic_mask}
        
        self.current_timestep = 0
        self.stride_count = 0
        
        # Dataset-specific parsers
        self.parsers = {
            'criekinge': self.parse_criekinge,
            'lencioni': self.parse_lencioni,
            'moreira': self.parse_moreira,
            'hu': self.parse_hu,
            'grimmer': self.parse_grimmer,
            'siat': self.parse_siat,
            'embry': self.parse_embry,
            'gait120': self.parse_gait120,
            'camargo': self.parse_camargo,
            'angelidou': self.parse_angelidou,
            'bacek': self.parse_bacek,
            'macaluso': self.parse_macaluso,
            'k2muse': self.parse_k2muse
            #'moghadam': self.parse_moghadam,

        }
    
    def get_next_patient_id(self, dataset_name):
        """Get next global patient ID and associate with dataset"""
        patient_id = self.global_patient_counter
        self.patient_to_dataset[patient_id] = dataset_name
        self.global_patient_counter += 1
        return patient_id
    
    def add_stride(self, emg, kinematic, kinetic, gait_pct, activity, direction, patient_id, dataset_name):
        """Add a single stride to the dataset"""
        if emg is None or len(emg) == 0:
            return
        
        # Get stride length
        stride_len = emg.shape[1] #if len(emg.shape) > 1 else len(emg)
        
        # Record stride boundaries
        self.stride_starts.append(self.current_timestep)
        self.stride_ends.append(self.current_timestep + stride_len)
        
        # Track which strides belong to which patient
        self.patient_stride_ranges[patient_id].append(self.stride_count)
        
        # Append data (transpose to [timesteps, channels])
        self.all_emg.append(emg.T)# if len(emg.shape) > 1 else emg)
        
        kin_reshaped = kinematic.reshape(9, -1).T
        self.all_kinematic.append(kin_reshaped)
        
        if kinetic is not None and len(kinetic) > 0:
            # Reshape kinetic from [3,3,hz] to [hz,9]
            # [joints, axes, time] -> [time, joints*axes]
            kin_reshaped = kinetic.transpose(2, 0, 1).reshape(-1, 9)
            self.all_kinetic.append(kin_reshaped)
            has_kinetic = True
        else:
            self.all_kinetic.append(np.zeros((stride_len, 9)))
            has_kinetic = False
        
        self.all_gait_percentage.append(gait_pct.T if len(gait_pct.shape) > 1 else gait_pct)

        # Metadata
        self.stride_activity_labels.append(activity)
        self.stride_direction_labels.append(direction)
        self.stride_patient_ids.append(patient_id)
        self.stride_dataset_ids.append(dataset_name)
        self.stride_kinetic_available.append(has_kinetic)
        
        self.current_timestep += stride_len
        self.stride_count += 1
    
    def extract_masks(self, data, dataset_name):
        """Extract masks from pickle file based on dataset-specific structure"""
        masks = {}
        
        # Dataset-specific mask extraction
        if dataset_name == 'criekinge':
            masks['emg'] = np.array(data['mask']['emg']) if data['mask']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['angle']) if data['mask']['angle'] is not None else None
            masks['kinetic'] = np.array(data['mask']['kinetics']) if data['mask']['kinetics'] is not None else None
            
        elif dataset_name == 'moghadam':
            masks['emg'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['left']['kinematic']) if data['mask']['left']['kinematic'] is not None else None
            masks['kinetic'] = np.array(data['mask']['left']['kinetic']) if data['mask']['left']['kinetic'] is not None else None
            
        elif dataset_name == 'lencioni':
            masks['emg'] = np.array(data['mask']['emg']) if data['mask']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['emg']) if data['mask']['emg'] is not None else None  # Note: uses emg mask
            masks['kinetic'] = np.array(data['mask']['kinetic']) if data['mask']['kinetic'] is not None else None
            
        elif dataset_name == 'moreira':
            masks['emg'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['left']['angle']) if data['mask']['left']['angle'] is not None else None
            masks['kinetic'] = np.array(data['mask']['left']['kinetic']) if data['mask']['left']['kinetic'] is not None else None
            
        elif dataset_name == 'hu':
            masks['emg'] = np.array(data['masks']['left']['emg']) if data['masks']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['masks']['left']['angles']) if data['masks']['left']['angles'] is not None else None
            masks['kinetic'] = None  # No kinetic data in hu
            
        elif dataset_name == 'grimmer':
            masks['emg'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None  # Note: uses emg mask
            masks['kinetic'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None  # Note: uses emg mask
            
        elif dataset_name == 'siat':
            masks['emg'] = np.array(data['masks']['left']['emg']) if data['masks']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['masks']['left']['angle']) if data['masks']['left']['angle'] is not None else None
            masks['kinetic'] = np.array(data['masks']['left']['kinetic']) if data['masks']['left']['kinetic'] is not None else None
            
        elif dataset_name == 'embry':
            masks['emg'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['left']['kinematic']) if data['mask']['left']['kinematic'] is not None else None
            masks['kinetic'] = np.array(data['mask']['left']['kinetic']) if data['mask']['left']['kinetic'] is not None else None
            
        elif dataset_name == 'gait120':
            masks['emg'] = np.array(data['mask']['emg']) if data['mask']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['angle']) if data['mask']['angle'] is not None else None
            masks['kinetic'] = None  # No kinetic data in gait120
            
        elif dataset_name == 'camargo':
            masks['emg'] = np.array(data['mask']['emg']) if data['mask']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['angle']) if data['mask']['angle'] is not None else None
            masks['kinetic'] = np.array(data['mask']['kinetic']) if data['mask']['kinetic'] is not None else None
            
        elif dataset_name == 'angelidou':
            masks['emg'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['left']['angle']) if data['mask']['left']['angle'] is not None else None
            masks['kinetic'] = np.array(data['mask']['left']['kinetic']) if data['mask']['left']['kinetic'] is not None else None
            
        elif dataset_name == 'bacek':
            # Bacek stores masks directly in the walk data structure
            masks['emg'] = np.array(data['walk']['left']['emg']) if 'emg' in data['walk']['left'] else None
            masks['kinematic'] = np.array(data['walk']['left']['angle']) if 'angle' in data['walk']['left'] else None
            masks['kinetic'] = None  # No kinetic data in bacek
            
        elif dataset_name == 'macaluso':
            masks['emg'] = np.array(data['mask']['left']['emg']) if data['mask']['left']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['left']['kinematic']) if data['mask']['left']['kinematic'] is not None else None
            masks['kinetic'] = np.array(data['mask']['left']['kinetic']) if data['mask']['left']['kinetic'] is not None else None
            
        elif dataset_name == 'k2muse':
            masks['emg'] = np.array(data['mask']['right']['emg']) if data['mask']['right']['emg'] is not None else None
            masks['kinematic'] = np.array(data['mask']['right']['angle']) if data['mask']['right']['angle'] is not None else None
            masks['kinetic'] = np.array(data['mask']['right']['kinetic']) if data['mask']['right']['kinetic'] is not None else None
                
        return masks
    
    def parse_criekinge(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Extract and store masks
        self.dataset_masks['criekinge'] = self.extract_masks(data, 'criekinge')
        
        directions = ['left', 'right', 'stroke']
        
        for leg in directions:
            patient_emg_list = data['walk'][leg]['emg']
            for pat_idx, (pat_emg, pat_kin, pat_kinetic) in enumerate(zip(
                patient_emg_list,
                data['walk'][leg]['angle'],
                data['walk'][leg]['kinetics']
            )):
                patient_id = self.get_next_patient_id('criekinge')
                
                for stride_emg, stride_kin, stride_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    gait_pct = None
                    if 'emg_gait_percentage' in data['walk'][leg] and len(data['walk'][leg]['emg_gait_percentage']) > pat_idx:
                        gait_list = data['walk'][leg]['emg_gait_percentage'][pat_idx]
                        if len(gait_list) > 0:
                            gait_pct = gait_list[0]
                    
                    self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                  'walk', leg, patient_id, 'criekinge')
    
    def parse_moghadam(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['moghadam'] = self.extract_masks(data, 'moghadam')
        
        directions = ['left', 'right']
        
        for leg in directions:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data['walk'][leg]['emg'],
                data['walk'][leg]['kinematic'],
                data['walk'][leg]['kinetic']
            ):
                patient_id = self.get_next_patient_id('moghadam')
                
                for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    if len(trial_emg) == 0 or len(trial_kin) == 0 or len(trial_kinetic) == 0:
                        continue
                    for stride_emg, stride_kin, stride_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                        gait_pct = data['walk'][leg].get('emg_gait_percentage')
                        self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                      'walk', leg, patient_id, 'moghadam')
    
    def parse_lencioni(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['lencioni'] = self.extract_masks(data, 'lencioni')
        
        activities = ['step up', 'step down', 'walk']
        
        for activity in activities:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data[activity]['emg'],
                data[activity]['angle'],
                data[activity]['kinetic']
            ):
                patient_id = self.get_next_patient_id('lencioni')
                
                for stride_emg, stride_kin, stride_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    gait_pct = data[activity].get('emg_gait_percentage')
                    self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                  activity, 'unknown', patient_id, 'lencioni')
    
    def parse_moreira(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['moreira'] = self.extract_masks(data, 'moreira')
        
        directions = ['left', 'right']
        
        for direction in directions:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data['walk'][direction]['emg'],
                data['walk'][direction]['angle'],
                data['walk'][direction]['kinetic']
            ):
                patient_id = self.get_next_patient_id('moreira')
                
                for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    for stride_emg, stride_kin, stride_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                        gait_pct = data['walk'][direction].get('emg_gait_percentage')
                        self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                      'walk', direction, patient_id, 'moreira')
    
    def parse_hu(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['hu'] = self.extract_masks(data, 'hu')
        
        activities = ['walk', 'ramp_up', 'ramp_down', 'stair_up', 'stair_down']
        directions = ['left', 'right']
        
        for activity in activities:
            for direction in directions:
                for pat_emg, pat_kin in zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['angle']
                ):
                    patient_id = self.get_next_patient_id('hu')
                    
                    for stride_emg, stride_kin in zip(pat_emg, pat_kin):
                        gait_pct = data[activity][direction].get('emg_gait_percentage')
                        self.add_stride(stride_emg, stride_kin, None, gait_pct,
                                      activity, direction, patient_id, 'hu')
    
    def parse_grimmer(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['grimmer'] = self.extract_masks(data, 'grimmer')
        
        activities = ['stairUp', 'stairDown']
        directions = ['left', 'right']
        
        for activity in activities:
            for direction in directions:
                for pat_emg, pat_kin, pat_kinetic in zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['angle'],
                    data[activity][direction]['kinetic']
                ):
                    patient_id = self.get_next_patient_id('grimmer')
                    
                    for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                        for stride_emg, stride_kin, stride_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                            gait_pct = data[activity][direction].get('emg_gait_percentage')
                            self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                          activity, direction, patient_id, 'grimmer')
    
    def parse_siat(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['siat'] = self.extract_masks(data, 'siat')
        
        activities = ['walk', 'stair_up', 'stair_down']
        
        for activity in activities:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data[activity]['left']['emg'],
                data[activity]['left']['angle'],
                data[activity]['left']['kinetic']
            ):
                patient_id = self.get_next_patient_id('siat')
                
                for session_emg, session_kin, session_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    for stride_emg, stride_kin, stride_kinetic in zip(session_emg, session_kin, session_kinetic):
                        gait_pct = data[activity]['left'].get('emg_gait_percentage')
                        self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                      activity, 'left', patient_id, 'siat')
    
    def parse_embry(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['embry'] = self.extract_masks(data, 'embry')
        
        activities = ['walk', 'rampup', 'rampdown']
        directions = ['left', 'right']
        
        for direction in directions:
            for activity in activities:
                for pat_emg, pat_kin, pat_kinetic in zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['kinematic'],
                    data[activity][direction]['kinetic']
                ):
                    patient_id = self.get_next_patient_id('embry')
                    
                    for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                        for stride_emg, stride_kin, stride_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                            gait_pct = data[activity][direction].get('emg_gait_percentage')
                            self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                          activity, direction, patient_id, 'embry')
    
    def parse_gait120(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['gait120'] = self.extract_masks(data, 'gait120')
        
        activities = ['levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 
                     'slopeDescent', 'sitToStand', 'standToSit']
        
        for activity in activities:
            for pat_emg, pat_kin in zip(
                data['right'][activity]['emg'],
                data['right'][activity]['angle']
            ):
                patient_id = self.get_next_patient_id('gait120')
                
                for stride_emg, stride_kin in zip(pat_emg, pat_kin):
                    gait_pct = data['right'][activity].get('emg_gait_percentage')
                    self.add_stride(stride_emg, stride_kin, None, gait_pct,
                                  activity, 'right', patient_id, 'gait120')
    
    def parse_camargo(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['camargo'] = self.extract_masks(data, 'camargo')
        
        activities = ['walk', 'stair', 'ramp']
        
        for activity in activities:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data['right'][activity]['emg'],
                data['right'][activity]['angle'],
                data['right'][activity]['kinetic']
            ):
                patient_id = self.get_next_patient_id('camargo')
                
                for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    for stride_emg, stride_kin, stride_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                        gait_pct = data['right'][activity].get('emg_gait_percentage')
                        self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                      activity, 'right', patient_id, 'camargo')
    
    def parse_angelidou(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['angelidou'] = self.extract_masks(data, 'angelidou')
        
        directions = ['left', 'right']
        
        for direction in directions:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data['walk'][direction]['emg'],
                data['walk'][direction]['angle'],
                data['walk'][direction]['kinetic']
            ):
                patient_id = self.get_next_patient_id('angelidou')
                
                for stride_emg, stride_kin, stride_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    gait_pct = data['walk'][direction].get('emg_gait_percentage')
                    self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                  'walk', direction, patient_id, 'angelidou')
    
    def parse_bacek(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['bacek'] = self.extract_masks(data, 'bacek')
        
        directions = ['left', 'right']
        
        for direction in directions:
            for pat_emg, pat_kin in zip(
                data['walk'][direction]['emg'],
                data['walk'][direction]['angle']
            ):
                patient_id = self.get_next_patient_id('bacek')
                
                for trial_emg, trial_kin in zip(pat_emg, pat_kin):
                    for stride_emg, stride_kin in zip(trial_emg, trial_kin):
                        gait_pct = data['walk'][direction].get('emg_gait_percentage')
                        self.add_stride(stride_emg, stride_kin, None, gait_pct,
                                      'walk', direction, patient_id, 'bacek')
    
    def parse_macaluso(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['macaluso'] = self.extract_masks(data, 'macaluso')
        
        activities = ['walk', 'rampup', 'rampdown']
        directions = ['left', 'right']
        
        for activity in activities:
            for direction in directions:
                for pat_emg, pat_kin, pat_kinetic in zip(
                    data[activity][direction]['emg'],
                    data[activity][direction]['kinematic'],
                    data[activity][direction]['kinetic']
                ):
                    patient_id = self.get_next_patient_id('macaluso')
                    
                    for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                        for stride_emg, stride_kin, stride_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                            gait_pct = data[activity][direction].get('emg_gait_percentage')
                            self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                          activity, direction, patient_id, 'macaluso')
    
    def parse_k2muse(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.dataset_masks['k2muse'] = self.extract_masks(data, 'k2muse')
        
        activities = ['walk', 'up_ramp', 'down_ramp']
        
        for activity in activities:
            for pat_emg, pat_kin, pat_kinetic in zip(
                data['right'][activity]['emg'],
                data['right'][activity]['angle'],
                data['right'][activity]['kinetic']
            ):
                patient_id = self.get_next_patient_id('k2muse')
                
                for trial_emg, trial_kin, trial_kinetic in zip(pat_emg, pat_kin, pat_kinetic):
                    for subtrial_emg, subtrial_kin, subtrial_kinetic in zip(trial_emg, trial_kin, trial_kinetic):
                        for stride_emg, stride_kin, stride_kinetic in zip(subtrial_emg, subtrial_kin, subtrial_kinetic):
                            gait_pct = data['right'][activity].get('emg_gait_percentage')
                            self.add_stride(stride_emg, stride_kin, stride_kinetic, gait_pct,
                                          activity, 'right', patient_id, 'k2muse')
    
    def convert_all(self):
        """Convert all datasets to HDF5"""
        print("Starting conversion of all datasets...")
        
        for dataset_name, parser_func in tqdm(self.parsers.items(), desc="Processing datasets"):
            pkl_path = self.input_dir / f"{dataset_name}.pkl"
            if pkl_path.exists():
                print(f"\nProcessing {dataset_name}...")
                try:
                    parser_func(pkl_path)
                    print(f"  Added {self.stride_count} strides so far")
                    print(f"  Total patients so far: {self.global_patient_counter}")
                except Exception as e:
                    print(f"  Error processing {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\nWarning: {pkl_path} not found, skipping...")
        
        print(f"\nTotal strides collected: {self.stride_count}")
        print(f"Total patients: {self.global_patient_counter}")
        print("Writing to HDF5...")
        self.write_hdf5()
        print(f"Conversion complete! Saved to {self.output_path}")
    
    def write_hdf5(self):
        """Write all collected data to HDF5 file"""
        with h5py.File(self.output_path, 'w') as hf:
            # Concatenate all time series data
            print("Concatenating EMG data...")
            emg_data = np.vstack(self.all_emg)
            hf.create_dataset('emg', data=emg_data, compression='gzip', compression_opts=4)
            
            print("Concatenating kinematic data...")
            kinematic_data = np.vstack(self.all_kinematic)
            hf.create_dataset('kinematic', data=kinematic_data, compression='gzip', compression_opts=4)
            
            print("Concatenating kinetic data...")
            kinetic_data = np.vstack(self.all_kinetic)
            hf.create_dataset('kinetic', data=kinetic_data, compression='gzip', compression_opts=4)
            
            print("Concatenating gait percentage data...")
            gait_pct_data = np.vstack(self.all_gait_percentage)
            hf.create_dataset('gait_percentage', data=gait_pct_data, compression='gzip', compression_opts=4)
            
            # Stride boundaries
            hf.create_dataset('stride_starts', data=np.array(self.stride_starts))
            hf.create_dataset('stride_ends', data=np.array(self.stride_ends))
            
            # Per-stride metadata
            stride_meta = hf.create_group('stride_metadata')
            stride_meta.create_dataset('activity', data=np.array(self.stride_activity_labels, dtype='S'))
            stride_meta.create_dataset('direction', data=np.array(self.stride_direction_labels, dtype='S'))
            stride_meta.create_dataset('patient_id', data=np.array(self.stride_patient_ids))
            stride_meta.create_dataset('dataset_id', data=np.array(self.stride_dataset_ids, dtype='S'))
            stride_meta.create_dataset('has_kinetic', data=np.array(self.stride_kinetic_available))
            
            # Patient-level information
            print("Building patient info...")
            patient_info = hf.create_group('patient_info')
            
            unique_patients = sorted(self.patient_to_dataset.keys())
            patient_datasets = [self.patient_to_dataset[pid] for pid in unique_patients]
            
            # Calculate first and last stride for each patient
            patient_first_stride = []
            patient_last_stride = []
            for pid in unique_patients:
                stride_indices = sorted(self.patient_stride_ranges[pid])
                patient_first_stride.append(stride_indices[0])
                patient_last_stride.append(stride_indices[-1])
            
            patient_info.create_dataset('patient_id', data=np.array(unique_patients))
            patient_info.create_dataset('dataset_id', data=np.array(patient_datasets, dtype='S'))
            patient_info.create_dataset('first_stride', data=np.array(patient_first_stride))
            patient_info.create_dataset('last_stride', data=np.array(patient_last_stride))
            patient_info.create_dataset('num_strides', data=np.array([len(self.patient_stride_ranges[pid]) for pid in unique_patients]))
            
            # Dataset masks
            print("Storing dataset masks...")
            masks_group = hf.create_group('dataset_masks')
            for dataset_name, masks in self.dataset_masks.items():
                dataset_mask_group = masks_group.create_group(dataset_name)
                
                if masks['emg'] is not None:
                    dataset_mask_group.create_dataset('emg_mask', data=masks['emg'])
                if masks['kinematic'] is not None:
                    dataset_mask_group.create_dataset('kinematic_mask', data=masks['kinematic'])
                if masks['kinetic'] is not None:
                    dataset_mask_group.create_dataset('kinetic_mask', data=masks['kinetic'])
            
            # Store dataset info
            hf.attrs['total_strides'] = self.stride_count
            hf.attrs['total_timesteps'] = self.current_timestep
            hf.attrs['total_patients'] = self.global_patient_counter
            hf.attrs['emg_channels'] = 13
            hf.attrs['kinematic_features'] = 9
            hf.attrs['kinetic_features'] = 9
            hf.attrs['num_datasets'] = len(self.dataset_masks)


if __name__ == "__main__":
    converter = EMGDatasetConverter(
        input_dir="D:/EMG/postprocessed_datasets",
        output_path="D:/EMG/combined_dataset.h5"
    )
    converter.convert_all()
    
    # Verify the output
    print("\n" + "="*50)
    print("Verifying HDF5 file...")
    with h5py.File("D:/EMG/combined_dataset.h5", 'r') as hf:
        print(f"\nData shapes:")
        print(f"  EMG: {hf['emg'].shape}")
        print(f"  Kinematic: {hf['kinematic'].shape}")
        print(f"  Kinetic: {hf['kinetic'].shape}")
        print(f"  Gait percentage: {hf['gait_percentage'].shape}")
        
        print(f"\nDataset statistics:")
        print(f"  Total strides: {hf.attrs['total_strides']}")
        print(f"  Total timesteps: {hf.attrs['total_timesteps']}")
        print(f"  Total patients: {hf.attrs['total_patients']}")
        print(f"  Number of datasets: {hf.attrs['num_datasets']}")
        
        print(f"\nPatient info:")
        print(f"  Unique patients: {len(hf['patient_info']['patient_id'][:])}")
        print(f"  Example patient 0: {hf['patient_info']['dataset_id'][0].decode()}")
        print(f"    First stride: {hf['patient_info']['first_stride'][0]}")
        print(f"    Last stride: {hf['patient_info']['last_stride'][0]}")
        print(f"    Num strides: {hf['patient_info']['num_strides'][0]}")
        
        print(f"\nDataset masks available:")
        for dataset_name in hf['dataset_masks'].keys():
            print(f"  {dataset_name}:")
            if 'emg_mask' in hf['dataset_masks'][dataset_name]:
                print(f"    EMG mask shape: {hf['dataset_masks'][dataset_name]['emg_mask'].shape}")
            if 'kinematic_mask' in hf['dataset_masks'][dataset_name]:
                print(f"    Kinematic mask shape: {hf['dataset_masks'][dataset_name]['kinematic_mask'].shape}")
            if 'kinetic_mask' in hf['dataset_masks'][dataset_name]:
                print(f"    Kinetic mask shape: {hf['dataset_masks'][dataset_name]['kinetic_mask'].shape}")
        
        print(f"\nSample stride 0:")
        print(f"  Activity: {hf['stride_metadata']['activity'][0].decode()}")
        print(f"  Direction: {hf['stride_metadata']['direction'][0].decode()}")
        print(f"  Patient ID: {hf['stride_metadata']['patient_id'][0]}")
        print(f"  Dataset: {hf['stride_metadata']['dataset_id'][0].decode()}")
        print(f"  Has kinetic: {hf['stride_metadata']['has_kinetic'][0]}")
        print(f"  Start: {hf['stride_starts'][0]}, End: {hf['stride_ends'][0]}")