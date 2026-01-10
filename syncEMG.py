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

#NOTE do we keep kinematic sample num normalized(map to a label of len(hz) to gait cycle percentage?)
#or do we convert all the kinematic back to their time freq? 
#^^ add gait cycle percentage latency..ie label = i - latencyIndices(2, ~100 ms)

#will model output, velocity, and acceleration with these being lossed against numerical differentiation?
#or only angles with numerical differention being used with model's thetas?
#predicting all orders in output would lead to greater ground truth?
#output phase % as input as well

#ie^^ velocity, acceleration losses of phase ss domain?

#TODO uniform sample rate, 
# find concatenations, 
# train test val split for each task and distribution(dataset)

#SIAT supposedly has maximum swing flexion segmentation script

def checkLists():
    Gait120EMGs = ['Vastus Lateralis','Rectus Femoris','Vastus Medialis','Tibialis Anterior','Biceps Femoris', 'Semitendinosus','Gastrocnemius Medialis','Gastrocnemius Lateralis','(Soleus Medialis'+ 'Soleus Lateralis','Peroneus Longus','Peroneus Brevis',0,0]
    emgCamargo = ["vastuslateralis",0,"vastusmedialis","tibialisanterior","bicepsfemoris",
                "semitendinosus","gastrocmed",0,"soleus",0,0,"gluteusmedius", 0]
    SIATemgs = [0, 'sEMG: rectus femoris',  'sEMG: vastus medialis', 'sEMG tibialis anterior merged', 0, 'sEMG: semimembranosus',   'sEMG: medial gastrocnemius', 'sEMG: lateral gastrocnemius',  'sEMG: soleus', 0, 0,0, 0]     
    Lencioniemgs = [0, 'Rectus Femoris', 'Vastus Medialis', 'Tibialis Anterior',  'Biceps Femoris', 0 ,'Gastrocnemius Medialis', 0, 'Soleus', 'Peroneus Longus', 0, 0,'Gluteus Maximus']
    UCIrvineEMGs = [0, 'RF', 'VM', 0, 'BF', 'ST', 0, 0, 0,0,0,0,0]
    MacalusoEMGs = [0,'RF_norm',0,'TA_norm','BF_norm',0,'GC_norm', 0, 0, 0,0, 0, 0]
    GrimmerEMGs = 	['vas_dir','rcf_dir',0,'tib_dir','bcf_dir',0,0,'gas_dir','sol_dir', 0, 0, 0,0]
    EmbryEMGs = 	[0,'RF',0,'TA','BF',0,'GC',0,0,0,0,0,0]
    angelidouEMGs=["DirVL","DirRF", 0,"DirTA","DirBF", 0,"DirGA",0,0,0,0,0,0]
    criekingeEMGs=["VLnorm", "RFnorm",0, "TAnorm", "BFnorm", "STnorm","GASnorm", 0,0,0,0,0,0 ]
    MoghadamEMGs = ['DirVastusLateralis','DirRectusFemoris',0,'DirTibialisAnterior','DirBicepsFemoris','DirSemitendinosus', 'DirGastrocnemiusMedialis',0,'DirSoleus',0,0,0,"DirGleuteusMaximus"]
    HuEMGs = 	['Dir_VL','Dir_RF',0,'Dir_TA','Dir_BF','Dir_ST', 'Dir_MG', 0, 'Dir_SOL',0,0,0,0]
    BacekEMGs = ['Dir_VastLat','Dir_RecFem', 0, 'Dir_TibAnt', 'Dir_BicFem', 'Dir_Semitend','Dir_GastroMed', 'Dir_GastroLat', 0,0,0,0,'Dir_GlutMax']
    K2museRightEMGs = ['VLO', 'RF', 'VMO', 'TA', 'BF','SEM', 'MG', 'ML', 'SOL',0,0,0,0]
    K2museLeftEMGs =  [0,'RF', 0,'TA','BF',0,'LG',0,0,0,0,0,0]
    MoreiraEMGs = 	['VL',0,0,'TA','BF',0,0,'GAL',0,0,0,0,0]
    MasterEMGs =['Vastus Lateralis','Rectus Femoris','Vastus Medialis','Tibialis Anterior',
        'Biceps Femoris','Semitendinosus or sEMG: semimembranosus','Gastrocnemius Medialis','Gastrocnemius Lateralis',
        'Soleus','Peroneus Longus','Peroneus Brevis',"gluteusmedius",'Gluteus Maximus']

    arrays = {
        'angelidouEMGs':angelidouEMGs,
        'CriekingeEMGs':criekingeEMGs,
        'Gait120EMGs': Gait120EMGs,
        'emgCamargo': emgCamargo,
        'SIATemgs': SIATemgs,
        'Lencioniemgs': Lencioniemgs,
        'UCIrvineEMGs': UCIrvineEMGs,
        'MacalusoEMGs': MacalusoEMGs,
        'GrimmerEMGs': GrimmerEMGs,
        'EmbryEMGs': EmbryEMGs,
        'MoghadamEMGs': MoghadamEMGs,
        'HuEMGs': HuEMGs,
        'BacekEMGs': BacekEMGs,
        'K2museRightEMGs': K2museRightEMGs,
        'K2museLeftEMGs': K2museLeftEMGs,
        'MoreiraEMGs': MoreiraEMGs,
        'masta': MasterEMGs
    }
    
    print("Array Lengths:")
    for name, arr in arrays.items():
        print(f"{name}: {len(arr)}")
    
    # Check for inconsistencies
    lengths = [len(arr) for arr in arrays.values()]
    assert len(set(lengths)) == 1, f"Inconsistent lengths found: {set(lengths)}"

def syncAll():

    def syncCriekinge(currPath="D:/EMG/processed_datasets/criekinge.pkl"):
        #only walking 

        directions = ['left','right','stroke']


        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetics']
        emgMask = currPickle['mask']['emg']

        for currLeg in directions:
            print(currPickle['walk'][currLeg].keys())
            #16 trials
            for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle['walk'][currLeg]['emg'],
                                                                                currPickle['walk'][currLeg]['angle'],
                                                                                currPickle['walk'][currLeg]['kinetics']):
                for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                    #currStrideEMG.shape = 13,hz currStrideKinematic/Kinetic.shape = 3,3,hz

                    continue
    def syncMoghadam(currPath="D:/EMG/processed_datasets/moghadam.pkl"):
        #TODO email about ind files and their continuity and strde count
        directions = ['left','right']
        #only walk here

        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        emgMask=currPickle['mask']['left']['emg']
        kinematicMask=currPickle['mask']['left']['kinematic']
        kineticMask=currPickle['mask']['left']['kinetic']


        print(currPickle['walk'].keys(),currPickle['mask'].keys())

        for currLeg in directions:
            print(currPickle['walk'][currLeg].keys())
            #16 trials
            for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle['walk'][currLeg]['emg'],
                                                                                currPickle['walk'][currLeg]['kinematic'],
                                                                                currPickle['walk'][currLeg]['kinetic']):
                for currTrialEMG, currTrialKinematic, currTrialKinetic in zip(currPatientEMG, currPatientKinematic, currPatientKinetic):    
                    print(len(currPatientEMG[0]))
                    if len(currTrialEMG)==0 or len(currTrialKinematic)==0 or len(currTrialKinetic) ==0:
                        print('continued')
                        continue
                    else:
                        for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currTrialEMG, currTrialKinematic, currTrialKinetic):
                        #currStrideEMG.shape = 13,hz currStrideKinematic/Kinetic.shape = 3,3,hz
                            pass

    def syncLencioni(currPath="D:/EMG/processed_datasets/lencioni.pkl"):
        activities = ['step up', 'step down', 'walk']
        directions = ['left','right']

        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        kineticMask = currPickle['mask']['kinetic']
        kinematicMask = currPickle['mask']['emg']
        emgMask = currPickle['mask']['emg']
        for currActivity in activities:

            for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currActivity]['emg'],
                                                            currPickle[currActivity]['angle'],
                                                            currPickle[currActivity]['kinetic']):
                for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                    pass
                    #currStrideEMG shape = 13,hz
                    #currStirdeKinematic shape = 3,3,hz
                    
            print(currPickle['mask'].keys())

    
    def syncMoreira(currPath="D:/EMG/processed_datasets/moreira.pkl"):
        directions = ['left','right']
        activities=['walk']

        #only walk data here

        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        print(currPickle.keys())

        leftEMGMask = currPickle['mask']['left']['emg']
        leftKineticMask = currPickle['mask']['left']['kinetic']
        leftKinematicMask = currPickle['mask']['left']['angle']
        rightEMGMask = currPickle['mask']['right']['emg']
        rightKineticMask = currPickle['mask']['right']['kinetic']
        rightKinematicMask = currPickle['mask']['right']['angle']

        for currDirection in directions:

            for currActivity in activities:
                for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                                    currPickle[currActivity][currDirection]['angle'],
                                                                                    currPickle[currActivity][currDirection]['kinetic']):
                    for currTrialEMG, currTrialKinematic,currTrialKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                        for currSuccessiveStrideEMG,currSuccessiveStrideKinematic,currSuccessiveStrideKinetic in  zip(currTrialEMG,currTrialKinematic,currTrialKinetic):
                            input(currSuccessiveStrideEMG.shape)
                                #currStrideEMG shape = 13,hz
                                #currStrideKinematic shape = 3,3,hz

    
    def syncHu(currPath="D:/EMG/processed_datasets/hu.pkl"):
        activities = ['walk', 'ramp_up', 'ramp_down', 'stair_up', 'stair_down']
        directions = ['left','right']

        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        kinematicMask=currPickle['masks']['left']['angles']
        emgMask=currPickle['masks']['left']['emg']

        print(currPickle['masks']['left'].keys(),currPickle['walk']['left'].keys())

        for currActivity in activities:

            for currDirection in directions:
                
                for currPatientEMG, currPatientKinematic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                currPickle[currActivity][currDirection]['angle']):
                                                                
                    for currStrideEMG, currStrideKinematic, in zip(currPatientEMG,currPatientKinematic):
                        print(len(currStrideEMG[0]),len(currStrideKinematic[0]),len(currStrideKinematic[0][0]))
                        input()
                        #currStrideEMG shape = 13,hz
                        #currStirdeKinematic shape = 3,3,hz

    def syncGrimmer(currPath="D:/EMG/processed_datasets/grimmer.pkl"):
        activities = ['stairUp','stairDown']
        directions = ['left','right']
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        print(currPickle['mask']['left'].keys())
        leftEMGMask = currPickle['mask']['left']['emg']
        leftKineticMask = currPickle['mask']['left']['emg']
        leftKinematicMask = currPickle['mask']['left']['emg']

        rightEMGMask = currPickle['mask']['right']['emg']
        rightKineticMask = currPickle['mask']['right']['emg']
        rightKinematicMask = currPickle['mask']['right']['emg']

        for currActivity in activities:

            for currDirection in directions:
                
                for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                                    currPickle[currActivity][currDirection]['angle'],
                                                                                    currPickle[currActivity][currDirection]['kinetic']):
                    for currTrialEMG, currTrialKinematic, currTrialKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                       for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currTrialEMG,currTrialKinematic,currTrialKinetic):
                            #currStrideEMG shape = 13, 238
                            #currStrideKinetic/Kinematic shape = 3,3,238
                        
                        print(len(currStrideEMG[0]),len(currStrideKinematic[0][0]),len(currStrideKinetic[0][0]))
                        input()

        print(currPickle['stairDown']['left'].keys())

    def syncSIAT(currPath="D:/EMG/processed_datasets/siat.pkl"):
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
            #only includes left
        activities= ['walk', 'stair_up', 'stair_down']

        maskDict = currPickle['masks']['left']
        
        emgMask = maskDict['emg']
        kinematicMask=maskDict['angle']
        kineticMask=maskDict['kinetic']

        for activityType in activities:
            if activityType == 'masks':
                continue
            
            print(currPickle[activityType]['left'].keys())

            emgDataDict = currPickle[activityType]['left']['emg']
            kinematicDataDict = currPickle[activityType]['left']['angle']
            kineticDataDict = currPickle[activityType]['left']['kinetic']

            #40 patients
            for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(emgDataDict,kinematicDataDict,kineticDataDict):
                #0th dim is of length 1, number of sessions for the activity currPatient[i]
                
                for currSessionEMG, currSessionKinematic, currSessionKinetic in zip(currPatientEMG, currPatientKinematic, currPatientKinetic):
                    #1st dim is the number of strides currPatient[i][j]

                    for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currSessionEMG, currSessionKinematic, currSessionKinetic):
                        break
                        #2nd dim is the number of channels  currPatient[i][j][k]

                        #currStrideEMG=(13,2000~2500)
                        #currKinematic/kinetic = (3,3,2000~2500)
                        #13: emg, 3:kinetic,kinematic
    def syncEmbry(currPath="D:/EMG/processed_datasets/embry.pkl"):
        #TODO #TODO
        directions = ['left','right']


        activities = ['walk', 'rampup', 'rampdown']
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        leftEMGMask = currPickle['mask']['left']['emg']
        leftKineticMask = currPickle['mask']['left']['kinetic']
        leftKinematicMask = currPickle['mask']['left']['kinematic']

        rightEMGMask = currPickle['mask']['right']['emg']
        rightKineticMask = currPickle['mask']['right']['kinetic']
        rightKinematicMask = currPickle['mask']['right']['kinematic']


        for currDirection in directions:
            for currActivity in activities:
                for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                                    currPickle[currActivity][currDirection]['kinematic'],
                                                                                    currPickle[currActivity][currDirection]['kinetic']):
                    for currTrialEMG, currTrialKinematic, currTrialKinetic in zip(currPatientEMG,currPatientKinematic, currPatientKinetic):
                        for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currTrialEMG,currTrialKinematic, currTrialKinetic):
                            input(currStrideEMG.shape)
                            #currStrideEMG shape = 13,hz
                            #currStirdeKinematic/Kinetic shape = 3,3,hz
    def syncGait120(currPath = "D:/EMG/processed_datasets/gait120.pkl"):
        activities =['levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 'slopeDescent', 'sitToStand', 'standToSit']
        with open(currPath,'rb') as file:
            currPickle=pickle.load(file)
        kinematicMask = currPickle['mask']['angle']
        emgMask = currPickle['mask']['emg']

        #only right here
        for currActivity in activities:
            for currPatientEMG, currPatientKinematic in zip(currPickle['right'][currActivity]['emg'],currPickle['right'][currActivity]['angle']):
                for currStrideEMG, currStrideKinematic in zip(currPatientEMG,currPatientKinematic):
                        input()
                        #currStrideEMG shape = 13,hz
                        #currStrideKinematic shape = 3,3,hz

    def syncCamargo(currPath = "D:/EMG/processed_datasets/camargo.pkl"):
        activities = ['walk', 'stair', 'ramp']
        with open(currPath, 'rb') as file:
            currPickle=pickle.load(file)
        kinematicMask = currPickle['mask']['angle']
        kineticMask = currPickle['mask']['kinetic']
        emgMask = currPickle['mask']['emg']

        for currActivity in activities:
            for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle['right'][currActivity]['emg'],
                                                                                currPickle['right'][currActivity]['angle'],
                                                                                currPickle['right'][currActivity]['kinetic']):
                for currTrialEMG, currTrialKinematic, currTrialKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                    for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currTrialEMG, currTrialKinematic, currTrialKinetic):

                        #input(f'{len(currStrideEMG)}')
                        input(f'{currStrideEMG.shape},{currStrideKinematic.shape}')

                        #currStrideEMG shape = 13,hz(NOTE 15k??)
                        #currStrideKinematic/Kinetic shape = 3,3,hz

    def syncAngelidou(currPath = "D:/EMG/processed_datasets/angelidou.pkl"):
        #TODO SYNC
        activities = ['walk']
        directions = ['left','right']


        with open(currPath, 'rb') as file:
            currPickle=pickle.load(file)
        leftKinematicMask = currPickle['mask']['left']['angle']
        leftKineticMask = currPickle['mask']['left']['kinetic']
        leftEMGMask = currPickle['mask']['left']['emg']

        rightKinematicMask = currPickle['mask']['right']['angle']
        rightKineticMask = currPickle['mask']['right']['kinetic']
        rightEMGMask = currPickle['mask']['right']['emg']
        #only walk data
        for currActivity in activities:
            for currDirection in directions:
                for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                                    currPickle[currActivity][currDirection]['angle'],
                                                                                    currPickle[currActivity][currDirection]['kinetic']):
                    for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                        input(f'{currStrideEMG.shape},{currStrideKinematic.shape},{currStrideKinetic.shape}')
                        
                        #currStrideEMG shape = 13,hz
                        #currStrideKinematic/Kinetic shape = 3,3,hz

    def syncBacek(currPath = "D:/EMG/processed_datasets/bacek.pkl"):
        #TODO sync script
        activities = ['walk']
        directions = ['left','right']
        with open(currPath, 'rb') as file:
            currPickle=pickle.load(file)
        kinematicLeftMask = currPickle['walk']['left']['angle']
        emgLeftMask = currPickle['walk']['left']['emg']
        kinematicRightMask = currPickle['walk']['right']['angle']
        emgRightMask = currPickle['walk']['right']['emg']

        for currActivity in activities:
            for currDirection in directions:
                for currPatientEMG, currPatientKinematic in zip(currPickle[currActivity][currDirection]['emg'],currPickle[currActivity][currDirection]['angle']):
                    for currTrialEMG, currTrialKinematic in zip(currPatientEMG,currPatientKinematic):
                        for currStrideEMG, currStrideKinematic in zip(currTrialEMG, currTrialKinematic):

                            #currStrideEMG shape = 13,hz
                            #currStrideKinematic shape = 3,3,hz
                            input(currStrideEMG.shape)
                    input()
    def syncMacaluso(currPath="D:/EMG/processed_datasets/macaluso.pkl"):
        activities = ['walk', 'rampup', 'rampdown']
        directions = ['right','left']
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        leftKinematicMask = currPickle['mask']['left']['kinematic']
        leftKineticMask = currPickle['mask']['left']['kinetic']

        leftEMGMask = currPickle['mask']['left']['emg']
        rightKinematicMask = currPickle['mask']['right']['kinematic']
        rightKineticMask = currPickle['mask']['right']['kinetic']

        rightEMGMask = currPickle['mask']['right']['emg']

        for currActivity in activities:
            for currDirection in directions:
                for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currActivity][currDirection]['emg'],
                                                                                    currPickle[currActivity][currDirection]['kinematic'],
                                                                                    currPickle[currActivity][currDirection]['kinetic']):
                    for currTrialEMG, currTrialKinematic,currTrialKinetic in zip(currPatientEMG,currPatientKinematic,currPatientKinetic):
                        for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currTrialEMG, currTrialKinematic,currTrialKinetic):
                            print(currStrideEMG.shape,currStrideKinematic.shape,currStrideKinetic.shape)
                            #currStrideEMG shape = 13,hzs
                            #currStrideKinematic/Kinetic shape = 3,3,hz
                    
                            input()
    def syncK2Muse(currPath = "D:/EMG/processed_datasets/k2muse.pkl"):
        direction = ['right']
        activities = ['walk', 'up_ramp', 'down_ramp']
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        print(currPickle['mask']['right'].keys())
        kinematicMask = currPickle['mask']['right']['angle']
        kineticMask = currPickle['mask']['right']['kinetic']
        emgMask = currPickle['mask']['right']['emg']

        print(currPickle['right']['walk'].keys())
        print(currPickle['mask'].keys(),currPickle['right'].keys())
        for currDirection in direction:
            for currActivity in activities:
                for currPatientEMG, currPatientKinematic, currPatientKinetic in zip(currPickle[currDirection][currActivity]['emg'],
                                                                                    currPickle[currDirection][currActivity]['angle'],
                                                                                    currPickle[currDirection][currActivity]['kinetic']):
                    for currTrialEMG, currTrialKinematic, currTrialKinetic in zip(currPatientEMG, currPatientKinematic, currPatientKinetic):
                        for currSubTrialEMG, currSubTrialKinematic, currSubTrialKinetic in zip(currTrialEMG, currTrialKinematic, currTrialKinetic):
                            for currStrideEMG, currStrideKinematic, currStrideKinetic in zip(currSubTrialEMG, currSubTrialKinematic, currSubTrialKinetic):
                                input(f'{currStrideEMG.shape},{currStrideKinematic.shape},{currStrideKinetic.shape}')
                            #currStrideEMG shape = 13,hz
                            #currStrideKinematic/Kinetic shape = 3,3,hz
    syncBacek()

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
        
    Examples:
    ---------
    >>> # For kinematic data with shape (3, 3, hz)
    >>> kinematic = np.random.randn(3, 3, 150)
    >>> mask = np.array([[1, 1, 1],
    ...                  [1, 1, 1],
    ...                  [1, 0, 0]])
    >>> resampled = resample_stride(kinematic, mask, 200)
    >>> resampled.shape
    (3, 3, 200)
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
    
    return resampled_data

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
    return resampled


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
            
            for patient_idx in range(len(currPickle['walk'][currLeg]['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                
                for stride_idx in range(len(currPickle['walk'][currLeg]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle['walk'][currLeg]['angle'][patient_idx][stride_idx])
                    patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                    
                    stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetics'][patient_idx][stride_idx])
                    patient_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                    
                    stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][stride_idx])
                    patient_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
            
            currPickle['walk'][currLeg]['angle'] = new_angles
            currPickle['walk'][currLeg]['kinetics'] = new_kinetics
            currPickle['walk'][currLeg]['emg'] = new_emgs
        
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
            
            for patient_idx in range(len(currPickle['walk'][currLeg]['kinematic'])):
                patient_kinematics = []
                patient_kinetics = []
                patient_emgs = []
                
                for trial_idx in range(len(currPickle['walk'][currLeg]['kinematic'][patient_idx])):
                    if len(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx]) == 0:
                        patient_kinematics.append([])
                        patient_kinetics.append([])
                        patient_emgs.append([])
                        continue
                    
                    trial_kinematics = []
                    trial_kinetics = []
                    trial_emgs = []
                    
                    for stride_idx in range(len(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx])):
                        stride_kinematic = np.array(currPickle['walk'][currLeg]['kinematic'][patient_idx][trial_idx][stride_idx])
                        trial_kinematics.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle['walk'][currLeg]['kinetic'][patient_idx][trial_idx][stride_idx])
                        trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        stride_emg = np.array(currPickle['walk'][currLeg]['emg'][patient_idx][trial_idx][stride_idx])
                        trial_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                    
                    patient_kinematics.append(trial_kinematics)
                    patient_kinetics.append(trial_kinetics)
                    patient_emgs.append(trial_emgs)
                
                new_kinematics.append(patient_kinematics)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
            
            currPickle['walk'][currLeg]['kinematic'] = new_kinematics
            currPickle['walk'][currLeg]['kinetic'] = new_kinetics
            currPickle['walk'][currLeg]['emg'] = new_emgs
        
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
            
            for patient_idx in range(len(currPickle[currActivity]['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                
                for stride_idx in range(len(currPickle[currActivity]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle[currActivity]['angle'][patient_idx][stride_idx])
                    patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                    
                    stride_kinetic = np.array(currPickle[currActivity]['kinetic'][patient_idx][stride_idx])
                    patient_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                    
                    stride_emg = np.array(currPickle[currActivity]['emg'][patient_idx][stride_idx])
                    patient_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
            
            currPickle[currActivity]['angle'] = new_angles
            currPickle[currActivity]['kinetic'] = new_kinetics
            currPickle[currActivity]['emg'] = new_emgs
        
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
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        trial_angles = []
                        trial_kinetics = []
                        trial_emgs = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            trial_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                        
                        patient_angles.append(trial_angles)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
        
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
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_emgs = []
                    
                    for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][stride_idx])
                        patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][stride_idx])
                        patient_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                    
                    new_angles.append(patient_angles)
                    new_emgs.append(patient_emgs)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['emg'] = new_emgs
        
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
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx])):
                        trial_angles = []
                        trial_kinetics = []
                        trial_emgs = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['angle'][patient_idx][trial_idx][stride_idx])
                            trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            trial_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                        
                        patient_angles.append(trial_angles)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                
                currPickle[currActivity][currDirection]['angle'] = new_angles
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
        
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
            
            for patient_idx in range(len(currPickle[activityType]['left']['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                
                for session_idx in range(len(currPickle[activityType]['left']['angle'][patient_idx])):
                    session_angles = []
                    session_kinetics = []
                    session_emgs = []
                    
                    for stride_idx in range(len(currPickle[activityType]['left']['angle'][patient_idx][session_idx])):
                        stride_kinematic = np.array(currPickle[activityType]['left']['angle'][patient_idx][session_idx][stride_idx])
                        session_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle[activityType]['left']['kinetic'][patient_idx][session_idx][stride_idx])
                        session_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        stride_emg = np.array(currPickle[activityType]['left']['emg'][patient_idx][session_idx][stride_idx])
                        session_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                    
                    patient_angles.append(session_angles)
                    patient_kinetics.append(session_kinetics)
                    patient_emgs.append(session_emgs)
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
            
            currPickle[activityType]['left']['angle'] = new_angles
            currPickle[activityType]['left']['kinetic'] = new_kinetics
            currPickle[activityType]['left']['emg'] = new_emgs
        
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
                
                for patient_idx in range(len(currPickle[currActivity][currDirection]['kinematic'])):
                    patient_kinematics = []
                    patient_kinetics = []
                    patient_emgs = []
                    
                    for trial_idx in range(len(currPickle[currActivity][currDirection]['kinematic'][patient_idx])):
                        trial_kinematics = []
                        trial_kinetics = []
                        trial_emgs = []
                        
                        for stride_idx in range(len(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx])):
                            stride_kinematic = np.array(currPickle[currActivity][currDirection]['kinematic'][patient_idx][trial_idx][stride_idx])
                            trial_kinematics.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                            
                            stride_kinetic = np.array(currPickle[currActivity][currDirection]['kinetic'][patient_idx][trial_idx][stride_idx])
                            trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                            
                            stride_emg = np.array(currPickle[currActivity][currDirection]['emg'][patient_idx][trial_idx][stride_idx])
                            trial_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                        
                        patient_kinematics.append(trial_kinematics)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                    
                    new_kinematics.append(patient_kinematics)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                
                currPickle[currActivity][currDirection]['kinematic'] = new_kinematics
                currPickle[currActivity][currDirection]['kinetic'] = new_kinetics
                currPickle[currActivity][currDirection]['emg'] = new_emgs
        
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
            
            for patient_idx in range(len(currPickle['right'][currActivity]['angle'])):
                patient_angles = []
                patient_emgs = []
                
                for stride_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx])):
                    stride_kinematic = np.array(currPickle['right'][currActivity]['angle'][patient_idx][stride_idx])
                    patient_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                    
                    stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][stride_idx])
                    patient_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                
                new_angles.append(patient_angles)
                new_emgs.append(patient_emgs)
            
            currPickle['right'][currActivity]['angle'] = new_angles
            currPickle['right'][currActivity]['emg'] = new_emgs
        
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
            
            for patient_idx in range(len(currPickle['right'][currActivity]['angle'])):
                patient_angles = []
                patient_kinetics = []
                patient_emgs = []
                
                for trial_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx])):
                    trial_angles = []
                    trial_kinetics = []
                    trial_emgs = []
                    
                    for stride_idx in range(len(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx])):
                        stride_kinematic = np.array(currPickle['right'][currActivity]['angle'][patient_idx][trial_idx][stride_idx])
                        trial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                        
                        stride_kinetic = np.array(currPickle['right'][currActivity]['kinetic'][patient_idx][trial_idx][stride_idx])
                        trial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                        
                        stride_emg = np.array(currPickle['right'][currActivity]['emg'][patient_idx][trial_idx][stride_idx])
                        trial_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                    
                    patient_angles.append(trial_angles)
                    patient_kinetics.append(trial_kinetics)
                    patient_emgs.append(trial_emgs)
                
                new_angles.append(patient_angles)
                new_kinetics.append(patient_kinetics)
                new_emgs.append(patient_emgs)
            
            currPickle['right'][currActivity]['angle'] = new_angles
            currPickle['right'][currActivity]['kinetic'] = new_kinetics
            currPickle['right'][currActivity]['emg'] = new_emgs
        
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
                
                for patient_idx in range(len(currPickle[currDirection][currActivity]['angle'])):
                    patient_angles = []
                    patient_kinetics = []
                    patient_emgs = []
                    
                    for trial_idx in range(len(currPickle[currDirection][currActivity]['angle'][patient_idx])):
                        trial_angles = []
                        trial_kinetics = []
                        trial_emgs = []
                        
                        for subtrial_idx in range(len(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx])):
                            subtrial_angles = []
                            subtrial_kinetics = []
                            subtrial_emgs = []
                            
                            for stride_idx in range(len(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx])):
                                stride_kinematic = np.array(currPickle[currDirection][currActivity]['angle'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_angles.append(resample_stride(stride_kinematic, kinematicMask, target_points))
                                
                                stride_kinetic = np.array(currPickle[currDirection][currActivity]['kinetic'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_kinetics.append(resample_stride(stride_kinetic, kineticMask, target_points))
                                
                                stride_emg = np.array(currPickle[currDirection][currActivity]['emg'][patient_idx][trial_idx][subtrial_idx][stride_idx])
                                subtrial_emgs.append(resample_emg(stride_emg, ORIGINAL_EMG_HZ, target_emgHz))
                            
                            trial_angles.append(subtrial_angles)
                            trial_kinetics.append(subtrial_kinetics)
                            trial_emgs.append(subtrial_emgs)
                        
                        patient_angles.append(trial_angles)
                        patient_kinetics.append(trial_kinetics)
                        patient_emgs.append(trial_emgs)
                    
                    new_angles.append(patient_angles)
                    new_kinetics.append(patient_kinetics)
                    new_emgs.append(patient_emgs)
                
                currPickle[currDirection][currActivity]['angle'] = new_angles
                currPickle[currDirection][currActivity]['kinetic'] = new_kinetics
                currPickle[currDirection][currActivity]['emg'] = new_emgs
        
        output_path = os.path.join(output_folder,'k2muse.pkl')
        with open(output_path, 'wb') as file:
            pickle.dump(currPickle, file)
        print(f"Saved: {output_path}")

    resample_moghadam()    # 100Hz → 1000Hz
    resample_grimmer()     # 1111.1111Hz → 1000Hz
    resample_siat()        # 1926Hz → 1000Hz
    resample_k2muse()      # 2000Hz → 1000Hz

def analyze_sample_counts():
    """Analyze both kinematic/kinetic and EMG sample counts across all datasets"""
    
    # Dictionaries to store stride counts for each dataset
    kinematic_stride_counts = {}
    emg_stride_counts = {}
    
    def analyze_dataset(name, kinematic_samples, emg_samples):
        """Analyze both kinematic and EMG sample counts"""
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")
        
        # Analyze Kinematic/Kinetic data
        print("\nKINEMATIC/KINETIC DATA:")
        if len(kinematic_samples) == 0:
            print("  NO DATA")
            kinematic_stride_counts[name] = 0
        else:
            kinematic_stride_counts[name] = len(kinematic_samples)
            unique_counts = Counter(kinematic_samples)
            print(f"  Total strides: {len(kinematic_samples)}")
            print(f"  Unique sample counts: {len(unique_counts)}")
            print(f"  Min samples: {min(kinematic_samples)}")
            print(f"  Max samples: {max(kinematic_samples)}")
            print(f"  Mean samples: {np.mean(kinematic_samples):.2f}")
            print(f"  Std samples: {np.std(kinematic_samples):.2f}")
            
            if len(unique_counts) == 1:
                print(f"  ✓ NORMALIZED: All strides have exactly {list(unique_counts.keys())[0]} samples")
            elif len(unique_counts) <= 3 and np.std(kinematic_samples) < 5:
                print(f"  ✓ LIKELY NORMALIZED: Very low variance")
                print(f"  Most common counts: {unique_counts.most_common(3)}")
            else:
                print(f"  ✗ VARIABLE LENGTH: Original Hz domain likely preserved")
                print(f"  Most common counts: {unique_counts.most_common(5)}")
        
        # Analyze EMG data
        print("\nEMG DATA:")
        if len(emg_samples) == 0:
            print("  NO DATA")
            emg_stride_counts[name] = 0
        else:
            emg_stride_counts[name] = len(emg_samples)
            unique_counts = Counter(emg_samples)
            print(f"  Total strides: {len(emg_samples)}")
            print(f"  Unique sample counts: {len(unique_counts)}")
            print(f"  Min samples: {min(emg_samples)}")
            print(f"  Max samples: {max(emg_samples)}")
            print(f"  Mean samples: {np.mean(emg_samples):.2f}")
            print(f"  Std samples: {np.std(emg_samples):.2f}")
            
            if len(unique_counts) == 1:
                print(f"  ✓ NORMALIZED: All strides have exactly {list(unique_counts.keys())[0]} samples")
            elif len(unique_counts) <= 3 and np.std(emg_samples) < 5:
                print(f"  ✓ LIKELY NORMALIZED: Very low variance")
                print(f"  Most common counts: {unique_counts.most_common(3)}")
            else:
                print(f"  ✗ VARIABLE LENGTH: Original Hz domain likely preserved")
                print(f"  Most common counts: {unique_counts.most_common(5)}")

    def analyzeCriekinge(currPath="D:/EMG/processed_datasets/criekinge.pkl"):
        kinematic_samples = []
        emg_samples = []
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        directions = ['left','right','stroke']
        for currLeg in directions:
            for currPatientKinematic in currPickle['walk'][currLeg]['angle']:
                for currStrideKinematic in currPatientKinematic:
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
            
            for currPatientEMG in currPickle['walk'][currLeg]['emg']:
                for currStrideEMG in currPatientEMG:
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Criekinge", kinematic_samples, emg_samples)

    def analyzeMoghadam(currPath="D:/EMG/processed_datasets/moghadam.pkl"):
        kinematic_samples = []
        emg_samples = []
        directions = ['left','right']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currLeg in directions:
            for currPatientKinematic in currPickle['walk'][currLeg]['kinematic']:
                for currTrialKinematic in currPatientKinematic:
                    if len(currTrialKinematic) == 0:
                        continue
                    for currStrideKinematic in currTrialKinematic:
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
            
            for currPatientEMG in currPickle['walk'][currLeg]['emg']:
                for currTrialEMG in currPatientEMG:
                    if len(currTrialEMG) == 0:
                        continue
                    for currStrideEMG in currTrialEMG:
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Moghadam", kinematic_samples, emg_samples)

    def analyzeLencioni(currPath="D:/EMG/processed_datasets/lencioni.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['step up', 'step down', 'walk']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currPatientKinematic in currPickle[currActivity]['angle']:
                for currStrideKinematic in currPatientKinematic:
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
            
            for currPatientEMG in currPickle[currActivity]['emg']:
                for currStrideEMG in currPatientEMG:
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Lencioni", kinematic_samples, emg_samples)
    
    def analyzeMoreira(currPath="D:/EMG/processed_datasets/moreira.pkl"):
        kinematic_samples = []
        emg_samples = []
        directions = ['left','right']
        activities = ['walk']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            for currActivity in activities:
                for currPatientKinematic in currPickle[currActivity][currDirection]['angle']:
                    for currTrialKinematic in currPatientKinematic:
                        for currSuccessiveStrideKinematic in currTrialKinematic:
                            if len(currSuccessiveStrideKinematic) > 0:
                                kinematic_samples.append(currSuccessiveStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currTrialEMG in currPatientEMG:
                        for currSuccessiveStrideEMG in currTrialEMG:
                            if len(currSuccessiveStrideEMG) > 0:
                                emg_samples.append(currSuccessiveStrideEMG.shape[1])
        
        analyze_dataset("Moreira", kinematic_samples, emg_samples)

    def analyzeHu(currPath="D:/EMG/processed_datasets/hu.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['walk', 'ramp_up', 'ramp_down', 'stair_up', 'stair_down']
        directions = ['left','right']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currDirection in directions:
                for currPatientKinematic in currPickle[currActivity][currDirection]['angle']:
                    for currStrideKinematic in currPatientKinematic:
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currStrideEMG in currPatientEMG:
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Hu", kinematic_samples, emg_samples)

    def analyzeGrimmer(currPath="D:/EMG/processed_datasets/grimmer.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['stairUp','stairDown']
        directions = ['left','right']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currDirection in directions:
                for currPatientKinematic in currPickle[currActivity][currDirection]['angle']:
                    for currTrialKinematic in currPatientKinematic:
                        for currStrideKinematic in currTrialKinematic:
                            if len(currStrideKinematic) > 0:
                                kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currTrialEMG in currPatientEMG:
                        for currStrideEMG in currTrialEMG:
                            if len(currStrideEMG) > 0:
                                emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Grimmer", kinematic_samples, emg_samples)

    def analyzeSIAT(currPath="D:/EMG/processed_datasets/siat.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['walk', 'stair_up', 'stair_down']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for activityType in activities:
            kinematicDataDict = currPickle[activityType]['left']['angle']
            for currPatientKinematic in kinematicDataDict:
                for currSessionKinematic in currPatientKinematic:
                    for currStrideKinematic in currSessionKinematic:
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
            
            emgDataDict = currPickle[activityType]['left']['emg']
            for currPatientEMG in emgDataDict:
                for currSessionEMG in currPatientEMG:
                    for currStrideEMG in currSessionEMG:
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("SIAT", kinematic_samples, emg_samples)

    def analyzeEmbry(currPath="D:/EMG/processed_datasets/embry.pkl"):
        kinematic_samples = []
        emg_samples = []
        directions = ['left','right']
        activities = ['walk', 'rampup', 'rampdown']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in directions:
            for currActivity in activities:
                for currPatientKinematic in currPickle[currActivity][currDirection]['kinematic']:
                    for currTrialKinematic in currPatientKinematic:
                        for currStrideKinematic in currTrialKinematic:                        
                            if len(currStrideKinematic) > 0:
                                kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currTrialEMG in currPatientEMG:
                        for currStrideEMG in currTrialEMG:                        
                            if len(currStrideEMG) > 0:
                                emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Embry", kinematic_samples, emg_samples)

    def analyzeGait120(currPath="D:/EMG/processed_datasets/gait120.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 
                     'slopeDescent', 'sitToStand', 'standToSit']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currPatientKinematic in currPickle['right'][currActivity]['angle']:
                for currStrideKinematic in currPatientKinematic:
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
            
            for currPatientEMG in currPickle['right'][currActivity]['emg']:
                for currStrideEMG in currPatientEMG:
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Gait120", kinematic_samples, emg_samples)

    def analyzeCamargo(currPath="D:/EMG/processed_datasets/camargo.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['walk', 'stair', 'ramp']
        
        with open(currPath, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currPatientKinematic in currPickle['right'][currActivity]['angle']:
                for currTrialKinematic in currPatientKinematic:
                    for currStrideKinematic in currTrialKinematic:
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
            
            for currPatientEMG in currPickle['right'][currActivity]['emg']:
                for currTrialEMG in currPatientEMG:
                    for currStrideEMG in currTrialEMG:
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Camargo", kinematic_samples, emg_samples)

    def analyzeAngelidou(currPath="D:/EMG/processed_datasets/angelidou.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['walk']
        directions = ['left','right']
        
        with open(currPath, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currDirection in directions:
                for currPatientKinematic in currPickle[currActivity][currDirection]['angle']:
                    for currStrideKinematic in currPatientKinematic:
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currStrideEMG in currPatientEMG:
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Angelidou", kinematic_samples, emg_samples)

    def analyzeBacek(currPath="D:/EMG/processed_datasets/bacek.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['walk']
        directions = ['left','right']
        
        with open(currPath, 'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currDirection in directions:
                for currPatientKinematic in currPickle[currActivity][currDirection]['angle']:
                    for currTrialKinematic in currPatientKinematic:
                        for currStrideKinematic in currTrialKinematic:
                            if len(currStrideKinematic) > 0:
                                kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currTrialEMG in currPatientEMG:
                        for currStrideEMG in currTrialEMG:
                            if len(currStrideEMG) > 0:
                                emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Bacek", kinematic_samples, emg_samples)

    def analyzeMacaluso(currPath="D:/EMG/processed_datasets/macaluso.pkl"):
        kinematic_samples = []
        emg_samples = []
        activities = ['walk', 'rampup', 'rampdown']
        directions = ['right','left']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currActivity in activities:
            for currDirection in directions:
                for currPatientKinematic in currPickle[currActivity][currDirection]['kinematic']:
                    for currTrialKinematic in currPatientKinematic:
                        for currStrideKinematic in currTrialKinematic:
                            if len(currStrideKinematic) > 0:
                                kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currActivity][currDirection]['emg']:
                    for currTrialEMG in currPatientEMG:
                        for currStrideEMG in currTrialEMG:
                            if len(currStrideEMG) > 0:
                                emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("Macaluso", kinematic_samples, emg_samples)

    def analyzeK2Muse(currPath="D:/EMG/processed_datasets/k2muse.pkl"):
        kinematic_samples = []
        emg_samples = []
        direction = ['right']
        activities = ['walk', 'up_ramp', 'down_ramp']
        
        with open(currPath,'rb') as file:
            currPickle = pickle.load(file)
        
        for currDirection in direction:
            for currActivity in activities:
                for currPatientKinematic in currPickle[currDirection][currActivity]['angle']:
                    for currTrialKinematic in currPatientKinematic:
                        for currSubTrialKinematic in currTrialKinematic:
                            for currStrideKinematic in currSubTrialKinematic:
                                if len(currStrideKinematic) > 0:
                                    kinematic_samples.append(currStrideKinematic.shape[-1])
                
                for currPatientEMG in currPickle[currDirection][currActivity]['emg']:
                    for currTrialEMG in currPatientEMG:
                        for currSubTrialEMG in currTrialEMG:
                            for currStrideEMG in currSubTrialEMG:
                                if len(currStrideEMG) > 0:
                                    emg_samples.append(currStrideEMG.shape[1])
        
        analyze_dataset("K2Muse", kinematic_samples, emg_samples)

    # Run all analyses
    print("="*60)
    print("COMBINED SAMPLE COUNT ANALYSIS")
    print("="*60)
    
    datasets = [
        ("Criekinge", analyzeCriekinge),
        ("Moghadam", analyzeMoghadam),
        ("Lencioni", analyzeLencioni),
        ("Moreira", analyzeMoreira),
        ("Hu", analyzeHu),
        ("Grimmer", analyzeGrimmer),
        ("SIAT", analyzeSIAT),
        ("Embry", analyzeEmbry),
        ("Gait120", analyzeGait120),
        ("Camargo", analyzeCamargo),
        ("Angelidou", analyzeAngelidou),
        ("Bacek", analyzeBacek),
        ("Macaluso", analyzeMacaluso),
        ("K2Muse", analyzeK2Muse)
    ]
    
    for dataset_name, analyze_func in datasets:
        try:
            analyze_func()
        except Exception as e:
            print(f"\n{dataset_name}: ERROR - {e}")
            kinematic_stride_counts[dataset_name] = 0
            emg_stride_counts[dataset_name] = 0

    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET STRIDE COUNT SUMMARY")
    print("="*60)
    
    total_kinematic = sum(kinematic_stride_counts.values())
    total_emg = sum(emg_stride_counts.values())
    
    print(f"\nTotal kinematic strides: {total_kinematic:,}")
    print(f"Total EMG strides: {total_emg:,}")
    
    print(f"\n{'Dataset':<15} {'Kinematic':>15} {'EMG':>15} {'Match':>8}")
    print("-" * 58)
    
    for dataset_name in sorted(kinematic_stride_counts.keys()):
        kin_count = kinematic_stride_counts[dataset_name]
        emg_count = emg_stride_counts[dataset_name]
        match = "✓" if kin_count == emg_count else "✗"
        print(f"{dataset_name:<15} {kin_count:>15,} {emg_count:>15,} {match:>8}")
    
    print("="*60)
    
    return kinematic_stride_counts, emg_stride_counts

def main():
    analyze_sample_counts()
    #analyze_kinematic_kinetic_sample_counts()
    #resample_all_datasets()
    #analyze_emg_sample_counts()
    #syncAll()
   #checkLists()
    #investigate1()
    #check_normalization()
    
if __name__ == '__main__':
    main()