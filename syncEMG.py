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
        leftKineticMask = currPickle['mask']['left']['kinetic']
        leftKinematicMask = currPickle['mask']['left']['angle']

        rightEMGMask = currPickle['mask']['right']['emg']
        rightKineticMask = currPickle['mask']['right']['kinetic']
        rightKinematicMask = currPickle['mask']['right']['angle']

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
        print(currPickle['masks'].keys())
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
        print(currPickle['mask']['left'].keys())
        kinematicLeftMask = currPickle['mask']['left']['angle']
        emgLeftMask = currPickle['mask']['left'] ['emg']
        kinematicRightMask = currPickle['mask']['right']['angle']
        emgRightMask = currPickle['mask']['right']['emg']

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

def analyze_sample_counts():
    """Analyze both kinematic/kinetic and EMG sample counts across all datasets"""
    
    # Dictionaries to store stride counts for each dataset
    kinematic_stride_counts = {}
    emg_stride_counts = {}
    
import pickle
import numpy as np
from collections import Counter

# Global dictionaries to store counts
kinematic_stride_counts = {}
emg_stride_counts = {}
patient_counts = {}

def analyze_dataset(name, kinematic_samples, emg_samples, num_patients):
    """Analyze both kinematic and EMG sample counts"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Number of patients: {num_patients}")
    
    # Store patient count
    patient_counts[name] = num_patients
    
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
    num_patients = len(currPickle['walk']['left']['angle'])
    
    for currLeg in directions:
        for currPatientKinematic, currPatientEMG in zip(currPickle['walk'][currLeg]['angle'], 
                                                         currPickle['walk'][currLeg]['emg']):
            for currStrideKinematic, currStrideEMG in zip(currPatientKinematic, currPatientEMG):
                if len(currStrideKinematic) > 0:
                    kinematic_samples.append(currStrideKinematic.shape[-1])
                if len(currStrideEMG) > 0:
                    emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Criekinge", kinematic_samples, emg_samples, num_patients)

def analyzeMoghadam(currPath="D:/EMG/processed_datasets/moghadam.pkl"):
    kinematic_samples = []
    emg_samples = []
    directions = ['left','right']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['kinematic'])
    
    for currLeg in directions:
        for currPatientKinematic, currPatientEMG in zip(currPickle['walk'][currLeg]['kinematic'],
                                                         currPickle['walk'][currLeg]['emg']):
            for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                if len(currTrialKinematic) == 0:
                    continue
                for currStrideKinematic, currStrideEMG in zip(currTrialKinematic, currTrialEMG):
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Moghadam", kinematic_samples, emg_samples, num_patients)

def analyzeLencioni(currPath="D:/EMG/processed_datasets/lencioni.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['step up', 'step down', 'walk']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['angle'])
    
    for currActivity in activities:
        for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity]['angle'],
                                                         currPickle[currActivity]['emg']):
            for currStrideKinematic, currStrideEMG in zip(currPatientKinematic, currPatientEMG):
                if len(currStrideKinematic) > 0:
                    kinematic_samples.append(currStrideKinematic.shape[-1])
                if len(currStrideEMG) > 0:
                    emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Lencioni", kinematic_samples, emg_samples, num_patients)

def analyzeMoreira(currPath="D:/EMG/processed_datasets/moreira.pkl"):
    kinematic_samples = []
    emg_samples = []
    directions = ['left','right']
    activities = ['walk']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['angle'])
    
    for currDirection in directions:
        for currActivity in activities:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['angle'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                    for currSuccessiveStrideKinematic, currSuccessiveStrideEMG in zip(currTrialKinematic, currTrialEMG):
                        if len(currSuccessiveStrideKinematic) > 0:
                            kinematic_samples.append(currSuccessiveStrideKinematic.shape[-1])
                        if len(currSuccessiveStrideEMG) > 0:
                            emg_samples.append(currSuccessiveStrideEMG.shape[1])
    
    analyze_dataset("Moreira", kinematic_samples, emg_samples, num_patients)

def analyzeHu(currPath="D:/EMG/processed_datasets/hu.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['walk', 'ramp_up', 'ramp_down', 'stair_up', 'stair_down']
    directions = ['left','right']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['angle'])
    
    for currActivity in activities:
        for currDirection in directions:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['angle'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currStrideKinematic, currStrideEMG in zip(currPatientKinematic, currPatientEMG):
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Hu", kinematic_samples, emg_samples, num_patients)

def analyzeGrimmer(currPath="D:/EMG/processed_datasets/grimmer.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['stairUp','stairDown']
    directions = ['left','right']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['stairUp']['left']['angle'])
    
    for currActivity in activities:
        for currDirection in directions:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['angle'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                    for currStrideKinematic, currStrideEMG in zip(currTrialKinematic, currTrialEMG):
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Grimmer", kinematic_samples, emg_samples, num_patients)

def analyzeSIAT(currPath="D:/EMG/processed_datasets/siat.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['walk', 'stair_up', 'stair_down']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['angle'])
    
    for activityType in activities:
        kinematicDataDict = currPickle[activityType]['left']['angle']
        emgDataDict = currPickle[activityType]['left']['emg']
        for currPatientKinematic, currPatientEMG in zip(kinematicDataDict, emgDataDict):
            for currSessionKinematic, currSessionEMG in zip(currPatientKinematic, currPatientEMG):
                for currStrideKinematic, currStrideEMG in zip(currSessionKinematic, currSessionEMG):
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("SIAT", kinematic_samples, emg_samples, num_patients)

def analyzeEmbry(currPath="D:/EMG/processed_datasets/embry.pkl"):
    kinematic_samples = []
    emg_samples = []
    directions = ['left','right']
    activities = ['walk', 'rampup', 'rampdown']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['kinematic'])
    
    for currDirection in directions:
        for currActivity in activities:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['kinematic'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                    for currStrideKinematic, currStrideEMG in zip(currTrialKinematic, currTrialEMG):
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Embry", kinematic_samples, emg_samples, num_patients)

def analyzeGait120(currPath="D:/EMG/processed_datasets/gait120.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['levelWalking', 'stairAscent', 'stairDescent', 'slopeAscent', 
                 'slopeDescent', 'sitToStand', 'standToSit']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['right']['levelWalking']['angle'])
    
    for currActivity in activities:
        for currPatientKinematic, currPatientEMG in zip(currPickle['right'][currActivity]['angle'],
                                                         currPickle['right'][currActivity]['emg']):
            for currStrideKinematic, currStrideEMG in zip(currPatientKinematic, currPatientEMG):
                if len(currStrideKinematic) > 0:
                    kinematic_samples.append(currStrideKinematic.shape[-1])
                if len(currStrideEMG) > 0:
                    emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Gait120", kinematic_samples, emg_samples, num_patients)

def analyzeCamargo(currPath="D:/EMG/processed_datasets/camargo.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['walk', 'stair', 'ramp']
    
    with open(currPath, 'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['right']['walk']['angle'])
    
    for currActivity in activities:
        for currPatientKinematic, currPatientEMG in zip(currPickle['right'][currActivity]['angle'],
                                                         currPickle['right'][currActivity]['emg']):
            for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                for currStrideKinematic, currStrideEMG in zip(currTrialKinematic, currTrialEMG):
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Camargo", kinematic_samples, emg_samples, num_patients)

def analyzeAngelidou(currPath="D:/EMG/processed_datasets/angelidou.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['walk']
    directions = ['left','right']
    
    with open(currPath, 'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['angle'])
    
    for currActivity in activities:
        for currDirection in directions:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['angle'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currStrideKinematic, currStrideEMG in zip(currPatientKinematic, currPatientEMG):
                    if len(currStrideKinematic) > 0:
                        kinematic_samples.append(currStrideKinematic.shape[-1])
                    if len(currStrideEMG) > 0:
                        emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Angelidou", kinematic_samples, emg_samples, num_patients)

def analyzeBacek(currPath="D:/EMG/processed_datasets/bacek.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['walk']
    directions = ['left','right']
    
    with open(currPath, 'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['angle'])
    
    for currActivity in activities:
        for currDirection in directions:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['angle'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                    for currStrideKinematic, currStrideEMG in zip(currTrialKinematic, currTrialEMG):
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Bacek", kinematic_samples, emg_samples, num_patients)

def analyzeMacaluso(currPath="D:/EMG/processed_datasets/macaluso.pkl"):
    kinematic_samples = []
    emg_samples = []
    activities = ['walk', 'rampup', 'rampdown']
    directions = ['right','left']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['walk']['left']['kinematic'])
    
    for currActivity in activities:
        for currDirection in directions:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currActivity][currDirection]['kinematic'],
                                                             currPickle[currActivity][currDirection]['emg']):
                for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                    for currStrideKinematic, currStrideEMG in zip(currTrialKinematic, currTrialEMG):
                        if len(currStrideKinematic) > 0:
                            kinematic_samples.append(currStrideKinematic.shape[-1])
                        if len(currStrideEMG) > 0:
                            emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("Macaluso", kinematic_samples, emg_samples, num_patients)

def analyzeK2Muse(currPath="D:/EMG/processed_datasets/k2muse.pkl"):
    kinematic_samples = []
    emg_samples = []
    direction = ['right']
    activities = ['walk', 'up_ramp', 'down_ramp']
    
    with open(currPath,'rb') as file:
        currPickle = pickle.load(file)
    
    num_patients = len(currPickle['right']['walk']['angle'])
    
    for currDirection in direction:
        for currActivity in activities:
            for currPatientKinematic, currPatientEMG in zip(currPickle[currDirection][currActivity]['angle'],
                                                             currPickle[currDirection][currActivity]['emg']):
                for currTrialKinematic, currTrialEMG in zip(currPatientKinematic, currPatientEMG):
                    for currSubTrialKinematic, currSubTrialEMG in zip(currTrialKinematic, currTrialEMG):
                        for currStrideKinematic, currStrideEMG in zip(currSubTrialKinematic, currSubTrialEMG):
                            if len(currStrideKinematic) > 0:
                                kinematic_samples.append(currStrideKinematic.shape[-1])
                            if len(currStrideEMG) > 0:
                                emg_samples.append(currStrideEMG.shape[1])
    
    analyze_dataset("K2Muse", kinematic_samples, emg_samples, num_patients)

# Run all analyses
def count():
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
            patient_counts[dataset_name] = 0

    # Print summary statistics
    print("\n" + "="*80)
    print("COMPREHENSIVE DATASET SUMMARY")
    print("="*80)

    total_kinematic = sum(kinematic_stride_counts.values())
    total_emg = sum(emg_stride_counts.values())
    total_patients = sum(patient_counts.values())

    print(f"\nOverall Totals:")
    print(f"  Total patients across all datasets: {total_patients:,}")
    print(f"  Total kinematic strides: {total_kinematic:,}")
    print(f"  Total EMG strides: {total_emg:,}")

    print(f"\n{'Dataset':<15} {'Patients':>10} {'%':>7} {'Kin Strides':>12} {'%':>7} {'EMG Strides':>12} {'%':>7} {'Match':>8}")
    print("-" * 95)

    for dataset_name in sorted(kinematic_stride_counts.keys()):
        patients = patient_counts[dataset_name]
        kin_count = kinematic_stride_counts[dataset_name]
        emg_count = emg_stride_counts[dataset_name]
        
        patient_pct = (patients / total_patients * 100) if total_patients > 0 else 0
        kin_pct = (kin_count / total_kinematic * 100) if total_kinematic > 0 else 0
        emg_pct = (emg_count / total_emg * 100) if total_emg > 0 else 0
        
        match = "✓" if kin_count == emg_count else "✗"
        
        print(f"{dataset_name:<15} {patients:>10,} {patient_pct:>6.1f}% {kin_count:>12,} {kin_pct:>6.1f}% {emg_count:>12,} {emg_pct:>6.1f}% {match:>8}")

    print("="*80)
    print(f"\n{'TOTALS':<15} {total_patients:>10,} {'100.0%':>7} {total_kinematic:>12,} {'100.0%':>7} {total_emg:>12,} {'100.0%':>7}")
    print("="*80)

def main():
    count()
    #analyze_kinematic_kinetic_sample_counts()
    #resample_all_datasets()
    #analyze_emg_sample_counts()
    #syncAll()
   #checkLists()
    
if __name__ == '__main__':
    main()