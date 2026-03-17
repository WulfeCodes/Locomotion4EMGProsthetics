import deprl
import sconegym
import gym
import argparse
from trainFM import EMGTransformer, ReplayBuffer,QNetwork, compute_impedance_torque, train_sac, train_sac_bilateral
import numpy as np
import torch
from visualizer import TrainingVisualizer
import torch.nn.functional as F

#TODO isometric actuation zeroing debug: default_activation, minimum_activation 
#TODO minimum replay buffer size 10k-50k (25k?)
#TODO prioritized experience replay -> binary tree? 
#TODO add loading functionality
#NOTE^^ training on different (t) policies actions by acting in the environment
#NOTE kinematic and impedance masks are applied at log pdf calculation and state variable representation(before Q parameterization) to prevent non used index gradient noise
#TODO github reformat
#TODO policy scheduler save and load and all network optim+scheduler form them into the dict

#TODO noise options
#TODO arg paramd saving and loading functionality :: RL save paths of policy, replayBuff and critic

def checkDis(env_num='444'):
    env = gym.make(f'sconewalk_h0{env_num}_osim-v1', clip_actions=True)
    body_mass = env.unwrapped.model.mass()
    obs=env.reset()
    print(f'input, body mass {body_mass}')
    print('new obs shape',obs.shape)

    print(f'actuator names {env_num}','='*80)
    for actuator in env.model.actuators():
        print(actuator.name())
    print(f'action vector shape {env_num}','='*80)

    print(f'{env.action_space.shape}',len(env.model.actuators()))

    dof_names = [d.name() for d in env.model.dofs()]
    muscle_names = [m.name() for m in env.model.muscles()]
    actuator_names = [a.name() for a in env.model.actuators()]

    labels = (
        [f"dof_pos_{n}" for n in dof_names] +       # dof_position_array()
        [f"dof_vel_{n}" for n in dof_names] +        # dof_velocity_array()
        [f"dof_acc_{n}" for n in dof_names] +        # derived acceleration
        [f"exc_{n}" for n in muscle_names] +          # muscle_excitation_array()
        [f"act_input_{n}" for n in actuator_names]   # actuator_input_array()
    )

    for i, label in enumerate(labels):
        print(i, label)
    print(len(labels))

    env = gym.make('sconewalk_h0918_osim-v1', clip_actions=True)
    obs=env.reset()
    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')
    print('old obs shape',obs.shape)

    print('actuator names 0918','='*80)
    for actuator in env.model.actuators():
        print(actuator.name())
    print('action vector shape 0918','='*80)

    print(f'{env.action_space.shape}',len(env.model.actuators()))

    dof_names = [d.name() for d in env.model.dofs()]
    muscle_names = [m.name() for m in env.model.muscles()]
    actuator_names = [a.name() for a in env.model.actuators()]

    labels = (
        [f"dof_pos_{n}" for n in dof_names] +       # dof_position_array()
        [f"dof_vel_{n}" for n in dof_names] +        # dof_velocity_array()
        [f"dof_acc_{n}" for n in dof_names] +        # derived acceleration
        [f"exc_{n}" for n in muscle_names] +          # muscle_excitation_array()
        [f"act_input_{n}" for n in actuator_names]   # actuator_input_array()
    )

    for i, label in enumerate(labels):
        print(i, label)
    print(len(labels))

def concatenate_actions(pred_torque,muscle_action,direction): 
    curr_ptr = 0

    if direction=='right' or direction=='left':
        full_action = np.zeros((21,))

        for i in range(pred_torque.shape[-1]):
            if (i+1)%3==0:
                full_action[curr_ptr]=pred_torque[:,i]
                curr_ptr+=1
            
        if direction=='left':
            full_action[(curr_ptr+9):] = muscle_action[9:]

        elif direction=='right':
            full_action[(curr_ptr):(curr_ptr+9)] = muscle_action[:9]

        return full_action

    elif direction == 'trans_right' or direction=='trans_left':
        full_action = np.zeros((20,))

        for i in range(pred_torque.shape[-1]):
            if (i+1)%3==0 and i>2:
                full_action[curr_ptr]=pred_torque[:,i]
                curr_ptr+=1
            
        #TODO verify with action space list
        if direction=='trans_left':
            full_action[2:11] = muscle_action[9:]
            PROSTHETIC_ZERO_INDICES = [17, 18, 19]

        elif direction=='trans_right':
            full_action[11:] = muscle_action[:9]
            PROSTHETIC_ZERO_INDICES = [8, 9, 10]


        # 10: gastroc_r  — shank/calcaneus
        # 11: soleus_r   — shank/calcaneus  
        # 12: tib_ant_r  — shank/foot
        # 19: gastroc_l
        # 20: soleus_l
        # 21: tib_ant_l
        full_action[PROSTHETIC_ZERO_INDICES] = 0.0

        return full_action

    elif direction.lower() == 'tibial_right' or direction.lower() == 'tibial_left':
        full_action = np.zeros((19,))

        full_action[0]=pred_torque[:,-1]
        curr_ptr+=1
            
        if direction=='tibial_left':
            PROSTHETIC_ZERO_INDICES = [8, 9]

        elif direction=='tibial_right':
            PROSTHETIC_ZERO_INDICES = [17, 18]

        # 10: gastroc_r  — shank/calcaneus
        # 11: soleus_r   — shank/calcaneus  
        # 12: tib_ant_r  — shank/foot
        # 19: gastroc_l
        # 20: soleus_l
        # 21: tib_ant_l
        full_action[1:] = muscle_action
        full_action[PROSTHETIC_ZERO_INDICES] = 0.0

        return full_action

    elif direction.lower() =='trans_both': 
        full_action = np.zeros((22,))

        #NOTE torques array is arranged by right then left this maps the sagittal plane actuations to the joints
        for j in range(2):
            for i in range(pred_torque[j].shape[-1]):
                if (i+1)%3==0 and i>2:
                    full_action[curr_ptr]=pred_torque[j][:,i]
                    curr_ptr+=1
        
        PROSTHETIC_ZERO_INDICES = [10, 11, 12, 19, 20, 21]

        #knee_angle_l
        # ankle_angle_l
        # knee_angle_r
        # ankle_angle_r
        # hamstrings_r
        # bifemsh_r
        # glut_max_r
        # iliopsoas_r
        # rect_fem_r
        # vasti_r
        # gastroc_r
        # soleus_r
        # tib_ant_r
        # hamstrings_l
        # bifemsh_l
        # glut_max_l
        # iliopsoas_l
        # rect_fem_l
        # vasti_l
        # gastroc_l
        # soleus_l
        # tib_ant_l
        full_action[4:] = muscle_action
        full_action[PROSTHETIC_ZERO_INDICES] = 0.0

        return full_action

    elif direction.lower() == 'tibial_both':
        full_action = np.zeros((20,))

        #NOTE torques array is arranged by right then left this maps the sagittal plane actuations to the joints
        for j in range(2):
            for i in range(pred_torque[j].shape[-1]):
                if (i+1)%3==0 and i>5:
                    full_action[curr_ptr]=pred_torque[j][:,i]
                    curr_ptr+=1
        
        PROSTHETIC_ZERO_INDICES = [8, 9, 10, 17, 18, 19]
        # 10: gastroc_r  — shank/calcaneus
        # 11: soleus_r   — shank/calcaneus  
        # 12: tib_ant_r  — shank/foot

        full_action[2:] = muscle_action
        full_action[PROSTHETIC_ZERO_INDICES] = 0.0

        return full_action

def get_sagittal(impedance_values):
    sagittal_impedances=np.zeros(9,)
    counter ==0
    for i in range(impedance_values.shape[-1]):
        if (i+1)%3==0:
            sagittal_impedances[counter]=impedance_values[i]
            counter+=1 
    return sagittal_impedances
    #TODO currently the model is getting sagittal plane 0th to second order angle values : size 3 * 3 = 9, 
    # the impedance params by nature are 27 of 3 joints * 3 axis * 3 orders 
    
def map_excitation_window(exc_window_9ch):
    """
    exc_window_9ch: np.array of shape (9, n_sim_steps)
    returns: torch.tensor of shape (13, n_sim_steps)
    """
    n = exc_window_9ch.shape[1]
    out = torch.zeros((13, n), dtype=torch.float32)

    out[0]  = torch.tensor(exc_window_9ch[5])  # Vastus Lateralis  ← vasti
    out[1]  = torch.tensor(exc_window_9ch[4])  # Rectus Femoris    ← rect_fem
    out[2]  = torch.tensor(exc_window_9ch[5])  # Vastus Medialis   ← vasti (lumped)
    out[3]  = 0.0                               # Tibialis Anterior ← below knee
    out[4]  = torch.tensor(exc_window_9ch[1])  # Biceps Femoris    ← bifemsh
    out[5]  = torch.tensor(exc_window_9ch[0])  # Semitendinosus    ← hamstrings
    out[6]  = 0.0                               # Gastroc Medialis  ← below knee
    out[7]  = 0.0                               # Gastroc Lateralis ← below knee
    out[8]  = 0.0                               # Soleus            ← below knee
    out[9]  = 0.0                               # Peroneus Longus   ← not in model
    out[10] = 0.0                               # Peroneus Brevis   ← not in model
    out[11] = 0.0                               # Gluteus Medius    ← not in model
    out[12] = torch.tensor(exc_window_9ch[2])  # Gluteus Maximus   ← glut_max

    return out  # (13, n_sim_steps)

def rearrange_obs(obs: torch.Tensor, direction_of_control='left'):
    """
    obs: 1D torch.tensor of shape (69,)
    direction_of_control: 'left' or 'right'
    Returns:
        dof_tensor: torch.tensor of shape (27,) — [adduction, rotation, flexion] x (hip, knee, ankle) x (pos, vel, acc)
        emg_tensor: torch.tensor of shape (13,) — master index order
    """
    def expand_to_plane(joints):
        out = []
        for v in joints:
            out.extend([0.0, 0.0, v.item()])
        return out

    if direction_of_control.lower() == 'right':
            pos = obs[3:6]
            vel = obs[12:15]
            acc = obs[21:24]
            leg = obs[27:36]

    elif direction_of_control.lower() == 'left':
        pos = obs[6:9]
        vel = obs[15:18]
        acc = obs[24:27]
        leg = obs[36:45]


    elif 'tibial' in direction_of_control.lower():

        pos_r = obs[3:6]
        pos_l = obs[6:9]
        vel_r = obs[12:15]
        vel_l = obs[15:18]
        acc_r = obs[21:24]
        acc_l = obs[24:27]
        leg_r = obs[27:36]
        leg_l = obs[36:45]

        dof_tensor_r = torch.tensor(
            expand_to_plane(pos_r) + expand_to_plane(vel_r) + expand_to_plane(acc_r),
            dtype=torch.float32
        )
        dof_tensor_l = torch.tensor(
            expand_to_plane(pos_l) + expand_to_plane(vel_l) + expand_to_plane(acc_l),
            dtype=torch.float32
        )

        def make_emg(leg):

            return torch.tensor([
                leg[5].item(),   # 0:  Vastus Lateralis    ← vasti (intermedius, lumped)
                leg[4].item(),   # 1:  Rectus Femoris      ← rect_fem
                leg[5].item(),   # 2:  Vastus Medialis     ← vasti (intermedius, lumped)
                0.0,   # 3:  Tibialis Anterior   ← tib_ant
                leg[1].item(),   # 4:  Biceps Femoris      ← bifemsh
                leg[0].item(),   # 5:  Semitendinosus      ← hamstrings
                leg[6].item(),   # 6:  Gastroc Medialis    ← gastroc (confirmed medialis)
                0.0,             # 7:  Gastroc Lateralis   ← not in model
                leg[7].item(),   # 8:  Soleus              ← soleus
                0.0,             # 9:  Peroneus Longus     ← not in model
                0.0,             # 10: Peroneus Brevis     ← not in model
                0.0,             # 11: Gluteus Medius      ← not in model
                leg[2].item(),   # 12: Gluteus Maximus     ← glut_max
            ], dtype=torch.float32)  # shape (13,)

        if direction_of_control.lower() == 'tibial_both':
            return (dof_tensor_r, make_emg(leg_r)), (dof_tensor_l, make_emg(leg_l))
        elif direction_of_control.lower() == 'tibial_left':
            return (dof_tensor_l, make_emg(leg_l))
        elif direction_of_control.lower() == 'tibial_right': 
            return (dof_tensor_r, make_emg(leg_r))
    
    elif 'trans' in direction_of_control.lower():
        
        pos_r = obs[3:6]
        pos_l = obs[6:9]
        vel_r = obs[12:15]
        vel_l = obs[15:18]
        acc_r = obs[21:24]
        acc_l = obs[24:27]
        leg_r = obs[27:36]
        leg_l = obs[36:45]

        dof_tensor_r = torch.tensor(
            expand_to_plane(pos_r) + expand_to_plane(vel_r) + expand_to_plane(acc_r),
            dtype=torch.float32
        )
        dof_tensor_l = torch.tensor(
            expand_to_plane(pos_l) + expand_to_plane(vel_l) + expand_to_plane(acc_l),
            dtype=torch.float32
        )

        def make_emg(leg):

            return torch.tensor([
                leg[5].item(),   # 0:  Vastus Lateralis
                leg[4].item(),   # 1:  Rectus Femoris
                leg[5].item(),   # 2:  Vastus Medialis
                0.0,             # 3:  Tibialis Anterior   ← below knee
                leg[1].item(),   # 4:  Biceps Femoris
                leg[0].item(),   # 5:  Semitendinosus
                0.0,             # 6:  Gastroc Medialis    ← below knee
                0.0,             # 7:  Gastroc Lateralis   ← below knee
                0.0,             # 8:  Soleus              ← below knee
                0.0,             # 9:  Peroneus Longus
                0.0,             # 10: Peroneus Brevis
                0.0,             # 11: Gluteus Medius
                leg[2].item(),   # 12: Gluteus Maximus
            ], dtype=torch.float32)
        if direction_of_control.lower() == 'trans_both':
            return (dof_tensor_r, make_emg(leg_r)), (dof_tensor_l, make_emg(leg_l))
        elif direction_of_control.lower() == 'trans_left':
            return (dof_tensor_l, make_emg(leg_l))
        elif direction_of_control.lower() == 'trans_right': 
            return (dof_tensor_r, make_emg(leg_r))

    # DOF
    if direction_of_control.lower() == 'right' or direction_of_control.lower() == 'left':

        dof_tensor = torch.tensor(
            expand_to_plane(pos) + expand_to_plane(vel) + expand_to_plane(acc),
            dtype=torch.float32
        )  # shape (27,)

        # EMG — leg order: hamstrings(0), bifemsh(1), glut_max(2), iliopsoas(3),
        #                   rect_fem(4), vasti(5), gastroc(6), soleus(7), tib_ant(8)

        emg_tensor = torch.tensor([
            leg[5].item(),   # 0:  Vastus Lateralis    ← vasti (intermedius, lumped)
            leg[4].item(),   # 1:  Rectus Femoris      ← rect_fem
            leg[5].item(),   # 2:  Vastus Medialis     ← vasti (intermedius, lumped)
            leg[8].item(),   # 3:  Tibialis Anterior   ← tib_ant
            leg[1].item(),   # 4:  Biceps Femoris      ← bifemsh
            leg[0].item(),   # 5:  Semitendinosus      ← hamstrings
            leg[6].item(),   # 6:  Gastroc Medialis    ← gastroc (confirmed medialis)
            0.0,             # 7:  Gastroc Lateralis   ← not in model
            leg[7].item(),   # 8:  Soleus              ← soleus
            0.0,             # 9:  Peroneus Longus     ← not in model
            0.0,             # 10: Peroneus Brevis     ← not in model
            0.0,             # 11: Gluteus Medius      ← not in model
            leg[2].item(),   # 12: Gluteus Maximus     ← glut_max
        ], dtype=torch.float32)  # shape (13,)

    return dof_tensor, emg_tensor

def rl_train_transtibial_isometric(prosthetic_controller,replay_buffer,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config,optimizers_and_schedulers,max_training_steps=100000,max_env_steps=10000,direction='left'):

    #TODO saving, loading, observations and actions

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    if direction =='left': env = gym.make('sconewalk_h0111_osim-v1', clip_actions=True)

    elif direction =='right': env = gym.make('sconewalk_h0222_osim-v1', clip_actions=True)

    else:
        print('invalid direction given')
        return

    env.action_indices = [0]

    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    n_actions = env.action_space.shape[0] - 1
    new_action_space = gym.spaces.Box(
        low=env.action_space.low[:n_actions],
        high=env.action_space.high[:n_actions],
        dtype=env.action_space.dtype
    )

    n_actions = env.observation_space.shape[0] - 1

    new_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:n_actions],
        high=env.observation_space.high[:n_actions],
        dtype=env.observation_space.dtype
    )

    agent.initialize(new_obs_space, new_action_space, seed=0)

    # load the checkpoint
    agent.load("C:/Users/vijay/OneDrive/Documents/SCONE/results/sconewalk_h0918_osimv1/260220.191743.H0918v2/checkpoints/step_12000000")
    print('agent loaded')
    # run
    curr_step = 0
    steps = 0
    episode_num = 0
    viz = TrainingVisualizer(save_dir='C:/EMG/software/plots/SAC', window=200)

    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'alpha_loss' : [],
        'log_probs': [],
        'q1_mean': [],
        'q2_mean': []
    }

    while curr_step in range(max_training_steps):
        curr_step +=steps

        obs = env.reset()
        env.unwrapped.store_next_episode()

        done = False
        steps = 1
        episode_reward = 0

        direction_obs=rearrange_obs(obs=obs,direction_of_control=f'tibial_{direction}')

        stride_emg = torch.zeros(13,100).to(prosthetic_controller.device)
        emg_window = torch.zeros(13,100).to(prosthetic_controller.device)

        stride_emg[:,0]=direction_obs[1]

        kinematic_state = direction_obs[0]

        while not done and steps<max_env_steps:
            if steps<=1:
                emg_start = steps-100
                pad_size = -emg_start
                # stride_emg is (channels, time), slice time dimension
                emg_window = F.pad(
                    stride_emg[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

            else: 
                #TODO switch per new class return functionality
                if direction == 'left':
                    emg_buffer=excitation_buffer[9:]
                elif direction == 'right':
                    emg_buffer=excitation_buffer[0:9]

                emg_window=map_excitation_window(emg_buffer).to(prosthetic_controller.device)

            #agent.step is inference with noise 
            muscle_action = agent.test_step(np.concatenate([obs[:45], obs[46:]]), steps)
            pros_action=prosthetic_controller(emg_window,kinematic_state.to(prosthetic_controller.device))

            torque_pred=compute_impedance_torque(input_kin_state=kinematic_state.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action['pred_kin_state'],pred_impedance= pros_action['pred_impedance'])

            full_action=concatenate_actions(torque_pred,muscle_action,direction=f'tibial_{direction}')

            curr_state=np.concatenate([emg_window.detach().cpu().numpy().flatten(),direction_obs[0].detach().cpu().numpy().flatten()])

            obs, reward, done, excitation_buffer = env.step(full_action)

            if direction == 'left':
                emg_buffer=excitation_buffer[9:]
            elif direction == 'right':
                emg_buffer=excitation_buffer[0:9]

            emg_window_=map_excitation_window(emg_buffer).to(prosthetic_controller.device)

            direction_obs=rearrange_obs(obs=obs,direction_of_control=f'trans_{direction}')

            next_state=np.concatenate([emg_window_.detach().cpu().numpy().flatten(),direction_obs[0].detach().cpu().numpy().flatten()])

            action_buff=np.concatenate([pros_action['pred_impedance'].detach().cpu().numpy().flatten(),pros_action['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred, torch.Tensor) else np.concatenate([pros_action['pred_impedance'].flatten(),pros_action['pred_kin_state'].flatten()])

            replay_buffer.store_transition(
                state=curr_state,
                action=action_buff,
                reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
                state_=next_state,
                done=bool(done)
            )
    
            train_sac(direction,optimizers_and_schedulers,policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
                replay_buff=replay_buffer,training_epochs=1,training_losses=training_losses)

            _reward_scalar = reward.detach().cpu().item() \
                            if isinstance(reward, torch.Tensor) else float(reward)
            episode_reward+=_reward_scalar

            viz.log_step(_reward_scalar)      # reward only — no Q args anymore
            viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

            #updating the curr_state inputs of the prosthetic model
            #had to keep both in memory for replay buffer storage

            steps+=1

        viz.log_episode()
        print(f'end of episode {episode_num} \n total steps: {steps} \n total reward: {episode_reward} \n avg step return: {episode_reward/steps}')
        episode_num+=1

        viz.save(tag=f'episode{episode_num}_end')

        if not done:
            env.unwrapped.model.write_results(
                env.unwrapped.output_dir,
                f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
            )

        print('completed! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

        env.close()
    viz.close()

def rl_train_transfemoral_isometric(prosthetic_controller,replay_buffer,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config,optimizers_and_schedulers,max_training_steps=100000,max_env_steps=10000,direction='left'):

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    if direction =='left': env = gym.make('sconewalk_h0444_osim-v1', clip_actions=True)

    elif direction =='right': env = gym.make('sconewalk_h0555_osim-v1', clip_actions=True)

    else:
        print('invalid direction given')
        return

    env.action_indices = [0,1]

    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    n_actions = env.action_space.shape[0] - 2
    new_action_space = gym.spaces.Box(
        low=env.action_space.low[:n_actions],
        high=env.action_space.high[:n_actions],
        dtype=env.action_space.dtype
    )

    n_actions = env.observation_space.shape[0] - 2

    new_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:n_actions],
        high=env.observation_space.high[:n_actions],
        dtype=env.observation_space.dtype
    )

    agent.initialize(new_obs_space, new_action_space, seed=0)

    # load the checkpoint
    agent.load("C:/Users/vijay/OneDrive/Documents/SCONE/results/sconewalk_h0918_osimv1/260220.191743.H0918v2/checkpoints/step_12000000")
    print('agent loaded')
    # run
    curr_step = 0
    steps = 0
    episode_num = 0
    viz = TrainingVisualizer(save_dir='C:/EMG/software/plots/SAC', window=200)

    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'alpha_loss' : [],
        'log_probs': [],
        'q1_mean': [],
        'q2_mean': []
    }

    while curr_step in range(max_training_steps):
        curr_step +=steps

        obs = env.reset()
        env.unwrapped.store_next_episode()

        done = False
        steps = 1
        episode_reward = 0

        direction_obs=rearrange_obs(obs=obs,direction_of_control='right')

        stride_emg = torch.zeros(13,100).to(prosthetic_controller.device)
        emg_window = torch.zeros(13,100).to(prosthetic_controller.device)

        stride_emg[:,0]=direction_obs[1]

        kinematic_state = direction_obs[0]

        while not done and steps<max_env_steps:
            if steps<=1:
                emg_start = steps-100
                pad_size = -emg_start
                # stride_emg is (channels, time), slice time dimension
                emg_window = F.pad(
                    stride_emg[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

            else: 
                #TODO switch per new class return functionality
                if direction == 'left':
                    emg_buffer=excitation_buffer[9:]
                elif direction == 'right':
                    emg_buffer=excitation_buffer[0:9]

                emg_window=map_excitation_window(emg_buffer).to(prosthetic_controller.device)

            #agent.step is inference with noise 
            muscle_action = agent.test_step(np.concatenate([obs[:45], obs[47:]]), steps)
            pros_action=prosthetic_controller(emg_window,kinematic_state.to(prosthetic_controller.device))

            torque_pred=compute_impedance_torque(input_kin_state=kinematic_state.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action['pred_kin_state'],pred_impedance= pros_action['pred_impedance'])

            full_action=concatenate_actions(torque_pred,muscle_action,direction=f'trans_{direction}')

            curr_state=np.concatenate([emg_window.detach().cpu().numpy().flatten(),direction_obs[0].detach().cpu().numpy().flatten()])

            obs, reward, done, excitation_buffer = env.step(full_action)

            if direction == 'left':
                emg_buffer=excitation_buffer[9:]
            elif direction == 'right':
                emg_buffer=excitation_buffer[0:9]

            emg_window_=map_excitation_window(emg_buffer).to(prosthetic_controller.device)

            direction_obs=rearrange_obs(obs=obs,direction_of_control=f'trans_{direction}')

            next_state=np.concatenate([emg_window_.detach().cpu().numpy().flatten(),direction_obs[0].detach().cpu().numpy().flatten()])

            action_buff=np.concatenate([pros_action['pred_impedance'].detach().cpu().numpy().flatten(),pros_action['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred, torch.Tensor) else np.concatenate([pros_action['pred_impedance'].flatten(),pros_action['pred_kin_state'].flatten()])

            replay_buffer.store_transition(
                state=curr_state,
                action=action_buff,
                reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
                state_=next_state,
                done=bool(done)
            )
    
            train_sac(direction,optimizers_and_schedulers,policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
                replay_buff=replay_buffer,training_epochs=1,training_losses=training_losses)

            _reward_scalar = reward.detach().cpu().item() \
                            if isinstance(reward, torch.Tensor) else float(reward)
            episode_reward+=_reward_scalar

            viz.log_step(_reward_scalar)      # reward only — no Q args anymore
            viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

            #updating the curr_state inputs of the prosthetic model
            #had to keep both in memory for replay buffer storage

            steps+=1

        viz.log_episode()
        print(f'end of episode {episode_num} \n total steps: {steps} \n total reward: {episode_reward} \n avg step return: {episode_reward/steps}')
        episode_num+=1

        viz.save(tag=f'episode{episode_num}_end')

        if not done:
            env.unwrapped.model.write_results(
                env.unwrapped.output_dir,
                f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
            )

        print('completed! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

        env.close()
    viz.close()

def rl_train_transfemoral_both(prosthetic_controller,replay_buffer,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config,
                                optimizers_and_schedulers,max_training_steps=100000,max_env_steps=10000):

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    env = gym.make('sconewalk_h0333_osim-v1', clip_actions=True)
    env.action_indices = [0,1,2,3]

    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    n_actions = env.action_space.shape[0] - 4
    new_action_space = gym.spaces.Box(
        low=env.action_space.low[:n_actions],
        high=env.action_space.high[:n_actions],
        dtype=env.action_space.dtype
    )

    n_actions = env.observation_space.shape[0] - 4

    new_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:n_actions],
        high=env.observation_space.high[:n_actions],
        dtype=env.observation_space.dtype
    )

    agent.initialize(new_obs_space, new_action_space, seed=0)

    # load the checkpoint
    agent.load("C:/Users/vijay/OneDrive/Documents/SCONE/results/sconewalk_h0918_osimv1/260220.191743.H0918v2/checkpoints/step_12000000")
    print('agent loaded')
    # run
    curr_step = 0
    steps = 0
    episode_num = 0
    episode_reward = 0

    viz = TrainingVisualizer(save_dir='C:/EMG/software/plots/SAC', window=200)

    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'alpha_loss' : [],
        'log_probs': [],
        'q1_mean': [],
        'q2_mean': []
    }

    while curr_step in range(max_training_steps):
    
        curr_step +=steps

        obs = env.reset()
        env.unwrapped.store_next_episode()

        done = False
        steps = 1
        max_steps = 10000

        #NOTE:: return shape :: (dof_tensor_r, make_emg(leg_r)), (dof_tensor_l, make_emg(leg_l))
        right,left=rearrange_obs(obs=obs,direction_of_control='trans_both')

        stride_emg_r = torch.zeros(13,100).to(prosthetic_controller.device)
        emg_window_r = torch.zeros(13,100).to(prosthetic_controller.device)

        stride_emg_l = torch.zeros(13,100).to(prosthetic_controller.device)
        emg_window_l = torch.zeros(13,100).to(prosthetic_controller.device)

        stride_emg_r[:,0]=right[1]
        stride_emg_l[:,0]=left[1]

        kinematic_r = right[0]
        kinematic_l = left[0]

        while not done and steps<max_steps:
            if steps<=1:
                emg_start = steps-100
                pad_size = -emg_start
                # stride_emg is (channels, time), slice time dimension
                emg_window_r = F.pad(
                    stride_emg_r[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

                emg_window_l = F.pad(
                    stride_emg_l[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

            else: 
                emg_window_r=map_excitation_window(excitation_buffer[0:9]).to(prosthetic_controller.device)
                emg_window_l=map_excitation_window(excitation_buffer[9:]).to(prosthetic_controller.device)


            #agent.step is inference with noise 
            muscle_action = agent.test_step(np.concatenate([obs[:45], obs[49:]]), steps)
            pros_action_r=prosthetic_controller(emg_window_r,kinematic_r.to(prosthetic_controller.device))

            pros_action_l=prosthetic_controller(emg_window_l,kinematic_l.to(prosthetic_controller.device))

            torque_pred_l=compute_impedance_torque(input_kin_state=kinematic_l.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action_l['pred_kin_state'],pred_impedance= pros_action_l['pred_impedance'])
            torque_pred_r=compute_impedance_torque(input_kin_state=kinematic_r.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action_r['pred_kin_state'],pred_impedance= pros_action_r['pred_impedance'])

            full_action=concatenate_actions([torque_pred_r,torque_pred_l],muscle_action,direction='trans_both')

            curr_state_r=np.concatenate([emg_window_r.detach().cpu().numpy().flatten(),right[0].detach().cpu().numpy().flatten()])
            curr_state_l=np.concatenate([emg_window_l.detach().cpu().numpy().flatten(),left[0].detach().cpu().numpy().flatten()])

            action_r=np.concatenate([pros_action_r['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_r['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_r, torch.Tensor) else np.concatenate([pros_action_r['pred_impedance'].flatten(),pros_action_r['pred_kin_state'].flatten()])
            obs, reward, done, excitation_buffer = env.step(full_action)

            right,left=rearrange_obs(obs=obs,direction_of_control='trans_both')

            next_state_l=np.concatenate([emg_window_l.detach().cpu().numpy().flatten(),kinematic_l.detach().cpu().numpy().flatten()])
            next_state_r=np.concatenate([emg_window_r.detach().cpu().numpy().flatten(),kinematic_r.detach().cpu().numpy().flatten()])

            action_r_buff=np.concatenate([pros_action_r['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_r['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_r, torch.Tensor) else np.concatenate([pros_action_r['pred_impedance'].flatten(),pros_action_r['pred_kin_state'].flatten()])
            action_l_buff=np.concatenate([pros_action_l['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_l['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_l, torch.Tensor) else np.concatenate([pros_action_l['pred_impedance'].flatten(),pros_action_l['pred_kin_state'].flatten()])

            replay_buffer.store_transition(
                state=np.concatenate([next_state_r,next_state_l]),
                action=np.concatenate([action_r_buff,action_l_buff]),
                reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
                state_=np.concatenate([next_state_r,next_state_l]),
                done=bool(done)
            )
    
            train_sac_bilateral(optimizers_and_schedulers,policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
                replay_buff=replay_buffer,training_epochs=1,training_losses=training_losses)

            _reward_scalar = reward.detach().cpu().item() \
                            if isinstance(reward, torch.Tensor) else float(reward)

            episode_reward+=_reward_scalar

            viz.log_step(_reward_scalar)      # reward only — no Q args anymore
            viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

            #updating the curr_state inputs of the prosthetic model
            #had to keep both in memory for replay buffer storage

            steps+=1

        viz.log_episode()
        print(f'end of episode {episode_num} \n total steps: {steps} \n total reward: {episode_reward} \n avg step return: {episode_reward/steps}')
        episode_num+=1
        viz.save(tag='episode_end')

        if not done:
            env.unwrapped.model.write_results(
                env.unwrapped.output_dir,
                f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
            )

        print('saved! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

        env.close()
    viz.close()
    print('completed! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

def rl_train_transtibial_both(prosthetic_controller,replay_buffer,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config,
                                optimizers_and_schedulers,max_training_steps=100000,max_env_steps=10000):

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    env = gym.make('sconewalk_h0888_osim-v1', clip_actions=True)
    env.action_indices = [0,1]

    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    n_actions = env.action_space.shape[0] - 2
    new_action_space = gym.spaces.Box(
        low=env.action_space.low[:n_actions],
        high=env.action_space.high[:n_actions],
        dtype=env.action_space.dtype
    )

    n_actions = env.observation_space.shape[0] - 2

    new_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:n_actions],
        high=env.observation_space.high[:n_actions],
        dtype=env.observation_space.dtype
    )

    agent.initialize(new_obs_space, new_action_space, seed=0)

    # load the checkpoint
    agent.load("C:/Users/vijay/OneDrive/Documents/SCONE/results/sconewalk_h0918_osimv1/260220.191743.H0918v2/checkpoints/step_12000000")
    print('agent loaded')
    # run
    curr_step = 0
    steps = 0
    episode_num = 0
    episode_reward = 0

    viz = TrainingVisualizer(save_dir='C:/EMG/software/plots/SAC', window=200)

    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'alpha_loss' : [],
        'log_probs': [],
        'q1_mean': [],
        'q2_mean': []
    }

    while curr_step in range(max_training_steps):
    
        curr_step +=steps

        obs = env.reset()
        env.unwrapped.store_next_episode()

        done = False
        steps = 1
        max_steps = 10000

        #NOTE:: return shape :: (dof_tensor_r, make_emg(leg_r)), (dof_tensor_l, make_emg(leg_l))
        right,left=rearrange_obs(obs=obs,direction_of_control='tibial_both')

        stride_emg_r = torch.zeros(13,100).to(prosthetic_controller.device)
        emg_window_r = torch.zeros(13,100).to(prosthetic_controller.device)

        stride_emg_l = torch.zeros(13,100).to(prosthetic_controller.device)
        emg_window_l = torch.zeros(13,100).to(prosthetic_controller.device)

        stride_emg_r[:,0]=right[1]
        stride_emg_l[:,0]=left[1]

        kinematic_r = right[0]
        kinematic_l = left[0]

        while not done and steps<max_steps:
            if steps<=1:
                emg_start = steps-100
                pad_size = -emg_start
                # stride_emg is (channels, time), slice time dimension
                emg_window_r = F.pad(
                    stride_emg_r[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

                emg_window_l = F.pad(
                    stride_emg_l[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

            else: 
                emg_window_r=map_excitation_window(excitation_buffer[0:9]).to(prosthetic_controller.device)
                emg_window_l=map_excitation_window(excitation_buffer[9:]).to(prosthetic_controller.device)


            #agent.step is inference with noise 
            muscle_action = agent.test_step(np.concatenate([obs[:45], obs[47:]]), steps)
            pros_action_r=prosthetic_controller(emg_window_r,kinematic_r.to(prosthetic_controller.device))

            pros_action_l=prosthetic_controller(emg_window_l,kinematic_l.to(prosthetic_controller.device))

            torque_pred_l=compute_impedance_torque(input_kin_state=kinematic_l.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action_l['pred_kin_state'],pred_impedance= pros_action_l['pred_impedance'])
            torque_pred_r=compute_impedance_torque(input_kin_state=kinematic_r.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action_r['pred_kin_state'],pred_impedance= pros_action_r['pred_impedance'])

            full_action=concatenate_actions([torque_pred_r,torque_pred_l],muscle_action,direction='tibial_both')

            curr_state_r=np.concatenate([emg_window_r.detach().cpu().numpy().flatten(),right[0].detach().cpu().numpy().flatten()])
            curr_state_l=np.concatenate([emg_window_l.detach().cpu().numpy().flatten(),left[0].detach().cpu().numpy().flatten()])

            action_r=np.concatenate([pros_action_r['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_r['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_r, torch.Tensor) else np.concatenate([pros_action_r['pred_impedance'].flatten(),pros_action_r['pred_kin_state'].flatten()])
            obs, reward, done, excitation_buffer = env.step(full_action)

            right,left=rearrange_obs(obs=obs,direction_of_control='trans_both')

            next_state_l=np.concatenate([emg_window_l.detach().cpu().numpy().flatten(),kinematic_l.detach().cpu().numpy().flatten()])
            next_state_r=np.concatenate([emg_window_r.detach().cpu().numpy().flatten(),kinematic_r.detach().cpu().numpy().flatten()])

            action_r_buff=np.concatenate([pros_action_r['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_r['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_r, torch.Tensor) else np.concatenate([pros_action_r['pred_impedance'].flatten(),pros_action_r['pred_kin_state'].flatten()])
            action_l_buff=np.concatenate([pros_action_l['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_l['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_l, torch.Tensor) else np.concatenate([pros_action_l['pred_impedance'].flatten(),pros_action_l['pred_kin_state'].flatten()])

            replay_buffer.store_transition(
                state=np.concatenate([next_state_r,next_state_l]),
                action=np.concatenate([action_r_buff,action_l_buff]),
                reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
                state_=np.concatenate([next_state_r,next_state_l]),
                done=bool(done)
            )
    
            train_sac_bilateral(optimizers_and_schedulers,policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
                replay_buff=replay_buffer,training_epochs=1,training_losses=training_losses)

            _reward_scalar = reward.detach().cpu().item() \
                            if isinstance(reward, torch.Tensor) else float(reward)

            episode_reward+=_reward_scalar

            viz.log_step(_reward_scalar)      # reward only — no Q args anymore
            viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

            #updating the curr_state inputs of the prosthetic model
            #had to keep both in memory for replay buffer storage

            steps+=1

        viz.log_episode()
        print(f'end of episode {episode_num} \n total steps: {steps} \n total reward: {episode_reward} \n avg step return: {episode_reward/steps}')
        episode_num+=1
        viz.save(tag='episode_end')

        if not done:
            env.unwrapped.model.write_results(
                env.unwrapped.output_dir,
                f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
            )

        print('saved! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

        env.close()
    viz.close()
    print('completed! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved


def rl_train_full_isometric(prosthetic_controller,replay_buff,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config,optimizers_schedulers,max_total_steps=100000,max_env_steps=10000,direction='left'):

    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'alpha_loss' : [],
        'log_probs': [],
        'q1_mean': [],
        'q2_mean': []
    }

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    if direction == 'left':
        env = gym.make('sconewalk_h0111_osim-v1', clip_actions=True)

    elif direction == 'right':
        env = gym.make('sconewalk_h0222_osim-v1', clip_actions=True)

    else:
        print('error in direction selection!')
        return
    
    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    n_actions = env.action_space.shape[0] - 3
    new_action_space = gym.spaces.Box(
        low=env.action_space.low[:n_actions],
        high=env.action_space.high[:n_actions],
        dtype=env.action_space.dtype
    )

    n_actions = env.observation_space.shape[0] - 3

    new_obs_space = gym.spaces.Box(
        low=env.observation_space.low[:n_actions],
        high=env.observation_space.high[:n_actions],
        dtype=env.observation_space.dtype
    )


    agent.initialize(new_obs_space, new_action_space, seed=0)

    # load the checkpoint
    agent.load("C:/Users/vijay/OneDrive/Documents/SCONE/results/sconewalk_h0918_osimv1/260220.191743.H0918v2/checkpoints/step_12000000")
    print('agent loaded')
    # run

    max_steps = 1000
    total_steps = 1000
    
    obs = env.reset()
    print('env reset')
    env.unwrapped.store_next_episode()

    kinematic,emg=rearrange_obs(obs=obs,direction_of_control=direction)

    stride_emg = torch.zeros(13,100).to(prosthetic_controller.device)
    emg_window = torch.zeros(13,100).to(prosthetic_controller.device)

    stride_emg[:,0]=emg
    steps = 0
    curr_step = 0

    viz = TrainingVisualizer(save_dir=f'C:/EMG/software/plots/SAC', window=200)

    while curr_step < max_total_steps:
        print('in loop')

        curr_step+=steps
        done = False
        steps = 1
        episode_reward = 0

        while not done and steps<max_env_steps:
            if steps<=1:
                emg_start = steps-100
                pad_size = -emg_start
                # stride_emg is (channels, time), slice time dimension
                emg_window = F.pad(
                    stride_emg[:, 0:steps],
                    (pad_size, 0),  # Pad time dimension (axis=1)
                    mode='replicate'
                )

            else: 
                if direction =='left':
                    emg_window=map_excitation_window(excitation_buffer[0:9]).to(prosthetic_controller.device)
                elif direction == 'right': 
                    emg_window=map_excitation_window(excitation_buffer[9:]).to(prosthetic_controller.device)

            #agent.step is inference with noise 
            muscle_action = agent.test_step(np.concatenate([obs[:45], obs[48:]]), steps)
            pros_action=prosthetic_controller(emg_window,kinematic.to(prosthetic_controller.device))
        
            torque_pred=compute_impedance_torque(input_kin_state=kinematic.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action['pred_kin_state'],pred_impedance= pros_action['pred_impedance'])
            full_action=concatenate_actions(torque_pred,muscle_action,direction)

            curr_state=np.concatenate([emg_window.detach().cpu().numpy().flatten(),kinematic.detach().cpu().numpy().flatten()])

            obs, reward, done, excitation_buffer = env.step(full_action)

            kinematic,emg=rearrange_obs(obs=obs,direction_of_control=direction)

            if direction =='left':
                emg_window=map_excitation_window(excitation_buffer[0:9]).to(prosthetic_controller.device)
            elif direction == 'right': 
                emg_window=map_excitation_window(excitation_buffer[9:]).to(prosthetic_controller.device)

            next_state=np.concatenate([emg_window.detach().cpu().numpy().flatten(),kinematic.detach().cpu().numpy().flatten()])
        
            replay_buff.store_transition(
                state=curr_state,
                action=np.concatenate([pros_action['pred_impedance'].detach().cpu().numpy().flatten(),pros_action['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred, torch.Tensor) else np.concatenate([pros_action['pred_impedance'].flatten(),pros_action['pred_kin_state'].flatten()]),
                reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
                state_=next_state,
                done=bool(done)
            )
    
            train_sac(optimizers_schedulers,policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
                replay_buff=replay_buff,training_epochs=1,training_losses=training_losses)

            _reward_scalar = reward.detach().cpu().item() \
                            if isinstance(reward, torch.Tensor) else float(reward)
            episode_reward+=_reward_scalar

            viz.log_step(_reward_scalar)      # reward only — no Q args anymore
            viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

            #updating the curr_state inputs of the prosthetic model
            #had to keep both in memory for replay buffer storage

            done = bool(done)

            steps+=1


        viz.log_episode()
        print(f'end of episode {episode_num} \n total steps: {steps} \n total reward: {episode_reward} \n avg step return: {episode_reward/steps}')

        viz.save(tag='episode_end')
        viz.close()

        if not done:
            env.unwrapped.model.write_results(
                env.unwrapped.output_dir,
                f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
            )

        print('completed! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

        env.close()

def visualize_muscle_control_models():

    viz = TrainingVisualizer(save_dir='./plots', window=200)

    training_losses = {
        'actor_loss': [],
        'q1_loss': [],
        'q2_loss': [],
        'alpha_loss' : [],
        'log_probs': [],
        'q1_mean': [],
        'q2_mean': []
    }

    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    env = gym.make('sconewalk_h0918_osim-v1', clip_actions=True)
    print(dir(env))
    print('='*100)
    print(dir(env.model))
    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    agent.initialize(env.observation_space, env.action_space, seed=0)

    # load the checkpoint
    agent.load("C:/EMG/baselines_DEPRL/sconewalk_h0918/checkpoints/step_50000000.pt")
    print('agent loaded')
    # run
    obs = env.reset()
    env.unwrapped.store_next_episode()

    done = False
    steps = 1
    max_steps = 10000

    while not done and steps<max_steps:

        #agent.step is inference with noise 
        full_action=agent(obs)

        obs, reward, terminated, info = env.step(full_action)

        #updating the curr_state inputs of the prosthetic model
        #had to keep both in memory for replay buffer storage

        done = terminated

        steps+=1

    viz.log_episode()

    viz.save(tag='episode_end')
    viz.close()

    if not done:
        env.unwrapped.model.write_results(
            env.unwrapped.output_dir,
            f"{env.unwrapped.episode:05d}_{env.unwrapped.total_reward:.3f}"
        )

    print('completed! stored at:',env.unwrapped.results_dir)  # find out where the .sto was saved

    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_dir', type=str, default='D:/EMG/postprocessed_datasets',
                       help='Directory containing pickle files')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_impedance', action='store_true',
                       help='Use impedance control with torque prediction',default=True)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    args = parser.parse_args()
    parser.add_argument('--checkpoint_path',type=str,default='C:/EMG/software/models/best_transformer_model100m.pth')
    parser.add_argument('--sac_checkpoint_path',type=str,default=None)#'C:/EMG/software/models/SAC')
    args = parser.parse_args()

    scone_emg_mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1])

    scone_kinematic_mask = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])

    prosthetic_controller = EMGTransformer(
        emg_channels=13,
        emg_window_size=100,
        kin_state_dim=27,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        predict_impedance=True,
        emg_mask=scone_emg_mask,
        kinematic_mask=scone_kinematic_mask
    ).to(args.device)

    scone_kinematic_mask_tf = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    scone_emg_mask_tf = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1])

    scone_emg_mask_tb = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1])

    scone_kinematic_mask_tb = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ])


    prosthetic_controller = EMGTransformer(
        emg_channels=13,
        emg_window_size=100,
        kin_state_dim=27,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        predict_impedance=True,
        emg_mask=scone_emg_mask_tf,
        kinematic_mask=scone_kinematic_mask_tf
    ).to(args.device)

    prosthetic_controller_tf = EMGTransformer(emg_mask=scone_emg_mask_tf,kinematic_mask=scone_kinematic_mask_tf,kinetic_mask=scone_kinematic_mask_tf).to(args.device)
    replay_buffer_tf_both = ReplayBuffer(max_size=int(1e5),input_shape=int(2*(13*100+27)),n_actions=2*(27*2))

    prosthetic_controller.eval()

    if args.sac_checkpoint_path != None:

        # ── Policy ────────────────────────────────────────────────────────────────
        policy_checkpoint = torch.load(f'{args.sac_checkpoint_path}/best_RL_transformer_model.pth')
        print('policy keys:', policy_checkpoint.keys())
        print('policy config keys:', policy_checkpoint.keys())


        policy = EMGTransformer(
            emg_channels=13,
            emg_window_size=100,
            kin_state_dim=27,
            d_model=policy_checkpoint['model_config']['d_model'],
            nhead=policy_checkpoint['model_config']['nhead'],
            num_encoder_layers=policy_checkpoint['model_config']['num_layers'],
            num_decoder_layers=policy_checkpoint['model_config']['num_layers'],
            predict_impedance=True,
            emg_mask=scone_emg_mask_tb,
            kinematic_mask=scone_kinematic_mask_tb
        ).to(args.device)

        # Restore log_alpha tensor
        policy.log_alpha = policy_checkpoint['log_alpha'].to(policy.device).requires_grad_(True)
        # Policy optimizer + scheduler
        policy_optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
        policy_optimizer.load_state_dict(policy_checkpoint['policy_optimizer_state_dict'])
        policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            policy_optimizer, T_max=args.epochs, eta_min=args.lr / 100
        )
        policy_scheduler.load_state_dict(policy_checkpoint['policy_scheduler_state_dict'])

        # log_alpha optimizer + scheduler (points at the restored tensor)
        policy_alpha_optimizer = torch.optim.AdamW([policy.log_alpha], lr=args.lr, weight_decay=0.01, eps=1e-8)
        policy_alpha_optimizer.load_state_dict(policy_checkpoint['log_alpha_optimizer'])
        policy_alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            policy_alpha_optimizer, T_max=args.epochs, eta_min=args.lr / 100
        )
        policy_alpha_scheduler.load_state_dict(policy_checkpoint['log_alpha_scheduler'])

        print('Policy + log_alpha loaded')

        # ── Q Networks ────────────────────────────────────────────────────────────
        paths = ['Q1B', 'Q2B', 'Q1T', 'Q2T']
        q_networks   = []
        q_optimizers = []
        q_schedulers = []

        for path_appendage in paths:
            checkpoint = torch.load(f'{args.sac_checkpoint_path}/{path_appendage}')
            config = checkpoint['config']

            q_net = QNetwork(**config).to(args.device)
            q_net.load_checkpoint(f'{args.sac_checkpoint_path}/{path_appendage}')

            q_optimizer = torch.optim.AdamW(q_net.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)
            q_optimizer.load_state_dict(checkpoint['optimizer'])

            q_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                q_optimizer, T_max=args.epochs, eta_min=args.lr / 100
            )
            q_scheduler.load_state_dict(checkpoint['scheduler'])

            q_networks.append(q_net)
            q_optimizers.append(q_optimizer)
            q_schedulers.append(q_scheduler)

        print(f'Loaded {len(q_networks)} Q networks')
        Q_config = config

        # ── Optimizers & Schedulers dict ──────────────────────────────────────────
        optimizers_and_schedulers = {
            'policy':            {'optimizer': policy_optimizer,       'scheduler': policy_scheduler},
            'policy_log_alpha':  {'optimizer': policy_alpha_optimizer, 'scheduler': policy_alpha_scheduler},
            'q1b':               {'optimizer': q_optimizers[0],        'scheduler': q_schedulers[0]},
            'q2b':               {'optimizer': q_optimizers[1],        'scheduler': q_schedulers[1]},
            'q1t':               {'optimizer': q_optimizers[2],        'scheduler': q_schedulers[2]},
            'q2t':               {'optimizer': q_optimizers[3],        'scheduler': q_schedulers[3]},
        }

        # ── Replay Buffer ─────────────────────────────────────────────────────────
        replay_buffer = ReplayBuffer(max_size=int(1e5), input_shape=int(13*100+27), n_actions=27*2)
        replay_buffer.load('tf_left')
        print('Loaded replay buffer')
    
    else: 
        q_network_learner1=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
        d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)
        
        q_network_learner2=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
                d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)
        
        q_network_teacher1=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
                d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)
        
        q_network_teacher2=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
                d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)

        Q_config = {
            'h_dim': 512,
            'num_bins': 54,
            'emg_channels': 13,
            'emg_window_size': 100,
            'kin_state_dim': 27,
            'action_dim': 54,
            'd_model': 50,
            'nhead': 2,
            'num_encoder_layers': 1,
            'num_decoder_layers': 1,
            'dim_feedforward': 1024,
            'dropout': 0.1
        }
            
        replay_buffer = ReplayBuffer(max_size=int(1e5),input_shape=int(13*100+27),n_actions=27*2)

        policy_optimizer = torch.optim.AdamW(prosthetic_controller.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)

        policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            policy_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )

        q1b_optimizer = torch.optim.AdamW(q_network_learner1.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)

        q1b_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            q1b_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )

        q2b_optimizer = torch.optim.AdamW(q_network_learner2.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)

        q2b_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            q2b_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )

        q1t_optimizer = torch.optim.AdamW(q_network_teacher1.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)

        q1t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            q1t_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )

        q2t_optimizer = torch.optim.AdamW(q_network_teacher2.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)

        q2t_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            q2t_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )

        policy_optimizer = torch.optim.AdamW(prosthetic_controller.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-8)

        policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            policy_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )
        policy_alpha_optimizer = torch.optim.Adam([prosthetic_controller.log_alpha], lr=3e-4)

        policy_alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            policy_alpha_optimizer, T_max=args.epochs, eta_min=args.lr/100
        )

        optimizers_and_schedulers = {
                    'policy':{'optimizer':policy_optimizer, 'scheduler':policy_scheduler},
                    'policy_log_alpha':{'optimizer':policy_alpha_optimizer,'scheduler':policy_alpha_scheduler},
                    'q1b':{'optimizer':q1b_optimizer, 'scheduler':q1b_scheduler},
                    'q2b':{'optimizer':q2b_optimizer, 'scheduler':q2b_scheduler},
                    'q1t':{'optimizer':q1t_optimizer, 'scheduler':q1t_scheduler},
                    'q2t':{'optimizer':q2t_optimizer, 'scheduler':q2t_scheduler}              
        }

    #rl_train_transfemoral_isometric(prosthetic_controller,replay_buffer,q_network_learner1,q_network_learner2,
                                #q_network_teacher1,q_network_teacher2,args,Q_config,optimizers_and_schedulers,direction='right')

    #rl_train_transtibial_isometric(prosthetic_controller,replay_buffer,q_network_learner1,q_network_learner2,
                                #q_network_teacher1,q_network_teacher2,args,Q_config,optimizers_and_schedulers,direction='right')

    #rl_train_transfemoral_both(prosthetic_controller,replay_buffer_tf_both,q_network_learner1,q_network_learner2,
                                #q_network_teacher1,q_network_teacher2,args,Q_config,optimizers_and_schedulers)

    rl_train_transtibial_both(prosthetic_controller,replay_buffer_tf_both,q_network_learner1,q_network_learner2,
                                q_network_teacher1,q_network_teacher2,args,Q_config,optimizers_and_schedulers)
    
    #rl_train_full_isometric(prosthetic_controller,replay_buffer,q_network_learner1,q_network_learner2,
             #q_network_teacher1,q_network_teacher2,args,Q_config,optimizers_and_schedulers)

if __name__ == '__main__':
    
    #checkDis('888')
    #visualize_muscle_control_models()

    main()
