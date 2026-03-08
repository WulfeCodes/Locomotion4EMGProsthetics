import deprl
import sconegym
import gym
import argparse
from trainFM import EMGTransformer, ReplayBuffer,QNetwork, compute_impedance_torque, train_sac
import numpy as np
import torch
from visualizer import TrainingVisualizer
import torch.nn.functional as F

#TODO isometric actuation zeroing debug: default_activation, minimum_activation 
#TODO minimum replay buffer size 10k-50k (25k?)
#TODO prioritized experience replay -> binary tree? 
#TODO update step after each online step

#NOTE^^ training on different (t) policies actions by acting in the environment
#TODO Q network optimizer and scheduler, EMG Optimizer
#TODO Q network transformer       
#TODO impedance loss
#TODO custom class inheriting from gaitgym, with specific clipping indexing by parameterized step
#TODO saving functionality :: RL save paths of policy, replayBuff and critic

def checkDis():
    env = gym.make('sconewalk_h0333_osim-v1', clip_actions=True)
    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    print('actuator names','='*80)
    for actuator in env.model.actuators():
        print(actuator.name())
    print('action vector shape','='*80)

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

    env = gym.make('sconewalk_h0918_osim-v1', clip_actions=True)
    body_mass = env.unwrapped.model.mass()
    print(f'input, body mass {body_mass}')

    print('actuator names','='*80)
    for actuator in env.model.actuators():
        print(actuator.name())
    print('action vector shape','='*80)

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


def concatenate_actions(pred_torque,muscle_action,direction): 

    #NOTE this 21 is the action vector shape
    if direction!='trans_both':
        full_action = np.zeros((21,))

        curr_ptr=0
        for i in range(pred_torque.shape[0]):
            if (i+1)%3==0:
                full_action[curr_ptr]=pred_torque[i]
                curr_ptr+=1
        
        full_action[3:] = muscle_action

        return full_action
    else: 
        full_action = np.zeros((22,))

        curr_ptr=0
        #NOTE torques array is arranged by right then left
        for j in range(2):
            for i in range(pred_torque[j].shape[0]):
                if (i+1)%3==0 and i>2:
                    full_action[curr_ptr]=pred_torque[i]
                    curr_ptr+=1
        
        full_action[4:] = muscle_action

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

    #NOTE transfemoral experiment will be done with single controller
    #TODO should we include the hip information? right now we are 

    elif direction_of_control.lower() == 'trans_both':
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
        return (dof_tensor_r, make_emg(leg_r)), (dof_tensor_l, make_emg(leg_l))

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

def rl_train_transfemoral_both(prosthetic_controller,replay_buffer,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config):

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
    env = gym.make('sconewalk_h0333_osim-v1', clip_actions=True)
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
    obs = env.reset()
    env.unwrapped.store_next_episode()

    done = False
    steps = 1
    max_steps = 10000

#(dof_tensor_r, make_emg(leg_r)), (dof_tensor_l, make_emg(leg_l))
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
            #TODO switch per new class return functionality
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
        Q1_m(curr_state_r,action_r)

        #TODO test inference on Q transformer

        obs, reward, done, excitation_buffer = env.step(full_action)

        right,left=rearrange_obs(obs=obs,direction_of_control='trans_both')

        next_state_l=np.concatenate([emg_window_l.detach().cpu().numpy().flatten(),kinematic_l.detach().cpu().numpy().flatten()])
        next_state_r=np.concatenate([emg_window_r.detach().cpu().numpy().flatten(),kinematic_r.detach().cpu().numpy().flatten()])

        #TODO figure out buffer and Q handling of diff legs
        action_r_buff=np.concatenate([pros_action_r['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_r['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_r, torch.Tensor) else np.concatenate([pros_action_r['pred_impedance'].flatten(),pros_action_r['pred_kin_state'].flatten()])
        action_l_buff=np.concatenate([pros_action_l['pred_impedance'].detach().cpu().numpy().flatten(),pros_action_l['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred_l, torch.Tensor) else np.concatenate([pros_action_l['pred_impedance'].flatten(),pros_action_l['pred_kin_state'].flatten()]),

        prosthetic_controller.replay_buffer.store_transition(
            state=np.concatenate([next_state_r,next_state_l]),
            action=np.concatenate([action_r_buff,action_l_buff]),
            reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
            state_=np.concatenate([next_state_r,next_state_l]),
            done=bool(done)
        )
 
        train_sac(policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
              replay_buff=replay_buffer,training_epochs=1,training_losses=training_losses)

        input('stepped')

        _reward_scalar = reward.detach().cpu().item() \
                        if isinstance(reward, torch.Tensor) else float(reward)

        viz.log_step(_reward_scalar)      # reward only — no Q args anymore
        viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

        #updating the curr_state inputs of the prosthetic model
        #had to keep both in memory for replay buffer storage

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

def rl_train(prosthetic_controller,Q1_b,Q2_b,Q1_m,Q2_m,args,critic_config,direction='left'):

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
    env = gym.make('sconewalk_h0888_osim-v1', clip_actions=True)
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
    obs = env.reset()
    env.unwrapped.store_next_episode()

    done = False
    steps = 1
    max_steps = 10000

    kinematic,emg=rearrange_obs(obs=obs,direction_of_control=direction)

    stride_emg = torch.zeros(13,100).to(prosthetic_controller.device)
    emg_window = torch.zeros(13,100).to(prosthetic_controller.device)

    stride_emg[:,0]=emg

    while not done and steps<max_steps:
        if steps<100:
            emg_start = steps-100
            pad_size = -emg_start
            # stride_emg is (channels, time), slice time dimension
            emg_window = F.pad(
                stride_emg[:, 0:steps],
                (pad_size, 0),  # Pad time dimension (axis=1)
                mode='replicate'
            )

        else: 
            emg_window[:, :-1] = emg_window[:, 1:]
            emg_window[:, -1] = emg

        #agent.step is inference with noise 
        muscle_action = agent.test_step(np.concatenate([obs[:45], obs[48:]]), steps)
        pros_action=prosthetic_controller(emg_window,kinematic.to(prosthetic_controller.device))

    
        torque_pred=compute_impedance_torque(input_kin_state=kinematic.to(prosthetic_controller.device).unsqueeze(dim=0), pred_kin_state=pros_action['pred_kin_state'],pred_impedance= pros_action['pred_impedance'])

        full_action=concatenate_actions(torque_pred,muscle_action,direction)

        curr_state=np.concatenate([emg_window.detach().cpu().numpy().flatten(),kinematic.detach().cpu().numpy().flatten()])

        obs, reward, terminated, info = env.step(full_action)

        kinematic,emg=rearrange_obs(obs=obs,direction_of_control=direction)

        next_state=np.concatenate([emg_window.detach().cpu().numpy().flatten(),kinematic.detach().cpu().numpy().flatten()])
    
        prosthetic_controller.replay_buffer.store_transition(
            state=curr_state,
            action=np.concatenate([pros_action['pred_impedance'].detach().cpu().numpy().flatten(),pros_action['pred_kin_state'].detach().cpu().numpy().flatten()]) if isinstance(torque_pred, torch.Tensor) else np.concatenate([pros_action['pred_impedance'].flatten(),pros_action['pred_kin_state'].flatten()]),
            reward=reward.detach().cpu().item() if isinstance(reward, torch.Tensor) else float(reward),
            state_=next_state,
            done=bool(terminated)
        )
 
        train_sac(policy_args=args,critic_args=critic_config,Policy=prosthetic_controller,QNetwork_base1=Q1_b,QNetwork_base2=Q2_b,QNetwork_target1=Q1_m,QNetwork_target2=Q2_m,
              replay_buff=prosthetic_controller.replay_buffer,training_epochs=1,training_losses=training_losses)

        _reward_scalar = reward.detach().cpu().item() \
                        if isinstance(reward, torch.Tensor) else float(reward)

        viz.log_step(_reward_scalar)      # reward only — no Q args anymore
        viz.log_losses(training_losses)   # losses + q1_mean + q2_mean all in one

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
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=8)
    args = parser.parse_args()
    parser.add_argument('--checkpoint_path',type=str,default='C:/EMG/software/models/best_transformer_model100m.pth')
    args = parser.parse_args()

    scone_emg_mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1])

    scone_kinematic_mask = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])

    scone_kinematic_mask_tf = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 0, 1],
    ])

    scone_emg_mask_tf = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1])

    prosthetic_controller = EMGTransformer(emg_mask=scone_emg_mask_tf,kinematic_mask=scone_kinematic_mask_tf,kinetic_mask=scone_kinematic_mask_tf).to(args.device)
    path=torch.load(args.checkpoint_path)
    #prosthetic_controller.load_state_dict(path['model_state_dict'])

    #checkDis()

    prosthetic_controller.eval()


    q_network_learner1=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
            d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)
    
    q_network_learner2=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
            d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)
    
    q_network_teacher1=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
            d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)
    
    q_network_teacher2=QNetwork(h_dim=512,num_bins=1,emg_channels=13,emg_window_size=100,kin_state_dim=27,action_dim=54,
            d_model=50,nhead=2,num_encoder_layers=1,num_decoder_layers=1,dim_feedforward=1024,dropout=0.1)


    Q_config = {'h_dim':512,
                'num_bins':54,
                'emg_channels':13,
                'emg_window_size':100,
                'kin_state_dim':27,
                'action_dim':54,
                'd_model':50,
                'nhead':2,
                'num_encoder_layers':1,
                'num_decoder_layers':1,
                'dim_feedforward':1024,
                'dropout':0.1}
    
    replay_buffer = ReplayBuffer(max_size=int(1e6),input_shape=int(13*100+27),n_actions=27*2)
    replay_buffer_tf_both = ReplayBuffer(max_size=int(1e6),input_shape=int(13*100+27),n_actions=27*2)

    rl_train_transfemoral_both(prosthetic_controller,replay_buffer_tf_both,q_network_learner1,q_network_learner2,q_network_teacher1,q_network_teacher2,args,Q_config)

if __name__ == '__main__':
    #checkDis()
    #visualize_muscle_control_models()

    main()
