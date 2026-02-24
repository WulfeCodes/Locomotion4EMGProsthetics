import deprl
import sconegym
import gym
import argparse
from trainFM import EMGTransformer
import numpy as np
import torch
import torch.nn.functional as F


def concatenate_input(prosthetic_action,muscle_action,direction): 

    pred_torque=prosthetic_action['pred_impedance'].flatten()
    #NOTE this 24 is the action vector shape
    saggital_plane_values = np.zeros((3,))
    full_action = np.zeros((24,))

    for i in range(pred_torque.shape[0]):
        if (i+1)%3==0:
            saggital_plane_values[(i+1)%3]=pred_torque[i]
    
    if direction.lower() == 'left':
        full_action[:3]=saggital_plane_values
        full_action[6:] = muscle_action

    elif direction.lower() == 'right': 
        full_action[3:6]=saggital_plane_values
        full_action[6:] = muscle_action

    return full_action

def rearrange_input(obs: torch.Tensor, direction_of_control='left'):
    """
    obs: 1D torch.tensor of shape (69,)
    direction_of_control: 'left' or 'right'
    Returns:
        dof_tensor: torch.tensor of shape (27,) — [adduction, rotation, flexion] x (hip, knee, ankle) x (pos, vel, acc)
        emg_tensor: torch.tensor of shape (13,) — master index order
    """
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

    # DOF
    def expand_to_plane(joints):
        out = []
        for v in joints:
            out.extend([0.0, 0.0, v.item()])
        return out

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


def visualize(prosthetic_controller,direction='left'):
    agent = deprl.custom_agents.dep_factory(3, deprl.custom_mpo_torch.TunedMPO())(
        replay=deprl.custom_replay_buffers.AdaptiveEnergyBuffer(
            return_steps=1, batch_size=256, steps_between_batches=1000,
            batch_iterations=30, steps_before_batches=2e5, num_acts=18
        )
    )

    # create the environment to initialize the agent
    env = gym.make('sconewalk_h0777_osim-v1', clip_actions=True)
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

    n_actions = env.action_space.shape[0] - 6
    new_action_space = gym.spaces.Box(
        low=env.action_space.low[:n_actions],
        high=env.action_space.high[:n_actions],
        dtype=env.action_space.dtype
    )

    n_actions = env.observation_space.shape[0] - 6

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

    kinematic,emg=rearrange_input(obs=obs)
    stride_emg = [[] for _ in range(13)]
    stride_emg[:].append(emg)

    stride_emg = torch.zeros(13,100).to(prosthetic_controller.device)

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
        muscle_action = agent.test_step(np.concatenate([obs[:45], obs[51:]]), steps)
        pros_action=prosthetic_controller(emg_window,kinematic.to(prosthetic_controller.device))

        full_action=concatenate_input(pros_action,muscle_action,direction)

        obs, reward, terminated, info = env.step(full_action)

        kinematic,emg=rearrange_input(obs=obs)

        done = terminated

        steps+=1

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
    parser.add_argument('--checkpoint_path',type=str,default='C:/EMG/models/best_transformer_model100m.pth')
    args = parser.parse_args()

    scone_emg_mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1])

    scone_kinematic_mask = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ])
    prosthetic_controller = EMGTransformer(emg_mask=scone_emg_mask,kinematic_mask=scone_kinematic_mask,kinetic_mask=scone_kinematic_mask).to(args.device)
    path=torch.load(args.checkpoint_path)
    #prosthetic_controller.load_state_dict(path['model_state_dict'])

    prosthetic_controller.eval()
    visualize(prosthetic_controller)

if __name__ == '__main__':
    main()
