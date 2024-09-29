import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from rsl_rl.storage.rollout_dataset import RolloutDataset

def main(args):
    dataset = RolloutDataset(
        data_dir= args.data_dir if args.data_dirs is None else args.data_dirs,
        num_envs= args.num_envs,
        subset_traj= (args.traj, None),
        load_data_to_device= False,
        random_shuffle_traj_order= False,
    )
    # These variables are manually computed from env_cfg
    forward_depth_shape = (1, 48, 64)
    forward_depth_slice = slice(48, 48 + np.prod(forward_depth_shape))
    # action_history = None
    timestep_i = 0
    last_len = 0
    while True:
        transitions, infos = dataset.get_transition_batch()
        if transitions is None:
            break
        plt.clf()
        observations = transitions.observation
        forward_depth = observations[:, forward_depth_slice].reshape(-1, *forward_depth_shape)
        plt.imshow(forward_depth[0, 0], cmap= "gray", vmin= 0, vmax= 1)
        # actions = transitions.action
        # if action_history is None:
        #     action_history = np.zeros((100, actions.shape[1]))
        # action_history = np.concatenate((action_history[1:], actions), axis= 0)
        # for joint_idx in range(actions.shape[1]):
        #     plt.plot(action_history[:, joint_idx], label= f"joint {joint_idx}")
        # plt.ylim(-1., 1.)
        # plt.legend(loc= "upper left")
        plt.draw()
        plt.pause(0.01)
        print(timestep_i, "frames, traj_i:", dataset.current_traj_dirs[0].split("_")[-1], ", last_traj_len:", last_len, "               ", end= "\r")
        if transitions.done:
            last_len = timestep_i
            timestep_i = 0
        else:
            timestep_i += 1
    print("replay finished")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
        type= str,
        default= None,
    )
    parser.add_argument("--data_dirs",
        type= str,
        nargs= "+",
        default= None,
    )
    parser.add_argument("--num_envs",
        type= int,
        default= 1,
    )
    parser.add_argument("--traj",
        type= int,
        default= 0,
        help= "The trajectory index start to play.",
    )
    
    args = parser.parse_args()
    main(args)
