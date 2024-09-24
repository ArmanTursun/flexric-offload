import xapp_sdk as ric
import time
import threading
from aggr_data import AggrData
import numpy as np
import csv

import time
from datetime import datetime
from PPO import PPO

#############################################
###  TODO
###  1. stats are {} -> np          -> done
###  2. ric ctril handle stats      -> done
###  3. reward function             -> done
###  4. ddpg for states             -> done
###  5. check actor and critic      -> done
###  6. make sure drl can run       -> done
###  7. run with real ran
#############################################

#############################
#### Global Variables
#############################

# Global dictionary to store bler and energy data
# It keeps an window of 1000 of most recent data
global_ue_aggr_data = AggrData(10)

# Global lock to ensure thread-safe access to the global dictionary
global_lock = threading.Lock()

# DDPG Parameters
#state_dim = 6  # BLER and energy
#action_dim = 2  # Weight for BLER and energy
#actor_lr = 1e-5
#critic_lr = 1e-5
#gamma = 0.995
#tau = 0.001
#buffer_size = 10000
#batch_size = 64
#actor_hidden_units=(256, 256)
#critic_hidden_units=(256, 256)
ppo_agent = PPO(state_dim=6, action_dim=1, lr_actor=0.0003, lr_critic=0.001,
                    gamma=0.99, K_epochs=10, eps_clip=0.2,
                    has_continuous_action_space=True)


class StateNormalizer:
    def __init__(self):
        # Dynamic min/max for energy stats
        self.energy_min = float('inf')
        self.energy_max = float('-inf')

    def update_energy(self, energy_mean, energy_max, energy_min):
        # Update the min and max for energy values dynamically
        self.energy_min = min(self.energy_min, energy_min)
        self.energy_max = max(self.energy_max, energy_max)

    def normalize_bler_skewness(self, bler_skewness):
        # Tanh normalization to ensure skewness is between 0 and 1
        return (np.tanh(bler_skewness) + 1) / 2

    def normalize_energy_skewness(self, energy_skewness):
        # Tanh normalization to ensure skewness is between 0 and 1
        return (np.tanh(energy_skewness) + 1) / 2

    def normalize_energy(self, value):
        # Min-max normalization for energy stats
        if self.energy_max == self.energy_min:
            return 0  # Avoid division by zero
        return (value - self.energy_min) / (self.energy_max - self.energy_min)

    def normalize_state(self, bler_mean, bler_max, bler_min, bler_skewness,
                        energy_mean, energy_max, energy_min, energy_skewness):
        # Normalize the BLER stats (they are already between 0 and 1)
        normalized_bler_mean = bler_mean
        normalized_bler_max = bler_max
        normalized_bler_min = bler_min
        normalized_bler_skewness = self.normalize_bler_skewness(bler_skewness)

        # Update energy min/max before normalization
        self.update_energy(energy_mean, energy_max, energy_min)

        # Normalize energy stats
        normalized_energy_mean = self.normalize_energy(energy_mean)
        normalized_energy_max = self.normalize_energy(energy_max)
        normalized_energy_min = self.normalize_energy(energy_min)
        normalized_energy_skewness = self.normalize_energy_skewness(energy_skewness)

        # Return the normalized state
        return np.array([normalized_bler_mean, normalized_bler_max, normalized_bler_min, #normalized_bler_skewness,
                         normalized_energy_mean, normalized_energy_max, normalized_energy_min])#, normalized_energy_skewness])


##################################################
#### MACCallback Class
##################################################

class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
        self.ind_num = 0

    def handle(self, ind):
        # save bler and energy values of each tbs of each ue into aggr_data
        if len(ind.ue_stats) > 0:
            ue_stats = ind.ue_stats[0]

            # Add the new BLER and energy data to the AggrData object
            #with global_lock:
            if (self.ind_num != 0):
                global_ue_aggr_data.add_bler(ue_stats.ul_bler, ind.tstamp)
                global_ue_aggr_data.add_energy(ue_stats.dl_bler, ind.tstamp)
            #print(ue_stats.ul_bler, ue_stats.dl_bler)
            self.ind_num = 1

            
####################
####  init RIC
####################

ric.init()
time.sleep(1)
conn_id = 0
conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for conn_id in range(0, len(conn)):
    print("Global E2 Node [" + str(conn_id) + "]: PLMN MCC = " + str(conn[conn_id].id.plmn.mcc))
    print("Global E2 Node [" + str(conn_id) + "]: PLMN MNC = " + str(conn[conn_id].id.plmn.mnc))

##################################################
#### DRL functions
##################################################

# Function to generate action and send control message
def send_action(action):
    msg = ric.mac_ctrl_msg_t()
    msg.action = 42
    msg.offload = float(action)
    
    # Call the C++ function, which should now receive the correctly populated array
    ric.control_mac_sm(conn[i].id, msg)
    #time.sleep(0.01)

'''
# Reward function calculation (this is an example, modify as per your needs)
def calculate_reward(current_bler, current_energy, previous_bler, previous_energy):
    reward = - (np.log(1 + current_bler / (previous_bler + 1e-6)) + 
                np.log(1 + current_energy / (previous_energy + 1e-6)))
    return reward
'''
'''
def calculate_reward_no_thresholds(current_bler, current_energy, alpha=1.0, beta=1.0):
    """
    Reward function that penalizes BLER and energy directly, without thresholds.

    Args:
    - current_bler (float): Current BLER value (should be between 0 and 1).
    - current_energy (float): Current energy consumption.
    - alpha (float): Weight for the BLER penalty.
    - beta (float): Weight for the energy penalty.

    Returns:
    - reward (float): The calculated reward based on the current BLER and energy.
    """
    # Penalize current BLER and energy directly
    bler_penalty = np.log(1 + current_bler)
    energy_penalty = np.log(1 + current_energy)

    # Combine penalties with weights alpha and beta
    reward = -(alpha * bler_penalty + beta * energy_penalty)
    
    return reward
'''
'''
def calculate_reward_bler_threshold(current_bler, current_energy, bler_threshold = 0.05, alpha=1.0, beta=1.0):
    """
    Reward function with a threshold for BLER and direct penalty for energy.

    Args:
    - current_bler (float): Current BLER value (should be between 0 and 1).
    - current_energy (float): Current energy consumption.
    - bler_threshold (float): Threshold for acceptable BLER.
    - alpha (float): Weight for the BLER penalty.
    - beta (float): Weight for the energy penalty.

    Returns:
    - reward (float): The calculated reward based on the current BLER and energy.
    """
    # Penalize BLER if it exceeds the threshold
    bler_penalty = np.log(1 + current_bler / (bler_threshold + 1e-6))
    
    # Penalize energy directly
    energy_penalty = np.log(1 + current_energy)

    # Combine penalties with weights alpha and beta
    reward = -(alpha * bler_penalty + beta * energy_penalty)
    
    return reward
'''
'''
def calculate_reward(current_bler, current_energy, previous_bler, previous_energy):
    reward = - (np.log(1 + current_bler / (previous_bler + 1e-6)) + 
                np.log(1 + current_energy / (previous_energy + 1e-6)))
    return reward
'''

def calculate_reward(current_bler, current_energy, bler_threshold=0.05, penalty_factor=100):
    # Penalty if BLER exceeds the threshold
    if current_bler > bler_threshold:
        penalty = penalty_factor * (current_bler - bler_threshold)
    else:
        penalty = 0  # No penalty if BLER is below the threshold

    # BLER term: reward for lower BLER (negative because higher BLER is worse)
    bler_term = - current_bler

    # Energy term: penalize for high energy consumption (directly)
    energy_term = - current_energy

    # Final reward: a combination of BLER and energy terms, with penalty for high BLER
    reward = bler_term - penalty + energy_term

    return reward


#################################
#### DRL main method
#################################
file_name_reward = '/home/nakaolab/drl_reward.csv'
file_name_memory = '/home/nakaolab/drl_memory.csv'
def write_reward_to_csv(reward, time_now, file_name):
    with open(file_name, mode = 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([reward, time_now])

def write_memory_to_csv(current_state, action, reward, done, next_state, time_now, file_name):
    with open(file_name, mode = 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([action, reward, done, time_now])

def run_drl(stop_event):
    """
    Run the DDPG training for a specified number of epochs.
    
    :param stop_event: threading event to stop the training
    :param num_epochs: number of training epochs
    :param max_steps_per_epoch: max steps per epoch
    """
    state_normalizer = StateNormalizer()
    has_continuous_action_space = True
    max_ep_len = 25  # or any length you prefer
    update_timestep = max_ep_len
    action_std_decay_rate = 0.05
    min_action_std = 0.1
    action_std_decay_freq = 50
    log_freq = max_ep_len
    save_model_freq = 500
    
    # Logging and checkpointing setup
    log_f_name = '/home/nakaolab/ppo/PPO_log.csv'
    checkpoint_path = "/home/nakaolab/ppo/PPO_checkpoint.pth"

    # Initialize logging
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward,latency\n')

    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0

    start_time = time.time()

    ################### Start training loop ###################
    while not stop_event.is_set():
        # Initial state from the environment (e.g., BLER and Energy)
        current_bler = global_ue_aggr_data.get_bler_stats()
        current_energy = global_ue_aggr_data.get_energy_stats()

        current_state = state_normalizer.normalize_state(current_bler[0], current_bler[1], 
                                                         current_bler[2], current_bler[3],
                                                         current_energy[0], current_energy[1], 
                                                         current_energy[2], current_energy[3])

        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            # Select action using PPO policy
            action = ppo_agent.select_action(current_state)
            #print(f"Selected Action: {action}")

            # Send action to the environment (RAN) via control message
            send_action(action)
            time.sleep(0.01)

            # Wait for environment feedback (new state and reward)
            next_bler = global_ue_aggr_data.get_bler_stats()
            next_energy = global_ue_aggr_data.get_energy_stats()

            next_state = state_normalizer.normalize_state(next_bler[0], next_bler[1], 
                                                          next_bler[2], next_bler[3],
                                                          next_energy[0], next_energy[1], 
                                                          next_energy[2], next_energy[3])

            # Calculate reward (based on BLER and energy metrics)
            reward = calculate_reward(next_bler[0], next_energy[0])

            # Store the experience in PPO's buffer
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(False)  # Modify this logic if needed

            current_state = next_state
            current_ep_reward += reward
            time_step += 1

            # Update PPO if it's time
            update_latency = 0
            if time_step % update_timestep == 0:
                time_start = time.time_ns() / 1000000.0
                ppo_agent.update()
                time_end = time.time_ns() / 1000000.0
                update_latency = time_end - time_start

            # Decay action std for exploration (if using continuous actions)
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Logging
            if time_step % log_freq == 0:
                avg_reward = print_running_reward / print_running_episodes if print_running_episodes > 0 else print_running_reward
                log_f.write('{},{},{},{}\n'.format(i_episode, time_step, round(avg_reward, 4), update_latency))
                log_f.flush()

                print_running_reward = 0
                print_running_episodes = 0

            # Save model checkpoint
            #if time_step % save_model_freq == 0:
                #ppo_agent.save(checkpoint_path)
                #print(f"Saved model checkpoint at timestep {time_step}")

            # Stop if the environment indicates it's done
            if stop_event.is_set():
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        i_episode += 1

    log_f.close()

    end_time = time.time()
    print(f"Total training time: {end_time - start_time} seconds")



##############################
#### MAC IND&CTRL with DRL
##############################
mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)
    time.sleep(1)

try:
    # Create a stop event for the drl thread
    stop_event = threading.Event()

    # Start the drl thread using the global dictionary
    drl_thread = threading.Thread(target=run_drl, args=(stop_event,))
    drl_thread.daemon = True  # Ensures the thread exits when the main program exits
    drl_thread.start()

    # Simulate main program running for a long time or until Ctrl+C is pressed
    time.sleep(1000)

except KeyboardInterrupt:
    print("Stopping drl and cleaning up...")

    # Set the stop event to stop the drl thread
    stop_event.set()

    # Wait for the drl thread to finish
    drl_thread.join()

    for i in range(0, len(mac_hndlr)):
        ric.rm_report_mac_sm(mac_hndlr[i])

    # Avoid deadlock. ToDo revise architecture 
    while ric.try_stop == 0:
        time.sleep(1)

    print("Test finished")



