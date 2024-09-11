import xapp_sdk as ric
import time
import threading
from aggr_data import AggrData
from ddpg import DDPG
import numpy as np

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
global_ue_aggr_data = AggrData(1000)

# Global lock to ensure thread-safe access to the global dictionary
global_lock = threading.Lock()

# DDPG Parameters
state_dim = 8  # BLER and energy
action_dim = 2  # Weight for BLER and energy
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.99
tau = 0.001
buffer_size = 100000
batch_size = 64
ddpg_agent = DDPG(state_size = state_dim, action_size = action_dim,actor_learning_rate=actor_lr, critic_learning_rate=critic_lr, 
    batch_size=batch_size, discount=gamma, memory_size=buffer_size, tau=tau)


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
        return np.array([normalized_bler_mean, normalized_bler_max, normalized_bler_min, normalized_bler_skewness,
                         normalized_energy_mean, normalized_energy_max, normalized_energy_min, normalized_energy_skewness])


##################################################
#### MACCallback Class
##################################################

class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)

    def handle(self, ind):
        # save bler and energy values of each tbs of each ue into aggr_data
        if len(ind.ue_stats) > 0:
            for i in range(ind.len_ue_stats):
                ue_stats = ind.ue_stats[i]

                # Calculate average energy from the TBS data
                #total_energy = 0
                #for tbs_stat in ue_stats.tbs[:ue_stats.num_tbs]:
                    #total_energy += tbs_stat.latency
                #avg_energy = total_energy / ue_stats.num_tbs if ue_stats.num_tbs > 0 else 0

                # Add the new BLER and energy data to the AggrData object
                with global_lock:
                    global_ue_aggr_data.add_bler(ue_stats.ul_bler, ind.tstamp)
                    global_ue_aggr_data.add_energy(ue_stats.dl_bler, ind.tstamp)


####################
####  init RIC
####################

ric.init()
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
    #msg.tms = time.time_ns() / 1000.0
    #msg.num_ues = 1
    #ues = ric.mac_ue_ctrl_array(msg.num_ues)

    # Assign values to each element of the array
    #for i in range(msg.num_ues):
        #ues[i] = ric.mac_ue_ctrl_t()
        #ues[i].rnti = i
        #ues[i].offload = float(action[1])  # Convert action[i] to float explicitly
        #print(f"Assigned offload for UE {i}: {ues[i].offload}")  # Debugging info

    # Explicitly assign the array back to msg.ues
    #msg.ues = ues

    #if np.isnan(action).any() or np.isinf(action).any():
        #action[1] = 0.0

    #ues = ric.mac_ue_ctrl_t()  # Create a single UE
    #ues.rnti = 1
    #ues.offload = float(action[1])  # Assign fixed value
    #msg.ues = ues  # Pass the single UE to the control message
    #print(f"Assigned offload: {ues.offload}")

    
    # Call the C++ function, which should now receive the correctly populated array
    ric.control_mac_sm(conn[conn_id].id, msg)

# Reward function calculation (this is an example, modify as per your needs)
def calculate_reward(current_bler, current_energy, previous_bler, previous_energy):
    reward = - (np.log(1 + current_bler / (previous_bler + 1e-6)) + 
                np.log(1 + current_energy / (previous_energy + 1e-6)))
    return reward

#################################
#### DRL main method
#################################
'''
def run_drl(stop_event):

    state_normalizer = StateNormalizer()

    current_bler = 0
    current_energy = 0
    next_bler = 0
    next_energy = 0

    with global_lock:
        # Retrieve the current aggregated state from the environment (RAN)
        current_bler = global_ue_aggr_data.get_bler_stats()
        current_energy = global_ue_aggr_data.get_energy_stats()
        
    #while not stop_event.is_set():
    for step in range(200):

        #current_state = np.array([current_bler, current_energy])
        current_state = state_normalizer.normalize_state(current_bler[0], current_bler[1], 
                                                         current_bler[2], current_bler[3], 
                                                         current_energy[0], current_energy[1], 
                                                         current_energy[2], current_energy[3])

        # Generate action using DDPG agent
        action = ddpg_agent.get_action(current_state)
        print("Actions:", action)
        
        # Send the action to the RAN via control message
        send_action(action)
        with global_lock:
            # Get the next state after applying the action
            next_bler = global_ue_aggr_data.get_bler_stats()
            next_energy = global_ue_aggr_data.get_energy_stats()
            #next_state = np.array([next_bler, next_energy])

        next_state = state_normalizer.normalize_state(next_bler[0], next_bler[1], 
                                                         next_bler[2], next_bler[3], 
                                                         next_energy[0], next_energy[1], 
                                                         next_energy[2], next_energy[3])
        
        # Calculate reward based on the transition (current_state -> next_state)
        reward = calculate_reward(next_bler[0], next_energy[0], current_bler[0], current_energy[0])

        # Remember the transition in the DDPG agent's memory
        ddpg_agent.remember(current_state, action, reward, False, next_state)

        # Train the DDPG agent with a batch of experiences
        ddpg_agent.train()

        # Update the previous state for the next iteration
        current_bler = next_bler
        current_energy = next_energy

        # Update the actor and critic loss monitoring
        actor_loss = ddpg_agent.actor_loss
        critic_loss = ddpg_agent.critic_loss
        print(f"Step {step + 1} - Actor Loss: {actor_loss:.5f}, Critic Loss: {critic_loss:.5f}")
'''

def run_drl(stop_event, num_epochs=10, max_steps_per_epoch=200, warmup_steps=0):
    """
    Run the DDPG training for a specified number of epochs.
    
    :param stop_event: threading event to stop the training
    :param num_epochs: number of training epochs
    :param max_steps_per_epoch: max steps per epoch
    """
    state_normalizer = StateNormalizer()
    current_bler = None
    current_energy = None
    current_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        while True:
            with global_lock:
                # Retrieve the current aggregated state from the environment (RAN)
                current_bler = global_ue_aggr_data.get_bler_stats()
                current_energy = global_ue_aggr_data.get_energy_stats()

            # Normalize the initial state
            current_state = state_normalizer.normalize_state(current_bler[0], current_bler[1],
                                        current_bler[2], current_bler[3],
                                        current_energy[0], current_energy[1],
                                        current_energy[2], current_energy[3])
            if np.isnan(current_bler).any() or np.isinf(current_bler).any() or np.isnan(current_energy).any() or np.isinf(current_energy).any():
                continue
            else:
                break
            
        # Reset noise and initial states at the beginning of each epoch
        ddpg_agent.reset_noise()
        total_reward = 0

        # Run the steps for each epoch
        for step in range(max_steps_per_epoch):
            #if step < warmup_steps:
                #print(f"Warm-up step {step + 1}/{warmup_steps}, skipping training...")
                
                #continue  # Skip training until valid states are generated

            # Generate action using DDPG agent
            action = ddpg_agent.get_action(current_state)
            #print("Actions:", action)

            # Send the action to the RAN via control message
            send_action(action)

            with global_lock:
                # Get the next state after applying the action
                next_bler = global_ue_aggr_data.get_bler_stats()
                next_energy = global_ue_aggr_data.get_energy_stats()

            # Normalize the next state
            next_state = state_normalizer.normalize_state(next_bler[0], next_bler[1],
                                                          next_bler[2], next_bler[3],
                                                          next_energy[0], next_energy[1],
                                                          next_energy[2], next_energy[3])

            # Calculate reward based on the transition (current_state -> next_state)
            reward = calculate_reward(next_bler[0], next_energy[0], current_bler[0], current_energy[0])

            # Check if episode is done (you can define your own condition here)
            # E.g., if BLER or energy hits some threshold, we can terminate the episode.
            done = (step == max_steps_per_epoch - 1)# or (reward > some_reward_threshold)

            # Remember the transition in the DDPG agent's memory
            # print(current_state, action, reward, done, next_state)
            ddpg_agent.remember(current_state, action, reward, done, next_state)

            # Train the DDPG agent with a batch of experiences
            ddpg_agent.train()

            # Update the previous state for the next iteration
            current_state = next_state
            current_bler = next_bler
            current_energy = next_energy

            # Update the total reward
            total_reward += reward

            # If the episode is done, break the step loop and start a new epoch
            if done:
                #print(f"Episode done at step {step + 1} with total reward: {total_reward:.5f}")
                break

            # Update the actor and critic loss monitoring
            actor_loss = ddpg_agent.actor_loss
            critic_loss = ddpg_agent.critic_loss
            # print(f"Step {step + 1} - Actor Loss: {actor_loss:.5f}, Critic Loss: {critic_loss:.5f}")

        print(f"Epoch {epoch + 1} finished with total reward: {total_reward:.5f}\n")
        if stop_event.is_set():
            break



##############################
#### MAC IND&CTRL with DRL
##############################

mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)

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



