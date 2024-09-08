import xapp_sdk as ric
import time
import threading
from aggr_data import AggrData
from ddpg import DDPG
from ou_noise import OUNoise
import numpy as np

#################################
###  TODO
###  1. stats are {} -> np
###  2. ric ctril handle stats
###  3. reward function
###  4. ddpg for states
###  5. check actor and critic
###  6. make sure drl can run
################################

#############################
#### Global Variables
#############################

# Global dictionary to store bler and energy data
# It keeps an window of 1000 of most recent data
global_ue_aggr_data = AggrData(1000)

# Global lock to ensure thread-safe access to the global dictionary
global_lock = threading.Lock()

# DDPG Parameters
state_dim = 2  # BLER and energy
action_dim = 2  # Weight for BLER and energy
actor_lr = 1e-4
critic_lr = 1e-3
gamma = 0.99
tau = 0.001
buffer_size = 100000
batch_size = 64
ddpg_agent = DDPG(state_size = state_dim, action_size = action_dim,actor_learning_rate=actor_lr, critic_learning_rate=critic_lr, 
    batch_size=batch_size, discount=gamma, memory_size=buffer_size, tau=tau)


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
                total_energy = 0
                for tbs_stat in ue_stats.tbs[:ue_stats.num_tbs]:
                    total_energy += tbs_stat.tbs
                avg_energy = total_energy / ue_stats.num_tbs if ue_stats.num_tbs > 0 else 0

                # Add the new BLER and energy data to the AggrData object
                with global_lock:
                    global_ue_aggr_data.add_bler(ue_stats.context.ul_bler, ind.tstamp)
                    global_ue_aggr_data.add_energy(avg_energy, ind.tstamp)


##################################################
#### DRL functions
##################################################

# Function to generate action and send control message
def send_action(action):
    msg = ric.mac_ctrl_msg_t()
    msg.action = 42
    msg.tms = time.time_ns() / 1000.0
    msg.num_ues = 3
    ues = ric.mac_ue_ctrl_array(msg.num_ues)
    
    # Example of how action could map to the control message (modify as needed)
    for i in range(msg.num_ues):
        ues[i].rnti = i
        ues[i].offload = action[i % 2]  # Assign actions to UEs based on the generated weights
    msg.ues = ues
    
    # Send control message to RAN
    ric.control_mac_sm(conn[0].id, msg)

# Reward function calculation (this is an example, modify as per your needs)
def calculate_reward(current_bler, current_energy, previous_bler, previous_energy):
    reward = - (np.log(1 + current_bler / (previous_bler + 1e-6)) + 
                np.log(1 + current_energy / (previous_energy + 1e-6)))
    return reward

####################
####  init RIC
####################

ric.init()

conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))

#################################
#### DRL main method
#################################

def run_drl(stop_event):
    current_bler = 0
    current_energy = 0
    next_bler = 0
    next_energy = 0

    with global_lock:
        # Retrieve the current aggregated state from the environment (RAN)
        current_bler = global_ue_aggr_data.aggr_bler['mean']
        current_energy = global_ue_aggr_data.aggr_enrg['mean']
        
    #while not stop_event.is_set():
    for step in range(200):

        current_state = np.array([current_bler, current_energy])

        # Generate action using DDPG agent
        action = ddpg_agent.get_action(current_state)
        
        # Send the action to the RAN via control message
        send_action(action)
        next_state = None

        with global_lock:
            # Get the next state after applying the action
            next_bler = global_ue_aggr_data.aggr_bler['mean']
            next_energy = global_ue_aggr_data.aggr_enrg['mean']
            next_state = np.array([next_bler, next_energy])

        # Calculate reward based on the transition (current_state -> next_state)
        reward = calculate_reward(next_bler, next_energy, current_bler, current_energy)

        # Remember the transition in the DDPG agent's memory
        ddpg_agent.remember(current_state, action, reward, False, next_state)

        # Train the DDPG agent with a batch of experiences
        ddpg_agent.train(batch_size)

        # Update the previous state for the next iteration
        current_bler = next_bler
        current_energy = next_energy


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



