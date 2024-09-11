import xapp_sdk as ric
import time
import threading
from aggr_data import AggrData

#############################
#### Global Variables
#############################

# Global dictionary to store bler and energy data
# It keeps an window of 1000 of most recent data
global_ue_aggr_data = AggrData(1000)

# Global lock to ensure thread-safe access to the global dictionary
global_lock = threading.Lock()

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

# fill mac_ctrl_msg with data
def fill_mac_ctrl_msg():
    msg = ric.mac_ctrl_msg_t()
    msg.action = 42
    msg.tms = time.time_ns() / 1000.0
    msg.num_ues = 3
    ues = ric.mac_ue_ctrl_array(msg.num_ues) ## define array and it's lenth
    for i in range(msg.num_ues):
        ues[i].rnti = i
        ues[i].offload = 1
    msg.ues = ues
    return msg
#ctrl = fill_mac_ctrl_msg(self.ldpc_offload)
#ric.control_mac_sm(conn[i].id, ctrl)


#################################
#### DRL main method
#################################

def run_drl(stop_event):   
    while not stop_event.is_set():
        time.sleep(1)  
        with global_lock:
            print("BLER Stats:", global_ue_aggr_data.aggr_bler['mean'])
            print("Energy Stats:", global_ue_aggr_data.aggr_enrg['mean'])


####################
####  init RIC
####################

ric.init()

conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))

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



