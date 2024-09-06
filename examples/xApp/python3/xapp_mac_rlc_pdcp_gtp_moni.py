import xapp_sdk as ric
import time
import os
import pdb
from collections import deque
import threading

#############################
#### Global Buffer and Lock
#############################

# Define the global buffer and a lock
global_tbs_buffer = deque(maxlen=1000)
buffer_lock = threading.Lock()
monitoring = True  # Flag to control the monitoring thread

####################
#### MAC INDICATION CALLBACK
####################
class AggrData():
    def __init__(self, maxlenth):
        self.maxqlenth = maxlenth
        self.aggr_bler = {
            'bler': deque(),  # save (bler, TS) tuple to queue
            'mean': 0,        # mean of bler
            'max': 0,         # max of bler
            'min': float('inf'), # min of bler (initialize with infinity)
            'num': 0,         # num of bler
            'sum': 0          # sum of bler
        }
        self.aggr_enrg = {
            'enrg': deque(),  # save (energy, TS) tuple to queue
            'mean': 0,        # mean of energy
            'max': 0,         # max of energy
            'min': float('inf'), # min of energy (initialize with infinity)
            'num': 0,         # num of energy
            'sum': 0          # sum of energy
        }

    def add_bler(self, bler_value, timestamp):
        """Add a new BLER value with its timestamp and update related metrics"""
        if len(self.aggr_bler['bler']) >= self.maxqlenth:
            # If queue is full, remove the oldest entry
            removed_bler, _ = self.aggr_bler['bler'].popleft()
            self.aggr_bler['sum'] -= removed_bler
            self.aggr_bler['num'] -= 1

        # Add new bler value
        self.aggr_bler['bler'].append((bler_value, timestamp))
        self.aggr_bler['sum'] += bler_value
        self.aggr_bler['num'] += 1

        # Update mean, max, and min
        self.aggr_bler['mean'] = self.aggr_bler['sum'] / self.aggr_bler['num']
        self.aggr_bler['max'] = max(self.aggr_bler['max'], bler_value)
        self.aggr_bler['min'] = min(self.aggr_bler['min'], bler_value)

    def add_energy(self, energy_value, timestamp):
        """Add a new energy value with its timestamp and update related metrics"""
        if len(self.aggr_enrg['enrg']) >= self.maxqlenth:
            # If queue is full, remove the oldest entry
            removed_enrg, _ = self.aggr_enrg['enrg'].popleft()
            self.aggr_enrg['sum'] -= removed_enrg
            self.aggr_enrg['num'] -= 1

        # Add new energy value
        self.aggr_enrg['enrg'].append((energy_value, timestamp))
        self.aggr_enrg['sum'] += energy_value
        self.aggr_enrg['num'] += 1

        # Update mean, max, and min
        self.aggr_enrg['mean'] = self.aggr_enrg['sum'] / self.aggr_enrg['num']
        self.aggr_enrg['max'] = max(self.aggr_enrg['max'], energy_value)
        self.aggr_enrg['min'] = min(self.aggr_enrg['min'], energy_value)

    def get_bler_stats(self):
        """Get current BLER statistics"""
        return {
            'mean': self.aggr_bler['mean'],
            'max': self.aggr_bler['max'],
            'min': self.aggr_bler['min'],
            'num': self.aggr_bler['num'],
            'sum': self.aggr_bler['sum']
        }

    def get_energy_stats(self):
        """Get current energy statistics"""
        return {
            'mean': self.aggr_enrg['mean'],
            'max': self.aggr_enrg['max'],
            'min': self.aggr_enrg['min'],
            'num': self.aggr_enrg['num'],
            'sum': self.aggr_enrg['sum']
        }
    


#  MACCallback class is defined and derived from C++ class mac_cb
class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
        self.cnt = 0
        self.tbs = 0
        self.ldpc_offload = {
            'offload' : 0
        }
        self.t_10 = time.time_ns() / 1000.0
    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_mac_ind_msg_t

        if len(ind.ue_stats) > 0:
            t_now = time.time_ns() / 1000.0
            self.cnt += 1
            if (True or t_now - self.t_10 >= 1000):
                ctrl_send = True
                self.t_10 = t_now
            else:
                ctrl_send = True

            for ue_id in range(len(ind.ue_stats)):
                ue = ind.ue_stats[ue_id]
                # Modify the global buffer inside a lock
                with buffer_lock:
                    global_tbs_buffer.append(ue.rnti)  # Save tbs in the global buffer
                #print('[xApp Monitor 2]: timestamp = ' + str(ind.tstamp) + ' rnti = ' + str(ue_context.rnti))               
                if (ue_id == 0):
                    self.ldpc_offload["offload"] += 1
                else:
                    self.ldpc_offload["offload"] += 1               
            #if (ctrl_send):
                ctrl = fill_mac_ctrl_msg(self.ldpc_offload)
                #print('[xApp Control]: num_ues = ' + str(ctrl.num_ues) + ' timestamp = ' + str(ctrl.tms))
                #ric.control_mac_sm(conn[i].id, ctrl)

def fill_mac_ctrl_msg(ctrl_msg):
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

####################
#### BUFFER MONITORING THREAD
####################

def monitor_buffer():
    cnt = 0
    while monitoring:
        with buffer_lock:  # Acquire the lock before accessing the buffer
            print(f"Global buffer size: {len(global_tbs_buffer)}")
        cnt += 1
        if (cnt == 5):
            ctrl = fill_mac_ctrl_msg(1)
            #print('[xApp Control]: num_ues = ' + str(ctrl.num_ues) + ' timestamp = ' + str(ctrl.tms))
            ric.control_mac_sm(conn[i].id, ctrl)
        time.sleep(1)  # Sleep for a while before checking the buffer again

####################
####  GENERAL 
####################

ric.init()

conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))

##############################
#### MAC INDICATION & CONTROL
##############################

mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)

# Create a thread to monitor the buffer
monitor_thread = threading.Thread(target=monitor_buffer)
monitor_thread.start()

time.sleep(10)

### End

for i in range(0, len(mac_hndlr)):
    ric.rm_report_mac_sm(mac_hndlr[i])

# Stop the monitoring thread
monitoring = False
monitor_thread.join()  # Wait for the monitoring thread to finish

# Avoid deadlock. ToDo revise architecture 
while ric.try_stop == 0:
    time.sleep(1)

print("Test finished")
