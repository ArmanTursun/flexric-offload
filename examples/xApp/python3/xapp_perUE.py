import xapp_sdk as ric
import time
import os
import pdb
from collections import deque
import threading


#############################
#### Aggregated Data
#############################

class AggrData():
    def __init__(self, maxlenth):
        self.maxqlenth = maxlenth
        self.aggr_bler = {
            'bler': deque(),       # save (bler, TS) tuple to queue
            'mean': 0,             # mean of bler
            'max': 0,              # max of bler in window
            'min': float('inf'),    # min of bler in window
            'num': 0,              # num of bler
            'sum': 0               # sum of bler
        }
        self.aggr_enrg = {
            'enrg': deque(),       # save (energy, TS) tuple to queue
            'mean': 0,             # mean of energy
            'max': 0,              # max of energy in window
            'min': float('inf'),    # min of energy in window
            'num': 0,              # num of energy
            'sum': 0               # sum of energy
        }
        
        # Deques to track min/max for bler and energy. Keeps add/pop in contant time
        self.bler_min_deque = deque()
        self.bler_max_deque = deque()
        self.enrg_min_deque = deque()
        self.enrg_max_deque = deque()

    def add_bler(self, bler_value, timestamp):
        """Add a new BLER value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_bler['bler']) >= self.maxqlenth:
            removed_bler, _ = self.aggr_bler['bler'].popleft()
            self.aggr_bler['sum'] -= removed_bler
            self.aggr_bler['num'] -= 1
            if self.bler_min_deque and self.bler_min_deque[0] == 0:
                self.bler_min_deque.popleft()
            if self.bler_max_deque and self.bler_max_deque[0] == 0:
                self.bler_max_deque.popleft()

            # Decrement indices in min/max deques
            self.bler_min_deque = deque([i-1 for i in self.bler_min_deque])
            self.bler_max_deque = deque([i-1 for i in self.bler_max_deque])

        # Add new BLER value
        self.aggr_bler['bler'].append((bler_value, timestamp))
        self.aggr_bler['sum'] += bler_value
        self.aggr_bler['num'] += 1

        # Update the min/max deques for bler
        while self.bler_min_deque and bler_value < self.aggr_bler['bler'][self.bler_min_deque[-1]][0]:
            self.bler_min_deque.pop()
        self.bler_min_deque.append(len(self.aggr_bler['bler']) - 1)

        while self.bler_max_deque and bler_value > self.aggr_bler['bler'][self.bler_max_deque[-1]][0]:
            self.bler_max_deque.pop()
        self.bler_max_deque.append(len(self.aggr_bler['bler']) - 1)

        # Update the mean
        self.aggr_bler['mean'] = self.aggr_bler['sum'] / self.aggr_bler['num']
        self.aggr_bler['min'] = self.aggr_bler['bler'][self.bler_min_deque[0]][0]
        self.aggr_bler['max'] = self.aggr_bler['bler'][self.bler_max_deque[0]][0]

    def add_energy(self, energy_value, timestamp):
        """Add a new energy value with its timestamp and update related metrics"""
        # Remove the oldest element if the queue is full
        if len(self.aggr_enrg['enrg']) >= self.maxqlenth:
            removed_enrg, _ = self.aggr_enrg['enrg'].popleft()
            self.aggr_enrg['sum'] -= removed_enrg
            self.aggr_enrg['num'] -= 1
            if self.enrg_min_deque and self.enrg_min_deque[0] == 0:
                self.enrg_min_deque.popleft()
            if self.enrg_max_deque and self.enrg_max_deque[0] == 0:
                self.enrg_max_deque.popleft()

            # Decrement indices in min/max deques
            self.enrg_min_deque = deque([i-1 for i in self.enrg_min_deque])
            self.enrg_max_deque = deque([i-1 for i in self.enrg_max_deque])

        # Add new energy value
        self.aggr_enrg['enrg'].append((energy_value, timestamp))
        self.aggr_enrg['sum'] += energy_value
        self.aggr_enrg['num'] += 1

        # Update the min/max deques for energy
        while self.enrg_min_deque and energy_value < self.aggr_enrg['enrg'][self.enrg_min_deque[-1]][0]:
            self.enrg_min_deque.pop()
        self.enrg_min_deque.append(len(self.aggr_enrg['enrg']) - 1)

        while self.enrg_max_deque and energy_value > self.aggr_enrg['enrg'][self.enrg_max_deque[-1]][0]:
            self.enrg_max_deque.pop()
        self.enrg_max_deque.append(len(self.aggr_enrg['enrg']) - 1)

        # Update the mean
        self.aggr_enrg['mean'] = self.aggr_enrg['sum'] / self.aggr_enrg['num']
        self.aggr_enrg['min'] = self.aggr_enrg['enrg'][self.enrg_min_deque[0]][0]
        self.aggr_enrg['max'] = self.aggr_enrg['enrg'][self.enrg_max_deque[0]][0]

    def get_bler_stats(self, timestamp):
        """Get current BLER statistics only if the first element's timestamp is later than the given timestamp"""
        if self.aggr_bler['bler'] and self.aggr_bler['bler'][0][1] > timestamp:
            return {
                'mean': self.aggr_bler['mean'],
                'max': self.aggr_bler['max'],
                'min': self.aggr_bler['min'],
                'num': self.aggr_bler['num'],
                'sum': self.aggr_bler['sum']
            }
        return None

    def get_energy_stats(self, timestamp):
        """Get current energy statistics only if the first element's timestamp is later than the given timestamp"""
        if self.aggr_enrg['enrg'] and self.aggr_enrg['enrg'][0][1] > timestamp:
            return {
                'mean': self.aggr_enrg['mean'],
                'max': self.aggr_enrg['max'],
                'min': self.aggr_enrg['min'],
                'num': self.aggr_enrg['num'],
                'sum': self.aggr_enrg['sum']
            }
        return None


#############################
#### Global Variables
#############################

# Global dictionary to store AggrData objects for each UE
global_ue_aggr_data = AggrData(1000)

# Global lock to ensure thread-safe access to the global dictionary
global_lock = threading.Lock()


##################################################
#### MACCallback Class (with Monitoring Thread)
##################################################

class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self, maxlen=1000):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
        self.ue_aggr_data = {}  # Dictionary to store AggrData objects for each UE
        self.maxlen = maxlen
        self.lock = threading.Lock()  # Lock to protect shared data

    def handle(self, ind):
        # Process each UE in the indication message
        for i in range(ind.len_ue_stats):
            ue_stats = ind.ue_stats[i]
            rnti = ue_stats.rnti

            # If this UE does not have an AggrData object yet, create one
            with self.lock:
                if rnti not in self.ue_aggr_data:
                    self.ue_aggr_data[rnti] = AggrData(self.maxlen)

            # Calculate average energy from the TBS data
            total_energy = 0
            for tbs_stat in ue_stats.tbs[:ue_stats.num_tbs]:
                total_energy += tbs_stat.tbs
            avg_energy = total_energy / ue_stats.num_tbs if ue_stats.num_tbs > 0 else 0

            # Add the new BLER and energy data to the AggrData object
            with self.lock:
                self.ue_aggr_data[rnti].add_bler(ue_stats.context.ul_bler, ind.tstamp)
                self.ue_aggr_data[rnti].add_energy(avg_energy, ind.tstamp)


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
#ctrl = fill_mac_ctrl_msg(self.ldpc_offload)
#ric.control_mac_sm(conn[i].id, ctrl)


####################
#### BUFFER MONITORING THREAD
####################

def monitor_ue_data(mac_cb, stop_event):
    """Monitor and print aggregated data for UEs in a separate thread until stop_event is set."""
    while not stop_event.is_set():
        time.sleep(1)  # Monitor every second
        with mac_cb.lock:
            for rnti, aggr_data in mac_cb.ue_aggr_data.items():
                print(f"UE {rnti} BLER Stats")
                print(f"UE {rnti} Energy Stats")


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

mac_cb = MACCallback()

# Create a stop event for the monitoring thread
stop_event = threading.Event()

# Start the monitoring thread
monitor_thread = threading.Thread(target=monitor_ue_data, args=(mac_cb, stop_event))
monitor_thread.daemon = True  # Ensures the thread exits when the main program exits
monitor_thread.start()

mac_hndlr = []
for i in range(0, len(conn)):
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)

try:
    time.sleep(1000)

except KeyboardInterrupt:
    print("Stopping monitoring and cleaning up...")

    # Set the stop event to stop the monitoring thread
    stop_event.set()

    # Wait for the monitoring thread to finish
    monitor_thread.join()

    for i in range(0, len(mac_hndlr)):
        ric.rm_report_mac_sm(mac_hndlr[i])

    # Avoid deadlock. ToDo revise architecture 
    while ric.try_stop == 0:
        time.sleep(1)

    print("Test finished")



