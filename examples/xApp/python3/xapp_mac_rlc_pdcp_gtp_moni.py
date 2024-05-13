import xapp_sdk as ric
import time
import os
import pdb

####################
#### MAC INDICATION CALLBACK
####################

#  MACCallback class is defined and derived from C++ class mac_cb
class MACCallback(ric.mac_cb):
    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        ric.mac_cb.__init__(self)
    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_mac_ind_msg_t
        if len(ind.ue_stats) > 0:
            t_now = time.time_ns() / 1000.0
            t_mac = ind.tstamp / 1.0
            t_diff = t_now - t_mac
            print('MAC Indication tstamp = ' + str(t_mac) + ' latency = ' + str(t_diff) + ' μs')
            #print('MAC rnti = ' + str(ind.ue_stats[0].rnti))


ldpc_offload = {
    "offload" : 1
}

def fill_slice_ctrl_msg(ctrl_msg):
    msg = ric.mac_ctrl_msg_t()
    msg.action = 42
    msg.offload = ctrl_msg["offload"]
    return msg

####################
####  GENERAL 
####################

ric.init()

conn = ric.conn_e2_nodes()
assert(len(conn) > 0)
for i in range(0, len(conn)):
    print("Global E2 Node [" + str(i) + "]: PLMN MCC = " + str(conn[i].id.plmn.mcc))
    print("Global E2 Node [" + str(i) + "]: PLMN MNC = " + str(conn[i].id.plmn.mnc))

####################
#### MAC INDICATION
####################

mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)
    msg = fill_slice_ctrl_msg(ldpc_offload)
    ric.control_mac_sm(conn[i].id, msg)
    time.sleep(1)

time.sleep(1)

### End

for i in range(0, len(mac_hndlr)):
    ric.rm_report_mac_sm(mac_hndlr[i])

# Avoid deadlock. ToDo revise architecture 
while ric.try_stop == 0:
    time.sleep(1)

print("Test finished")
