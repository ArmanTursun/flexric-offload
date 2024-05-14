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
        self.cnt = 0    
    # Override C++ method: virtual void handle(swig_mac_ind_msg_t a) = 0;
    def handle(self, ind):
        # Print swig_mac_ind_msg_t
        if len(ind.ue_stats) > 0:
            def fill_mac_ctrl_msg(ctrl_msg):
                msg = ric.mac_ctrl_msg_t()
                msg.action = 42
                msg.offload = ctrl_msg["offload"]
                return msg
            t_now = time.time_ns() / 1000.0
            t_mac = ind.tstamp / 1.0
            t_diff = t_now - t_mac
            self.cnt += 1

            ldpc_offload = {
                "offload" : 0
            }

            ue_context = ind.ue_stats[0]
            tbs = ue_context.ul_curr_tbs
            if (tbs > 5000):
                ldpc_offload["offload"] = 1
                print('MAC Indication tstamp = ' + str(t_mac) + ' latency = ' + str(t_diff) + ' μs')
                #print('MAC rnti = ' + str(ind.ue_stats[0].rnti))            
                print('TBS: ', ue_context.ul_curr_tbs)
                ctrl = fill_mac_ctrl_msg(ldpc_offload)
                ric.control_mac_sm(conn[i].id, ctrl)
            else:
                ldpc_offload["offload"] = 0

def fill_mac_ctrl_msg(ctrl_msg):
    #wr = ric.mac_ctrl_req_data_t()
    #wr.hdr.dummy = 1
    #wr.msg.action = 42
    #wr.msg.offload = ctrl_msg["offload"]
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

##############################
#### MAC INDICATION & CONTROL
##############################

mac_hndlr = []
for i in range(0, len(conn)):
    mac_cb = MACCallback()
    hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_1, mac_cb)
    mac_hndlr.append(hndlr)
    #ctrl = fill_mac_ctrl_msg(ldpc_offload)
    #print(ctrl.hdr.dummy, ctrl.msg.action, ctrl.msg.offload)
    #ric.control_mac_sm(conn[i].id, ctrl)
    #time.sleep(1)

time.sleep(2)

#offload_ind = 22

#ldpc_offload = {
#    "offload" : 0
#}

#for j in range(5):
#    offload_ind = offload_ind + j
#    ldpc_offload['offload'] = offload_ind
#    ctrl = fill_mac_ctrl_msg(ldpc_offload)
#    ric.control_mac_sm(conn[i].id, ctrl)

### End

for i in range(0, len(mac_hndlr)):
    ric.rm_report_mac_sm(mac_hndlr[i])

# Avoid deadlock. ToDo revise architecture 
while ric.try_stop == 0:
    time.sleep(1)

print("Test finished")
