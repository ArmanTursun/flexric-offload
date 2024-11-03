import time
import numpy as np
import csv
import os
import sys
from datetime import datetime
import argparse

cur_dir = os.path.dirname(os.path.abspath(__file__))
# print("Current Directory:", cur_dir)
sdk_path = cur_dir + "/../xapp_sdk/"
sys.path.append(sdk_path)
import xapp_sdk as ric

# MACCallback class is defined and derived from C++ class mac_cb
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
            print(f"MAC Indication tstamp {t_now} diff {t_diff}")
            print('MAC rnti = ' + str(ind.ue_stats[0].rnti))


def create_conf(mcs, prb, add):
    conf = ric.mac_conf_t()
    conf.isset_pusch_mcs = add
    conf.pusch_mcs = mcs
    conf.rnti = prb
    return conf

node_idx = 0
mac_hndlr = []

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description="MAC Control")
    parser.add_argument('-m', '--mcs', type=int, default=28, help="MCS value (default: 28)")
    parser.add_argument('-r', '--prb', type=int, default=106, help="PRB value (default: 106)")
    parser.add_argument('-a', '--add', type=int, default=True, help="Add Configurations? (default: True)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the action value
    mcs = args.mcs
    prb = args.prb
    add = True if args.add else False

    # Initialize RIC and connections
    ric.init()
    conn = ric.conn_e2_nodes()
    assert len(conn) > 0, "No connected E2 nodes found."
    
    for i in range(0, len(conn)):
        print(f"Global E2 Node [{i}]: PLMN MCC = {conn[i].id.plmn.mcc}")
        print(f"Global E2 Node [{i}]: PLMN MNC = {conn[i].id.plmn.mnc}")

    try:
        for i in range(0, len(conn)):
            mac_cb = MACCallback()
            hndlr = ric.report_mac_sm(conn[i].id, ric.Interval_ms_10, mac_cb)
            mac_hndlr.append(hndlr)
            time.sleep(1)
        
        msg = ric.mac_ctrl_msg_t()
        msg.ran_conf_len = 2
        confs = ric.mac_conf_array(2)
        for i in range(0, msg.ran_conf_len):
            confs[i] = create_conf(mcs, prb, add)
        
        msg.ran_conf = confs
        print(f"Sending mcs value: {mcs}, prb value: {prb}, add value: {add}")
        ric.control_mac_sm(conn[node_idx].id, msg)
        
        for i in range(0, len(mac_hndlr)):
            ric.rm_report_mac_sm(mac_hndlr[i])
        
        while ric.try_stop == 0:
            time.sleep(1)
        print("Test finished")

    except KeyboardInterrupt:
        print("Stopping DRL and cleaning up...")

        # Avoid deadlock. ToDo revise architecture 
        while ric.try_stop == 0:
            time.sleep(1)

        print("Test finished")

