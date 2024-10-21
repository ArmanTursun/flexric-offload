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

node_idx = 0

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description="MAC Control")
    parser.add_argument('-m', '--mcs', type=int, default=28, help="MCS value (default: 28)")
    parser.add_argument('-r', '--prb', type=int, default=106, help="PRB value (default: 106)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the action value
    mcs = args.mcs
    prb = args.prb

    # Initialize RIC and connections
    ric.init()
    conn = ric.conn_e2_nodes()
    assert len(conn) > 0, "No connected E2 nodes found."
    
    for i in range(0, len(conn)):
        print(f"Global E2 Node [{i}]: PLMN MCC = {conn[i].id.plmn.mcc}")
        print(f"Global E2 Node [{i}]: PLMN MNC = {conn[i].id.plmn.mnc}")

    try:
        msg = ric.mac_ctrl_msg_t()
        msg.action = 42
        msg.mcs = mcs
        msg.prb = prb
        print(f"Sending mcs value: {mcs}, prb value: {prb}")
        ric.control_mac_sm(conn[node_idx].id, msg)

    except KeyboardInterrupt:
        print("Stopping DRL and cleaning up...")

        # Avoid deadlock. ToDo revise architecture 
        while ric.try_stop == 0:
            time.sleep(1)

        print("Test finished")

