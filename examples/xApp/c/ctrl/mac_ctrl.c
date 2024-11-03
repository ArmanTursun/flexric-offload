/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "../../../../src/xApp/e42_xapp_api.h"
#include "../../../../src/util/alg_ds/alg/defer.h"
#include "../../../../src/util/time_now_us.h"
#include "sm/mac_sm/mac_sm_id.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>

static sm_ag_if_wr_t fill_dummy_mac_sm_ctrl_req (void)
{
  sm_ag_if_wr_t wr = {0};
  wr.type = CONTROL_SM_AG_IF_WR;
  wr.ctrl.type = MAC_CTRL_REQ_V0;
  wr.ctrl.mac_ctrl.hdr.dummy = 0;
  wr.ctrl.mac_ctrl.msg.ran_conf_len = 2;
  // call 
  wr.ctrl.mac_ctrl.msg.ran_conf = calloc(wr.ctrl.mac_ctrl.msg.ran_conf_len, sizeof(wr.ctrl.mac_ctrl.msg.ran_conf));
  assert(wr.ctrl.mac_ctrl.msg.ran_conf != NULL && "mem exhausted\n");
  
  for (size_t i = 0; i < wr.ctrl.mac_ctrl.msg.ran_conf_len; i++) {
  	wr.ctrl.mac_ctrl.msg.ran_conf[i].isset_pusch_mcs = true;
  	wr.ctrl.mac_ctrl.msg.ran_conf[i].pusch_mcs = i + 1;
  	wr.ctrl.mac_ctrl.msg.ran_conf[i].rnti = i + 1; // XXX: collect the RNTI from connected UE and use that value here instead of a fix one.
  }
  
  return wr;
}

static
uint64_t cnt_mac;

static
void sm_cb_mac(sm_ag_if_rd_t const* rd)
{
  assert(rd != NULL);
  assert(rd->type ==INDICATION_MSG_AGENT_IF_ANS_V0);
  assert(rd->ind.type == MAC_STATS_V0);
 
  int64_t now = time_now_us();
  if(cnt_mac % 100 == 0){
    for (size_t i = 0; i < rd->ind.mac.msg.len_ue_stats; i++){
      mac_ue_stats_impl_t* ue_context = &rd->ind.mac.msg.ue_stats[i];
      printf("UE %d:\n", ue_context->rnti);
    }
    printf("MAC ind_msg latency = %ld μs\n", now - rd->ind.mac.msg.tstamp);
    printf("\n");
  }
  cnt_mac++;
}



int main(int argc, char *argv[])
{
  fr_args_t args = init_fr_args(argc, argv);

  //Init the xApp
  init_xapp_api(&args);
  sleep(1);

  e2_node_arr_xapp_t nodes = e2_nodes_xapp_api();
  defer({ free_e2_node_arr_xapp(&nodes); });

  assert(nodes.len > 0);

  printf("Connected E2 nodes = %d\n", nodes.len);

  // MAC indication
  const char* i_0 = "1_ms";
  sm_ans_xapp_t* mac_handle = NULL;

  if(nodes.len > 0){
    mac_handle = calloc( nodes.len, sizeof(sm_ans_xapp_t) ); 
    assert(mac_handle  != NULL);
  }

  for (int i = 0; i < nodes.len; i++) {
    e2_node_connected_xapp_t* n = &nodes.n[i];

    for (size_t j = 0; j < n->len_rf; j++)
      printf("Registered node %d ran func id = %d \n ", i, n->rf[j].id);

    if(n->id.type == ngran_gNB || n->id.type == ngran_gNB_DU){
      // mac report
      mac_handle[i] = report_sm_xapp_api(&nodes.n[i].id, SM_MAC_ID, (void*)i_0, sm_cb_mac);
      assert(mac_handle[i].success == true);

      sleep(1);

      // mac control
  
      sm_ag_if_wr_t ctrl_msg_add = fill_dummy_mac_sm_ctrl_req();
      control_sm_xapp_api(&nodes.n[i].id, SM_MAC_ID, &ctrl_msg_add);
      
    }
  }

 //Stop the xApp
  while(try_stop_xapp_api() == false)
    usleep(1000);

  printf("Test xApp run SUCCESSFULLY\n");
}

