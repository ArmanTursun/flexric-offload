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



#include "mac_enc_plain.h"

#include <assert.h>
#include <stdlib.h>


byte_array_t mac_enc_event_trigger_plain(mac_event_trigger_t const* event_trigger)
{
  assert(event_trigger != NULL);
  byte_array_t  ba = {0};
 
  ba.len = sizeof(event_trigger->ms);
  ba.buf = malloc(ba.len);
  assert(ba.buf != NULL && "Memory exhausted");

  memcpy(ba.buf, &event_trigger->ms, ba.len);

  return ba;
}

byte_array_t mac_enc_action_def_plain(mac_action_def_t const* action_def)
{
  assert(0!=0 && "Not implemented");

  assert(action_def != NULL);
  byte_array_t  ba = {0};
  return ba;
}

byte_array_t mac_enc_ind_hdr_plain(mac_ind_hdr_t const* ind_hdr)
{
  assert(ind_hdr != NULL);

  byte_array_t ba = {0};

  ba.len = sizeof(mac_ind_hdr_t);
  ba.buf = calloc(ba.len,  sizeof(uint8_t));
  assert(ba.buf != NULL && "memory exhausted");
  memcpy(ba.buf, ind_hdr, ba.len);

  return ba;
}


uint32_t cal_ind_ue_msg_len(mac_ue_stats_impl_t *ind_ue_msg){
  //uint32_t len = sizeof(uint64_t) * 8 + sizeof(float) * 4 + sizeof(uint32_t) * 20 + sizeof(uint16_t) * 2 + sizeof(uint8_t) * 5 + sizeof(int8_t) + sizeof(uint32_t) + sizeof(uint32_t) * ind_ue_msg->num_tbs * 5;
  //return len;
  uint32_t len = 0;

  len += sizeof(ind_ue_msg->dl_aggr_tbs);
  len += sizeof(ind_ue_msg->ul_aggr_tbs);
  len += sizeof(ind_ue_msg->dl_aggr_bytes_sdus);
  len += sizeof(ind_ue_msg->ul_aggr_bytes_sdus);
  len += sizeof(ind_ue_msg->dl_curr_tbs);
  len += sizeof(ind_ue_msg->ul_curr_tbs);
  len += sizeof(ind_ue_msg->dl_sched_rb);
  len += sizeof(ind_ue_msg->ul_sched_rb);
 
  len += sizeof(ind_ue_msg->pusch_snr); //: float = -64;
  len += sizeof(ind_ue_msg->pucch_snr); //: float = -64;

  len += sizeof(ind_ue_msg->dl_bler);
  len += sizeof(ind_ue_msg->ul_bler);

  len += sizeof(ind_ue_msg->dl_harq);
  len += sizeof(ind_ue_msg->ul_harq);
  len += sizeof(ind_ue_msg->dl_num_harq);
  len += sizeof(ind_ue_msg->ul_num_harq);

  len += sizeof(ind_ue_msg->rnti);
  len += sizeof(ind_ue_msg->dl_aggr_prb); 
  len += sizeof(ind_ue_msg->ul_aggr_prb);
  len += sizeof(ind_ue_msg->dl_aggr_sdus);
  len += sizeof(ind_ue_msg->ul_aggr_sdus);
  len += sizeof(ind_ue_msg->dl_aggr_retx_prb);
  len += sizeof(ind_ue_msg->ul_aggr_retx_prb);

  len += sizeof(ind_ue_msg->bsr);
  len += sizeof(ind_ue_msg->frame);
  len += sizeof(ind_ue_msg->slot);

  len += sizeof(ind_ue_msg->wb_cqi); 
  len += sizeof(ind_ue_msg->dl_mcs1);
  len += sizeof(ind_ue_msg->ul_mcs1);
  len += sizeof(ind_ue_msg->dl_mcs2); 
  len += sizeof(ind_ue_msg->ul_mcs2); 
  len += sizeof(ind_ue_msg->phr); 
  
  len += sizeof(ind_ue_msg->num_tbs);
  //len += sizeof(ind_ue_msg->tbs);
  //len += sizeof(ind_ue_msg->tbs_frame);
  //len += sizeof(ind_ue_msg->tbs_slot);
  //len += sizeof(ind_ue_msg->tbs_latency);
  //len += sizeof(ind_ue_msg->tbs_crc);
  len += sizeof(tbs_stats_t) * ind_ue_msg->num_tbs;

  return len;

}

uint32_t fill_ind_ue_msg(void* ptr, mac_ue_stats_impl_t *ind_ue_msg)
{
  uint32_t len = cal_ind_ue_msg_len(ind_ue_msg) - sizeof(tbs_stats_t) * ind_ue_msg->num_tbs;
  uint32_t len_ptr = 0;
  memcpy(ptr, ind_ue_msg, len);
  //ptr += len;
  len_ptr += len;
  memcpy(ptr, ind_ue_msg->tbs, sizeof(tbs_stats_t) * ind_ue_msg->num_tbs);
  //ptr += sizeof(tbs_stats_t) * ind_ue_msg->num_tbs;
  len_ptr += sizeof(tbs_stats_t) * ind_ue_msg->num_tbs;
  
  //memcpy(ptr, ind_ue_msg->tbs_frame, sizeof(ind_ue_msg->tbs_frame));
  //ptr += sizeof(ind_ue_msg->tbs_frame);
  //memcpy(ptr, ind_ue_msg->tbs_slot, sizeof(ind_ue_msg->tbs_slot));
  //ptr += sizeof(ind_ue_msg->tbs_slot);
  //memcpy(ptr, ind_ue_msg->tbs_latency, sizeof(ind_ue_msg->tbs_latency));
  //ptr += sizeof(ind_ue_msg->tbs_latency);
  //memcpy(ptr, ind_ue_msg->tbs_crc, sizeof(ind_ue_msg->tbs_crc));
  //ptr += sizeof(ind_ue_msg->tbs_crc);
  return len_ptr;
}

byte_array_t mac_enc_ind_msg_plain(mac_ind_msg_t const* ind_msg)
{
  assert(ind_msg != NULL);

  byte_array_t ba = {0};
  //const uint32_t len = sizeof(ind_msg->len_ue_stats) 
  //                    + sizeof(mac_ue_stats_impl_t) * ind_msg->len_ue_stats
  //                    + sizeof(ind_msg->tstamp); 
  
  uint32_t len = 0;
  //uint32_t len_ptr = 0;
  for(uint32_t i = 0; i < ind_msg->len_ue_stats; ++i){
    len += cal_ind_ue_msg_len(&ind_msg->ue_stats[i]);
  }
  len += sizeof(ind_msg->len_ue_stats);
  len += sizeof(ind_msg->tstamp);

  ba.buf = calloc(1, len); 
  assert(ba.buf != NULL);
  void* ptr = ba.buf;

  memcpy(ba.buf, &ind_msg->len_ue_stats, sizeof(ind_msg->len_ue_stats));
  ptr += sizeof(ind_msg->len_ue_stats);
  //len_ptr += sizeof(ind_msg->len_ue_stats);

  for(uint32_t i = 0; i < ind_msg->len_ue_stats; ++i){
    //memcpy(ptr, &ind_msg->ue_stats[i], sizeof(ind_msg->ue_stats[0]));
    ptr += fill_ind_ue_msg(ptr, &ind_msg->ue_stats[i]); 
    //ptr += sizeof(ind_msg->ue_stats[0]);
    //ptr += ue_len;
    //ptr += len_ptr;
  }

  memcpy(ptr, &ind_msg->tstamp, sizeof(ind_msg->tstamp));
  ptr += sizeof(ind_msg->tstamp);
  //len_ptr += sizeof(ind_msg->tstamp);

  //printf("len = %u, len_ptr = %u \n", len, len_ptr);
  assert(ptr == ba.buf + len && "Data layout mismacth");

  ba.len = len;
  return ba;
}


byte_array_t mac_enc_call_proc_id_plain(mac_call_proc_id_t const* call_proc_id)
{
  assert(0!=0 && "Not implemented");

  assert(call_proc_id != NULL);
  byte_array_t  ba = {0};
  return ba;
}

byte_array_t mac_enc_ctrl_hdr_plain(mac_ctrl_hdr_t const* ctrl_hdr)
{
  assert(ctrl_hdr != NULL);
  byte_array_t  ba = {0};
  ba.len = sizeof(mac_ctrl_hdr_t);
  ba.buf = calloc(ba.len ,sizeof(uint8_t)); 
  assert(ba.buf != NULL);

  memcpy(ba.buf, ctrl_hdr, ba.len);

  return ba;
}

byte_array_t mac_enc_ctrl_msg_plain(mac_ctrl_msg_t const* ctrl_msg)
{
  assert(ctrl_msg != NULL);

  byte_array_t  ba = {0};
  ba.len = sizeof(mac_ctrl_msg_t);
  ba.buf = calloc(ba.len, sizeof(uint8_t)); 
  assert(ba.buf != NULL);

  memcpy(ba.buf, ctrl_msg, ba.len);

  return ba;
}

byte_array_t mac_enc_ctrl_out_plain(mac_ctrl_out_t const* ctrl) 
{

  assert(ctrl != NULL );
  byte_array_t  ba = {0};

  //ba.len = sizeof(ctrl->len_diag) + ctrl->len_diag;

  //ba.buf = malloc(ba.len);
  //assert(ba.buf != NULL && "Memory exhausted");
  //uint8_t* it = ba.buf;

  //memcpy(it, &ctrl->len_diag, sizeof(ctrl->len_diag));
  //it += sizeof(ctrl->len_diag);

  //memcpy(it, ctrl->diagnostic, ctrl->len_diag);
  //it += ctrl->len_diag;

  //assert(it == ba.buf + ba.len);


  return ba;
}

byte_array_t mac_enc_func_def_plain(mac_func_def_t const* func)
{
  assert(0!=0 && "Not implemented");

  assert(func != NULL);
  byte_array_t  ba = {0};
  return ba;
}

