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



#include "mac_dec_plain.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

mac_event_trigger_t mac_dec_event_trigger_plain(size_t len, uint8_t const ev_tr[len])
{
  mac_event_trigger_t ev = {0};
  memcpy(&ev.ms, ev_tr, sizeof(ev.ms));
  return ev;
}

mac_action_def_t mac_dec_action_def_plain(size_t len, uint8_t const action_def[len])
{
  assert(0!=0 && "Not implemented");
  assert(action_def != NULL);
  mac_action_def_t act_def;// = {0};
  return act_def;
}

mac_ind_hdr_t mac_dec_ind_hdr_plain(size_t len, uint8_t const ind_hdr[len])
{
  assert(len == sizeof(mac_ind_hdr_t)); 
  mac_ind_hdr_t ret;
  memcpy(&ret, ind_hdr, len);
  return ret;
}

uint32_t cal_ind_ue_msg_len_half(mac_ue_stats_impl_t *ind_ue_msg){
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
  //len += sizeof(tbs_stats_t) * ind_ue_msg->num_tbs;

  return len;

}

mac_ind_msg_t mac_dec_ind_msg_plain(size_t len, uint8_t const ind_msg[len])
{
//  assert(len == sizeof(mac_ind_msg_t)); 
  mac_ind_msg_t ret;

  static_assert(sizeof(uint32_t) == sizeof(ret.len_ue_stats), "Different sizes!");

  const size_t len_sizeof = sizeof(ret.len_ue_stats);
  memcpy(&ret.len_ue_stats, ind_msg, len_sizeof);

  if(ret.len_ue_stats > 0){
    ret.ue_stats = calloc(ret.len_ue_stats, sizeof(mac_ue_stats_impl_t));
    assert(ret.ue_stats != NULL && "Memory exhausted!");
  }
  
  void* ptr = (void*)&ind_msg[len_sizeof];
  
  for(uint32_t i = 0; i < ret.len_ue_stats; ++i){
    mac_ue_stats_impl_t *ind_ue_msg = &ret.ue_stats[i];
    //memcpy(&ret.ue_stats[i], ptr, sizeof( mac_ue_stats_impl_t) );
    //uint32_t ue_len = sizeof(uint64_t) * 8 + sizeof(float) * 4 + sizeof(uint32_t) * 20 + sizeof(uint16_t) * 2 + sizeof(uint8_t) * 5 + sizeof(int8_t) + sizeof(uint32_t);
    uint32_t ue_len = cal_ind_ue_msg_len_half(ind_ue_msg);
    memcpy(ind_ue_msg, ptr, ue_len );
    //ptr += sizeof( mac_ue_stats_impl_t); 
    ptr += ue_len; 

    ind_ue_msg->tbs = calloc(ind_ue_msg->num_tbs, sizeof(tbs_stats_t));
    //tbs_stats_t* tbs = ind_ue_msg->tbs;
    memcpy(ind_ue_msg->tbs, ptr, sizeof(tbs_stats_t) * ind_ue_msg->num_tbs);
    ptr += sizeof(tbs_stats_t) * ind_ue_msg->num_tbs;

/*
    mac_ue_stats_impl_t *ind_ue_msg = &ret.ue_stats[i];

    ind_ue_msg->tbs = calloc(ind_ue_msg->num_tbs, sizeof(uint32_t));
    memcpy(ind_ue_msg->tbs, ptr, sizeof(uint32_t) * ind_ue_msg->num_tbs);
    ptr += sizeof(uint32_t) * ind_ue_msg->num_tbs;

    ind_ue_msg->tbs_frame = calloc(ind_ue_msg->num_tbs, sizeof(uint32_t));
    memcpy(ind_ue_msg->tbs_frame, ptr, sizeof(uint32_t) * ind_ue_msg->num_tbs);
    ptr += sizeof(uint32_t) * ind_ue_msg->num_tbs;

    ind_ue_msg->tbs_slot = calloc(ind_ue_msg->num_tbs, sizeof(uint32_t));
    memcpy(ind_ue_msg->tbs_slot, ptr, sizeof(uint32_t) * ind_ue_msg->num_tbs);
    ptr += sizeof(uint32_t) * ind_ue_msg->num_tbs;

    ind_ue_msg->tbs_latency = calloc(ind_ue_msg->num_tbs, sizeof(uint32_t));
    memcpy(ind_ue_msg->tbs_latency, ptr, sizeof(uint32_t) * ind_ue_msg->num_tbs);
    ptr += sizeof(uint32_t) * ind_ue_msg->num_tbs;

    ind_ue_msg->tbs_crc = calloc(ind_ue_msg->num_tbs, sizeof(uint32_t));
    memcpy(ind_ue_msg->tbs_crc, ptr, sizeof(uint32_t) * ind_ue_msg->num_tbs);
    ptr += sizeof(uint32_t) * ind_ue_msg->num_tbs;
*/
  }

  memcpy(&ret.tstamp, ptr, sizeof(ret.tstamp));

  ptr += sizeof(ret.tstamp);
  assert(ptr == ind_msg + len && "data layout mismacth");

  return ret;
}

mac_call_proc_id_t mac_dec_call_proc_id_plain(size_t len, uint8_t const call_proc_id[len])
{
  assert(0!=0 && "Not implemented");
  assert(call_proc_id != NULL);
}

mac_ctrl_hdr_t mac_dec_ctrl_hdr_plain(size_t len, uint8_t const ctrl_hdr[len])
{
  assert(len == sizeof(mac_ctrl_hdr_t)); 
  mac_ctrl_hdr_t ret;
  memcpy(&ret, ctrl_hdr, len);
  return ret;
}

mac_ctrl_msg_t mac_dec_ctrl_msg_plain(size_t len, uint8_t const ctrl_msg[len])
{
  assert(len == sizeof(mac_ctrl_msg_t)); 
  mac_ctrl_msg_t ret;
  memcpy(&ret, ctrl_msg, len);
  return ret;
}

mac_ctrl_out_t mac_dec_ctrl_out_plain(size_t len, uint8_t const ctrl_out[len]) 
{
  assert(ctrl_out != NULL);
  mac_ctrl_out_t ret = {0};

  return ret;
}

mac_func_def_t mac_dec_func_def_plain(size_t len, uint8_t const func_def[len])
{
  assert(0!=0 && "Not implemented");
  assert(func_def != NULL);
}


