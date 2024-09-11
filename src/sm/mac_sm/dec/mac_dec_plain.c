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

static inline
size_t fill_context(context_stats_t* cnxt, uint8_t const* it)
{
  assert(it != NULL);
  assert(cnxt != NULL);

  memcpy(&cnxt->pusch_snr, it, sizeof(cnxt->pusch_snr));
  it += sizeof(cnxt->pusch_snr);
  size_t sz = sizeof(cnxt->pusch_snr);

  memcpy(&cnxt->pucch_snr, it, sizeof(cnxt->pucch_snr));
  it += sizeof(cnxt->pucch_snr);
  sz += sizeof(cnxt->pucch_snr);

  memcpy(&cnxt->dl_bler, it, sizeof(cnxt->dl_bler));
  it += sizeof(cnxt->dl_bler);
  sz += sizeof(cnxt->dl_bler);

  memcpy(&cnxt->ul_bler, it, sizeof(cnxt->ul_bler));
  it += sizeof(cnxt->ul_bler);
  sz += sizeof(cnxt->ul_bler);

  memcpy(&cnxt->bsr, it, sizeof(cnxt->bsr));
  it += sizeof(cnxt->bsr);
  sz += sizeof(cnxt->bsr);

  memcpy(&cnxt->wb_cqi, it, sizeof(cnxt->wb_cqi));
  it += sizeof(cnxt->wb_cqi);
  sz += sizeof(cnxt->wb_cqi);

  memcpy(&cnxt->dl_mcs1, it, sizeof(cnxt->dl_mcs1));
  it += sizeof(cnxt->dl_mcs1);
  sz += sizeof(cnxt->dl_mcs1);
  
  memcpy(&cnxt->ul_mcs1, it, sizeof(cnxt->ul_mcs1));
  it += sizeof(cnxt->ul_mcs1);
  sz += sizeof(cnxt->ul_mcs1);

  memcpy(&cnxt->dl_mcs2, it, sizeof(cnxt->dl_mcs2));
  it += sizeof(cnxt->dl_mcs2);
  sz += sizeof(cnxt->dl_mcs2);

  memcpy(&cnxt->ul_mcs2, it, sizeof(cnxt->ul_mcs2));
  it += sizeof(cnxt->ul_mcs2);
  sz += sizeof(cnxt->ul_mcs2);

  memcpy(&cnxt->phr, it, sizeof(cnxt->phr));
  it += sizeof(cnxt->phr);
  sz += sizeof(cnxt->phr);

  return sz;
}

static inline
size_t fill_tbs(mac_tbs_stats_t* tbs, uint8_t const* it)
{
  assert(it != NULL);
  assert(tbs != NULL);

  memcpy(&tbs->tbs, it, sizeof(tbs->tbs));
  it += sizeof(tbs->tbs);
  size_t sz = sizeof(tbs->tbs);

  memcpy(&tbs->frame, it, sizeof(tbs->frame));
  it += sizeof(tbs->frame);
  sz += sizeof(tbs->frame);

  memcpy(&tbs->slot, it, sizeof(tbs->slot));
  it += sizeof(tbs->slot);
  sz += sizeof(tbs->slot);

  memcpy(&tbs->latency, it, sizeof(tbs->latency));
  it += sizeof(tbs->latency);
  sz += sizeof(tbs->latency);

  memcpy(&tbs->crc, it, sizeof(tbs->crc));
  it += sizeof(tbs->crc);
  sz += sizeof(tbs->crc);

  return sz;
}

static inline
size_t fill_ue_stats(mac_ue_stats_impl_t* ue, uint8_t const* it)
{
  assert(it != NULL);
  assert(ue != NULL);

  memcpy(&ue->rnti, it, sizeof(ue->rnti));
  it += sizeof(ue->rnti);
  size_t sz = sizeof(ue->rnti);

  size_t temp = fill_context(&ue->context, it);
  it += temp;
  sz += temp;

  memcpy(&ue->num_tbs, it, sizeof(ue->num_tbs));
  it += sizeof(ue->num_tbs);
  sz += sizeof(ue->num_tbs);

  if(ue->num_tbs > 0){
    ue->tbs = calloc(ue->num_tbs, sizeof(mac_tbs_stats_t));
    assert(ue->tbs != NULL && "memory exhausted");
  }

  for(size_t i = 0; i < ue->num_tbs; ++i){
    size_t tmp = fill_tbs(&ue->tbs[i], it);
    it += tmp;
    sz += tmp;
  }

  return sz;
}


mac_ind_msg_t mac_dec_ind_msg_plain(size_t len, uint8_t const ind_msg[len])
{
//  assert(len == sizeof(mac_ind_msg_t)); 
  mac_ind_msg_t ind = {0};

  uint8_t const* it = ind_msg;
  static_assert(sizeof(uint32_t) == sizeof(ind.len_ue_stats), "Different sizes!");

  memcpy(&ind.len_ue_stats, it, sizeof(ind.len_ue_stats));
  it += sizeof(ind.len_ue_stats);
  size_t sz = sizeof(ind.len_ue_stats);

  if(ind.len_ue_stats > 0){
    ind.ue_stats = calloc(ind.len_ue_stats, sizeof(mac_ue_stats_impl_t));
    assert(ind.ue_stats != NULL && "memory exhausted");
  }
  
  for(uint32_t i = 0; i < ind.len_ue_stats; ++i){
    sz = fill_ue_stats(&ind.ue_stats[i], it);
    it += sz;
  }
  
  memcpy(&ind.tstamp, it, sizeof(ind.tstamp));
  it += sizeof(ind.tstamp);
  
  assert(it == ind_msg + len && "data layout mismacth");

  return ind;
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

static inline
size_t fill_ue(mac_ue_ctrl_t* ue, uint8_t const* it)
{
  assert(it != NULL);
  assert(ue != NULL);

  memcpy(&ue->rnti, it, sizeof(ue->rnti));
  it += sizeof(ue->rnti);
  size_t sz = sizeof(ue->rnti);

  memcpy(&ue->offload, it, sizeof(ue->offload));
  it += sizeof(ue->offload);
  sz += sizeof(ue->offload);

  return sz;
}

mac_ctrl_msg_t mac_dec_ctrl_msg_plain(size_t len, uint8_t const ctrl_msg[len])
{
  mac_ctrl_msg_t ctrl = {0};

  uint8_t const* it = ctrl_msg;

  memcpy(&ctrl.action, it, sizeof(ctrl.action));
  it += sizeof(ctrl.action);
  size_t sz = sizeof(ctrl.action);

  memcpy(&ctrl.num_ues, it, sizeof(ctrl.num_ues));
  it += sizeof(ctrl.num_ues);
  sz += sizeof(ctrl.num_ues);

  //if(ctrl.num_ues > 0){
    //ctrl.ues = calloc(ctrl.num_ues, sizeof(mac_ue_ctrl_t));
    //assert(ctrl.ues != NULL && "memory exhausted");
  //}
  
  //for(uint32_t i = 0; i < ctrl.num_ues; ++i){
    //sz = fill_ue(&ctrl.ues[i], it);
    //it += sz;
  //}
  
  memcpy(&ctrl.tms, it, sizeof(ctrl.tms));
  it += sizeof(ctrl.tms);
  
  assert(it == ctrl_msg + len && "data layout mismacth");

  return ctrl;
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


