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

static
size_t cal_context(context_stats_t const* ind_msg)
{
  assert(ind_msg != NULL);

  size_t sz = sizeof(ind_msg->pusch_snr);
  sz += sizeof(ind_msg->pucch_snr);
  sz += sizeof(ind_msg->dl_bler);
  sz += sizeof(ind_msg->ul_bler);
  sz += sizeof(ind_msg->bsr);
  sz += sizeof(ind_msg->wb_cqi); 
  sz += sizeof(ind_msg->dl_mcs1);
  sz += sizeof(ind_msg->ul_mcs1);
  sz += sizeof(ind_msg->dl_mcs2); 
  sz += sizeof(ind_msg->ul_mcs2); 
  sz += sizeof(ind_msg->phr);
  return sz;
}

static
size_t cal_tbs(mac_tbs_stats_t const* ind_msg)
{
  assert(ind_msg != NULL);

  size_t sz = sizeof(ind_msg->tbs);
  sz += sizeof(ind_msg->frame);
  sz += sizeof(ind_msg->slot);
  sz += sizeof(ind_msg->latency);
  sz += sizeof(ind_msg->crc);

  return sz;
}

static
size_t cal_ind_msg_payload(mac_ue_stats_impl_t const* ind_msg)
{
  assert(ind_msg != NULL);

  size_t sz = sizeof(ind_msg->rnti);
  sz += cal_context(&ind_msg->context);
  sz += sizeof(ind_msg->num_tbs);
  for (uint32_t i = 0; i < ind_msg->num_tbs; i++)
    sz += cal_tbs(&ind_msg->tbs[i]);

  return sz;
}

static
uint8_t* end;

static inline
size_t fill_context(uint8_t* it, context_stats_t* cnxt)
{
  assert(it != NULL);
  assert(cnxt != NULL);

  assert(it < end && "could not fill more context");

  memcpy(it, &cnxt->pusch_snr, sizeof(cnxt->pusch_snr));
  it += sizeof(cnxt->pusch_snr);
  size_t sz = sizeof(cnxt->pusch_snr);

  memcpy(it, &cnxt->pucch_snr, sizeof(cnxt->pucch_snr));
  it += sizeof(cnxt->pucch_snr);
  sz += sizeof(cnxt->pucch_snr);

  memcpy(it, &cnxt->dl_bler, sizeof(cnxt->dl_bler));
  it += sizeof(cnxt->dl_bler);
  sz += sizeof(cnxt->dl_bler);

  memcpy(it, &cnxt->ul_bler, sizeof(cnxt->ul_bler));
  it += sizeof(cnxt->ul_bler);
  sz += sizeof(cnxt->ul_bler);

  memcpy(it, &cnxt->bsr, sizeof(cnxt->bsr));
  it += sizeof(cnxt->bsr);
  sz += sizeof(cnxt->bsr);

  memcpy(it, &cnxt->wb_cqi, sizeof(cnxt->wb_cqi));
  it += sizeof(cnxt->wb_cqi);
  sz += sizeof(cnxt->wb_cqi);

  memcpy(it, &cnxt->dl_mcs1, sizeof(cnxt->dl_mcs1));
  it += sizeof(cnxt->dl_mcs1);
  sz += sizeof(cnxt->dl_mcs1);
  
  memcpy(it, &cnxt->ul_mcs1, sizeof(cnxt->ul_mcs1));
  it += sizeof(cnxt->ul_mcs1);
  sz += sizeof(cnxt->ul_mcs1);

  memcpy(it, &cnxt->dl_mcs2, sizeof(cnxt->dl_mcs2));
  it += sizeof(cnxt->dl_mcs2);
  sz += sizeof(cnxt->dl_mcs2);

  memcpy(it, &cnxt->ul_mcs2, sizeof(cnxt->ul_mcs2));
  it += sizeof(cnxt->ul_mcs2);
  sz += sizeof(cnxt->ul_mcs2);

  memcpy(it, &cnxt->phr, sizeof(cnxt->phr));
  it += sizeof(cnxt->phr);
  sz += sizeof(cnxt->phr);

  return sz;
}

static inline
size_t fill_tbs(uint8_t* it, mac_tbs_stats_t* tbs)
{
  assert(it != NULL);
  assert(tbs != NULL);

  assert(it < end && "could not fill more tbs");

  memcpy(it, &tbs->tbs, sizeof(tbs->tbs));
  it += sizeof(tbs->tbs);
  size_t sz = sizeof(tbs->tbs);

  memcpy(it, &tbs->frame, sizeof(tbs->frame));
  it += sizeof(tbs->frame);
  sz += sizeof(tbs->frame);

  memcpy(it, &tbs->slot, sizeof(tbs->slot));
  it += sizeof(tbs->slot);
  sz += sizeof(tbs->slot);

  memcpy(it, &tbs->latency, sizeof(tbs->latency));
  it += sizeof(tbs->latency);
  sz += sizeof(tbs->latency);

  memcpy(it, &tbs->crc, sizeof(tbs->crc));
  it += sizeof(tbs->crc);
  sz += sizeof(tbs->crc);

  return sz;
}

static inline
size_t fill_ue_stats(uint8_t* it, mac_ue_stats_impl_t* ue)
{
  assert(it != NULL);
  assert(ue != NULL);

  memcpy(it, &ue->rnti, sizeof(ue->rnti));
  it += sizeof(ue->rnti);
  size_t sz = sizeof(ue->rnti);

  size_t pos1 = fill_context(it, &ue->context);
  it += pos1;
  sz += pos1;

  memcpy(it, &ue->num_tbs, sizeof(ue->num_tbs));
  it += sizeof(ue->num_tbs);
  sz += sizeof(ue->num_tbs);

  for(size_t i = 0; i < ue->num_tbs; ++i){
    size_t tmp = fill_tbs(it, &ue->tbs[i]);
    it += tmp;
    sz += tmp;
  }

  return sz;
}


byte_array_t mac_enc_ind_msg_plain(mac_ind_msg_t const* ind_msg)
{
  assert(ind_msg != NULL);

  byte_array_t ba = {0};

  size_t sz = sizeof(ind_msg->len_ue_stats);
  for (uint32_t i = 0; i < ind_msg->len_ue_stats; i++)
    sz += cal_ind_msg_payload(&ind_msg->ue_stats[i]);
  sz += sizeof(ind_msg->tstamp);
  

  ba.buf = malloc(sz);
  assert(ba.buf != NULL && "Memory exhausted");
  end = ba.buf + sz;

  uint8_t* it = ba.buf;
  memcpy(it, &ind_msg->len_ue_stats, sizeof(ind_msg->len_ue_stats));
  it += sizeof(ind_msg->len_ue_stats);

  for(uint32_t i = 0; i < ind_msg->len_ue_stats; ++i){
    size_t pos1 = fill_ue_stats(it, &ind_msg->ue_stats[i]);
    it += pos1;
  }

  memcpy(it, &ind_msg->tstamp, sizeof(ind_msg->tstamp));
  it += sizeof(ind_msg->tstamp);
  
  assert(it == ba.buf + sz && "Data layout mismacth");

  ba.len = sz;
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


////////////////// ctrl msg
static
uint8_t* ctrl_end;

static
size_t cal_ctrl_ue_msg(mac_ue_ctrl_t const* ue)
{
  assert(ue != NULL);

  size_t sz = sizeof(ue->rnti);
  sz += sizeof(ue->offload);

  return sz;
}

static inline
size_t fill_ue(uint8_t* it, mac_ue_ctrl_t* ue)
{
  assert(it != NULL);
  assert(ue != NULL);

  assert(it < ctrl_end && "could not fill more ues to ctrl msg");

  memcpy(it, &ue->rnti, sizeof(ue->rnti));
  it += sizeof(ue->rnti);
  size_t sz = sizeof(ue->rnti);

  memcpy(it, &ue->offload, sizeof(ue->offload));
  it += sizeof(ue->offload);
  sz += sizeof(ue->offload);

  return sz;
}

byte_array_t mac_enc_ctrl_msg_plain(mac_ctrl_msg_t const* ctrl_msg)
{
  assert(ctrl_msg != NULL);

  byte_array_t ba = {0};

  size_t sz = sizeof(ctrl_msg->action);
  sz += sizeof(ctrl_msg->num_ues);
  for (uint32_t i = 0; i < ctrl_msg->num_ues; i++){
    sz += cal_ctrl_ue_msg(&ctrl_msg->ues[i]);
  }
  sz += sizeof(ctrl_msg->tms);
  
  ba.buf = malloc(sz);
  assert(ba.buf != NULL && "Memory exhausted");
  ctrl_end = ba.buf + sz;

  uint8_t* it = ba.buf;
  memcpy(it, &ctrl_msg->action, sizeof(ctrl_msg->action));
  it += sizeof(ctrl_msg->action);

  memcpy(it, &ctrl_msg->num_ues, sizeof(ctrl_msg->num_ues));
  it += sizeof(ctrl_msg->num_ues);

  for(uint32_t i = 0; i < ctrl_msg->num_ues; ++i){
    size_t pos1 = fill_ue(it, &ctrl_msg->ues[i]);
    it += pos1;
  }

  memcpy(it, &ctrl_msg->tms, sizeof(ctrl_msg->tms));
  it += sizeof(ctrl_msg->tms);
  
  assert(it == ba.buf + sz && "Data layout mismacth");

  ba.len = sz;
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

