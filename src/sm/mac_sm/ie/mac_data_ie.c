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


#include "mac_data_ie.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../../../util/alg_ds/alg/eq_float.h"


//////////////////////////////////////
// RIC Event Trigger Definition
/////////////////////////////////////

void free_mac_event_trigger(mac_event_trigger_t* src)
{
  assert(src != NULL);
  assert(0!=0 && "Not implemented" ); 
}

mac_event_trigger_t cp_mac_event_trigger( mac_event_trigger_t const* src)
{
  assert(src != NULL);
  assert(0!=0 && "Not implemented" ); 

  mac_event_trigger_t et = {0};
  return et;
}

bool eq_mac_event_trigger(mac_event_trigger_t const* m0, mac_event_trigger_t const* m1)
{
  assert(m0 != NULL);
  assert(m1 != NULL);

  assert(0!=0 && "Not implemented" ); 

  return true;
}


//////////////////////////////////////
// RIC Action Definition 
/////////////////////////////////////

void free_mac_action_def(mac_action_def_t* src)
{
  assert(src != NULL);

  assert(0!=0 && "Not implemented" ); 
}

mac_action_def_t cp_mac_action_def(mac_action_def_t* src)
{
  assert(src != NULL);

  assert(0!=0 && "Not implemented" ); 
  mac_action_def_t ad = {0};
  return ad;
}

bool eq_mac_action_def(mac_event_trigger_t* m0,  mac_event_trigger_t* m1)
{
  assert(m0 != NULL);
  assert(m1 != NULL);

  assert(0!=0 && "Not implemented" ); 

  return true;
}


//////////////////////////////////////
// RIC Indication Header 
/////////////////////////////////////


void free_mac_ind_hdr(mac_ind_hdr_t* src)
{
  assert(src != NULL);
  (void)src;
}

mac_ind_hdr_t cp_mac_ind_hdr(mac_ind_hdr_t const* src)
{
  assert(src != NULL);
  mac_ind_hdr_t dst = {0}; 
  dst.dummy = src->dummy;
  return dst;
}

bool eq_mac_ind_hdr(mac_ind_hdr_t* m0, mac_ind_hdr_t* m1)
{
  assert(m0 != 0);
  assert(m1 != 0);

  if(m0->dummy != m1->dummy)
    return false;
  return true;
}


//////////////////////////////////////
// RIC Indication Message 
/////////////////////////////////////


void free_ue_stats(mac_ue_stats_impl_t* src)
{
  assert(src != NULL);
  if(src->num_tbs > 0){
    assert(src->tbs != NULL);
    free(src->tbs);
  }
}

void free_mac_ind_msg(mac_ind_msg_t* src)
{
  assert(src != NULL);
  if(src->len_ue_stats > 0){
    assert(src->ue_stats != NULL);
    for (uint32_t i = 0; i < src->len_ue_stats; i++)
      free_tbs(src->ue_stats);
    free(src->ue_stats);
  }
}


bool eq_context(context_stats_t* ue0, context_stats_t* ue1)
{
  assert(ue0 != NULL);
  assert(ue1 != NULL);

  if(
        ue0->wb_cqi != ue1->wb_cqi || 
        ue0->dl_mcs1 != ue1->dl_mcs1 ||
        ue0->ul_mcs1 != ue1->ul_mcs1 ||
        ue0->dl_mcs2 != ue1->dl_mcs2 || 
        ue0->ul_mcs2 != ue1->ul_mcs2 || 
        ue0->phr != ue1->phr || 
        ue0->bsr != ue1->bsr ||
        ue0->dl_bler != ue1->dl_bler ||
        ue0->ul_bler != ue1->ul_bler ||

        eq_float(ue0->pusch_snr, ue1->pusch_snr, 0.0000001) == false ||
        eq_float(ue0->pucch_snr, ue0->pucch_snr, 0.0000001) == false 
  )
    return false;
  return true;
}

bool eq_tbs(mac_tbs_stats_t* ue0, mac_tbs_stats_t* ue1)
{
  assert(ue0 != NULL);
  assert(ue1 != NULL);

  if(
        ue0->tbs != ue1->tbs || 
        ue0->frame != ue1->frame ||
        ue0->slot != ue1->slot ||
        ue0->latency != ue1->latency || 
        ue0->crc != ue1->crc
  )
    return false;
  return true;
}

bool eq_mac_ind_msg(mac_ind_msg_t* m0, mac_ind_msg_t* m1)
{
  assert(m0 != NULL);
  assert(m1 != NULL);

  if(m0->len_ue_stats != m1->len_ue_stats || m0->tstamp != m1->tstamp)
    return false;

  for(uint32_t i = 0 ; i < m0->len_ue_stats; ++i){
    mac_ue_stats_impl_t* ue0 = &m0->ue_stats[i]; 
    mac_ue_stats_impl_t* ue1 = &m1->ue_stats[i]; 
    if (ue0->rnti != ue1->rnti)
      return false;
    if (!eq_context(&ue0->context, &ue1->context)){
      printf("context data is not equal\n");
      return false;
    }
    if (ue0->num_tbs != ue1->num_tbs)
      return false;
    if (ue0->num_tbs > 0){
      for (uint32_t i = 0; i < ue0->num_tbs; i++){
        if (!eq_tbs(&ue0->tbs[i], &ue1->tbs[i])){
          printf("tbs data is not equal\n");
          return false;
        }
      }
    }
  }
  return true;
}

mac_tbs_stats_t cp_tbs_stats_impl(mac_tbs_stats_t const* src)
{
  assert(src != NULL);
  mac_tbs_stats_t dst = {0};

  dst.tbs = src->tbs;
  dst.frame = src->frame;
  dst.slot = src->slot; 
  dst.latency = src->latency;
  dst.crc = src->crc;
  return dst;
}

context_stats_t cp_context_stats_impl(context_stats_t const* src)
{
  assert(src != NULL);
  context_stats_t dst = {0};

  dst.pusch_snr = src->pusch_snr; //: float = -64;
  dst.pucch_snr = src->pucch_snr; //: float = -64;

  dst.wb_cqi = src->wb_cqi;
  dst.dl_mcs1 = src->dl_mcs1;
  dst.ul_mcs1 = src->ul_mcs1;
  dst.dl_mcs2 = src->dl_mcs2; 
  dst.ul_mcs2 = src->ul_mcs2; 
  dst.phr = src->phr;
  dst.bsr = src->bsr;
  dst.dl_bler = src->dl_bler;
  dst.ul_bler = src->ul_bler;
  return dst;
}

mac_ue_stats_impl_t cp_mac_ue_stats_impl(mac_ue_stats_impl_t const* src)
{
  assert(src != NULL);
  mac_ue_stats_impl_t dst = {0};
  
  dst.rnti = src->rnti;
  dst.context = cp_context_stats_impl(&src->context);
  dst.num_tbs = src->num_tbs;

  if(dst.num_tbs > 0){
    dst.tbs = calloc(dst.num_tbs, sizeof( mac_tbs_stats_t) );
    assert(dst.tbs != NULL && "Memory exhausted");
  }

  for (uint32_t i = 0; i < dst.num_tbs; i++){
    dst.tbs[i] = cp_tbs_stats_impl(&src->tbs[i]);
  }
  
  return dst;
}


mac_ind_msg_t cp_mac_ind_msg( mac_ind_msg_t const* src)
{
  assert(src != NULL);

  mac_ind_msg_t dst = {0};

  dst.len_ue_stats = src->len_ue_stats;
  if(dst.len_ue_stats > 0){
    dst.ue_stats = calloc(dst.len_ue_stats, sizeof( mac_ue_stats_impl_t) );
    assert(dst.ue_stats != NULL && "Memory exhausted");
  }

  for(size_t i = 0; i < dst.len_ue_stats; ++i){
    dst.ue_stats[i] = cp_mac_ue_stats_impl(&src->ue_stats[i]); 
  }

  //memcpy(ret.ue_stats, src->ue_stats, sizeof( mac_ue_stats_impl_t )*ret.len_ue_stats);

  dst.tstamp = src->tstamp; 

  assert(eq_mac_ind_msg(src, &dst) && "mac_ind_msg src and dst is not equal");

  return dst;
}

//////////////////////////////////////
// RIC Call Process ID 
/////////////////////////////////////

void free_mac_call_proc_id(mac_call_proc_id_t* src)
{
  // Note that the src could be NULL
  free(src);
}

mac_call_proc_id_t cp_mac_call_proc_id( mac_call_proc_id_t* src)
{
  assert(src != NULL); 
  mac_call_proc_id_t dst = {0};

  dst.dummy = src->dummy;

  return dst;
}

bool eq_mac_call_proc_id(mac_call_proc_id_t* m0, mac_call_proc_id_t* m1)
{
  if(m0 == NULL && m1 == NULL)
    return true;
  if(m0 == NULL)
    return false;
  if(m1 == NULL)
    return false;

  if(m0->dummy != m1->dummy)
    return false;

  return true;
}


//////////////////////////////////////
// RIC Control Header 
/////////////////////////////////////

void free_mac_ctrl_hdr( mac_ctrl_hdr_t* src)
{

  assert(src != NULL);
  //assert(0!=0 && "Not implemented" );
  free(src);
}

mac_ctrl_hdr_t cp_mac_ctrl_hdr(mac_ctrl_hdr_t* src)
{
  assert(src != NULL);
  //assert(0!=0 && "Not implemented" );
  //mac_ctrl_hdr_t ret = {0};
  //return ret;
  mac_ctrl_hdr_t dst = {.dummy = src->dummy};
  return dst;
}

bool eq_mac_ctrl_hdr(mac_ctrl_hdr_t* m0, mac_ctrl_hdr_t* m1)
{
  assert(m0 != NULL);
  assert(m1 != NULL);

  //assert(0!=0 && "Not implemented" );

  //return true;
  return m0->dummy == m1->dummy;
}


//////////////////////////////////////
// RIC Control Message 
/////////////////////////////////////


void free_mac_ctrl_msg( mac_ctrl_msg_t* src)
{
  assert(src != NULL);

  //assert(0!=0 && "Not implemented" );
  free(src);
}

mac_ctrl_msg_t cp_mac_ctrl_msg(mac_ctrl_msg_t* src)
{
  assert(src != NULL);

  //assert(0!=0 && "Not implemented" );
  //mac_ctrl_msg_t ret = {0};
  //return ret;
  mac_ctrl_msg_t dst = {0};
  dst.action = src->action;
  dst.offload = src->offload;
  dst.tms = src->tms;
  return dst;
}

bool eq_mac_ctrl_msg(mac_ctrl_msg_t* m0, mac_ctrl_msg_t* m1)
{
  assert(m0 != NULL);
  assert(m1 != NULL);

  //assert(0!=0 && "Not implemented" );

  //return true;
  if (m0->action != m1->action || m0->offload != m1->offload || 
    m0->tms != m1->tms) {
    return false;
  }
  return true;
}


//////////////////////////////////////
// RIC Control Outcome 
/////////////////////////////////////

void free_mac_ctrl_out(mac_ctrl_out_t* src)
{
  assert(src != NULL);
  free(src);

  //assert(0!=0 && "Not implemented" );
  // Do nothing
}

mac_ctrl_out_t cp_mac_ctrl_out(mac_ctrl_out_t* src)
{
  assert(src != NULL);

  //assert(0!=0 && "Not implemented" );
  mac_ctrl_out_t ret = {0}; 
  return ret;
}

bool eq_mac_ctrl_out(mac_ctrl_out_t* m0, mac_ctrl_out_t* m1)
{
  assert(m0 != NULL);
  assert(m1 != NULL);

  //assert(0!=0 && "Not implemented" );

  return true;
}


//////////////////////////////////////
// RAN Function Definition 
/////////////////////////////////////

void free_mac_func_def(mac_func_def_t* src)
{
  assert(src != NULL);
  free(src->buf);
}

mac_func_def_t cp_mac_func_def(mac_func_def_t const* src)
{
  assert(src != NULL);

  mac_func_def_t dst = {.len = src->len};
  if(src->len > 0){
    dst.buf = calloc(dst.len, sizeof(uint8_t)); 
    assert(dst.buf != NULL && "memory exhausted");
    memcpy(dst.buf, src->buf, dst.len);
  }

  return dst;
}

bool eq_mac_func_def(mac_func_def_t const* m0, mac_func_def_t const* m1)
{
  if(m0 == m1)
    return true;

  if(m0 == NULL || m1 == NULL)
    return false;

  if(m0->len != m1->len)
    return false;

  int rc = memcmp(m0, m1, m0->len);
  return rc == 0;
}

///////////////
// RIC Indication
///////////////

mac_ind_data_t cp_mac_ind_data( mac_ind_data_t const* src)
{
  assert(src != NULL);
  mac_ind_data_t dst = {0};
  dst.hdr = cp_mac_ind_hdr(&src->hdr);
  dst.msg = cp_mac_ind_msg(&src->msg);
  
  if(src->proc_id != NULL){
    dst.proc_id = malloc(sizeof(mac_call_proc_id_t)); 
    assert(dst.proc_id != NULL && "Memory exhausted");
    *dst.proc_id = cp_mac_call_proc_id(src->proc_id);
  }

  return dst;
}

void free_mac_ind_data(mac_ind_data_t* ind)
{
  assert(ind != NULL);
  free_mac_ind_hdr(&ind->hdr);
  free_mac_ind_msg(&ind->msg);
  free_mac_call_proc_id(ind->proc_id);
}


