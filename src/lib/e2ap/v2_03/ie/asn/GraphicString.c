/*-
 * Copyright (c) 2003 Lev Walkin <vlm@lionet.info>. All rights reserved.
 * Redistribution and modifications are permitted subject to BSD license.
 */
#include <asn_internal.h>
#include <GraphicString.h>

/*
 * GraphicString basic type description.
 */
static const ber_tlv_tag_t asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03[] = {
    (ASN_TAG_CLASS_UNIVERSAL | (25 << 2)),  /* [UNIVERSAL 25] IMPLICIT ...*/
    (ASN_TAG_CLASS_UNIVERSAL | (4 << 2))    /* ... OCTET STRING */
};
asn_TYPE_operation_t asn_OP_GraphicString_e2ap_v2_03 = {
    OCTET_STRING_free_e2ap_v2_03,
#if !defined(ASN_DISABLE_PRINT_SUPPORT)
    OCTET_STRING_print_e2ap_v2_03,  /* non-ascii string */
#else
    0,
#endif  /* !defined(ASN_DISABLE_PRINT_SUPPORT) */
    OCTET_STRING_compare_e2ap_v2_03,
#if !defined(ASN_DISABLE_BER_SUPPORT)
    OCTET_STRING_decode_ber_e2ap_v2_03,  /* Implemented in terms of OCTET STRING */
    OCTET_STRING_encode_der_e2ap_v2_03,
#else
    0,
    0,
#endif  /* !defined(ASN_DISABLE_BER_SUPPORT) */
#if !defined(ASN_DISABLE_XER_SUPPORT)
    OCTET_STRING_decode_xer_hex_e2ap_v2_03,
    OCTET_STRING_encode_xer_e2ap_v2_03,  /* Can't expect it to be ASCII/UTF8 */
#else
    0,
    0,
#endif  /* !defined(ASN_DISABLE_XER_SUPPORT) */
#if !defined(ASN_DISABLE_JER_SUPPORT)
    OCTET_STRING_encode_jer,  /* Can't expect it to be ASCII/UTF8 */
#else
    0,
#endif  /* !defined(ASN_DISABLE_JER_SUPPORT) */
#if !defined(ASN_DISABLE_OER_SUPPORT)
    OCTET_STRING_decode_oer,
    OCTET_STRING_encode_oer,
#else
    0,
    0,
#endif  /* !defined(ASN_DISABLE_OER_SUPPORT) */
#if !defined(ASN_DISABLE_UPER_SUPPORT)
    OCTET_STRING_decode_uper_e2ap_v2_03,  /* Implemented in terms of OCTET STRING */
    OCTET_STRING_encode_uper_e2ap_v2_03,
#else
    0,
    0,
#endif  /* !defined(ASN_DISABLE_UPER_SUPPORT) */
#if !defined(ASN_DISABLE_APER_SUPPORT)
    OCTET_STRING_decode_aper_e2ap_v2_03,  /* Implemented in terms of OCTET STRING */
    OCTET_STRING_encode_aper_e2ap_v2_03,
#else
    0,
    0,
#endif  /* !defined(ASN_DISABLE_APER_SUPPORT) */
#if !defined(ASN_DISABLE_RFILL_SUPPORT)
    OCTET_STRING_random_fill_e2ap_v2_03,
#else
    0,
#endif  /* !defined(ASN_DISABLE_RFILL_SUPPORT) */
    0  /* Use generic outmost tag fetcher */
};
asn_TYPE_descriptor_t asn_DEF_GraphicString_e2ap_v2_03 = {
    "GraphicString",
    "GraphicString",
    &asn_OP_GraphicString_e2ap_v2_03,
    asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03,
    sizeof(asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03)
      / sizeof(asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03[0]) - 1,
    asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03,
    sizeof(asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03)
      / sizeof(asn_DEF_GraphicString_e2ap_v2_03_tags_e2ap_v2_03[0]),
    {
#if !defined(ASN_DISABLE_OER_SUPPORT)
        0,
#endif  /* !defined(ASN_DISABLE_OER_SUPPORT) */
#if !defined(ASN_DISABLE_UPER_SUPPORT) || !defined(ASN_DISABLE_APER_SUPPORT)
        0,
#endif  /* !defined(ASN_DISABLE_UPER_SUPPORT) || !defined(ASN_DISABLE_APER_SUPPORT) */
        asn_generic_unknown_constraint_e2ap_v2_03
    },
    0, 0,  /* No members */
    0  /* No specifics */
};
