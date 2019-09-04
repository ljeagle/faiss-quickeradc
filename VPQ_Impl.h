/**
 * Copyright (c) 2018 - Present – Thomson Licensing, SAS
 * Copyright (c) 2019 - Present – Nicolas Le Scouarnec
 * All rights reserved.
 *
 * This source code is licensed under the Clear BSD license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_VPQI_H
#define FAISS_VPQI_H

#ifndef SWIG
#include "VecProductQuantizer.h"
#include "IndexVPQ.h"
#include "IndexIVFVPQ.h"
#endif

namespace faiss {
// First character: precision (1: 32 bits, 2: 64 bits, 4: 128 bits, ...)
// Second character is the implementation:
//     q : quantized scalar,
//     t: quantized scalar (table based)
//     s: non-quantized scalar
//     v: vectorized
//     V,T,Q,S: cross-lane (2 lanes) implem (pshufb over AVX256)
//     W: cross-lane (4 lanes) implem (pshufb over AVX512)
//
//  Saved indexes can be converted by editing the file with an hexeditor
//  and changing the letter as long as the layout and precision remain the same
//
// Third character is the layout:
//   4,4 => 4  (unsigned), F (signed)
//   6,6,4 => 6 (unsigned), S (signed)
//   6,5,5 => a (unsigned), A (signed)
//   5,5,5 => b (unsigned), B (signed)
//   8,8 => c (unsigned), C (signed)
//   8 => 8 (unsigned), H (signed)
//   4,4 (Bolt - accumulation on 16 bits) => r
//   4,4 (Bolt - accumulation on 8 bits) => R
//   4,4 (QuickADC - global minimum) => q


#ifdef SWIG
#define DVP_TAIL(b,...) \
	__VA_ARGS__


#define DVP_HEAD_CONCAT(a,b,...) \
	a##b

#define DVP(a,...) \
	%template(DVP_HEAD_CONCAT(IndexVPQ_,__VA_ARGS__)) faiss::IndexVPQ<faiss::DVP_TAIL(__VA_ARGS__)>; \

#else
#define DVP(fn, a,...) \
	typedef __VA_ARGS__ VPQ_##a; \
	typedef IndexVPQ<__VA_ARGS__> IndexVPQ_##a ; \
	typedef IndexIVFVPQ<__VA_ARGS__> IndexIVFVPQ_##a ; \
	inline std::string cc_vpq(const VPQ_##a * n){ return #a;} \
	inline std::string factName(const VPQ_##a * n){ return #fn;}
#endif


// -----------------------------------------------------------------------------
// 64bit quantizers
DVP(VPQ_12x6.6.4_NV    , 2q6 , VecProductQuantizer_NoVec<12,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_12x6.5.5_NV    , 2qa , VecProductQuantizer_NoVec<12,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_12x5.5.5_NV    , 2qb , VecProductQuantizer_NoVec<12,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_16x4.4_NV      , 2q4 , VecProductQuantizer_NoVec<16,2,4,4,0,0,uint8_t,__m128i,__m128i>)
DVP(VPQ_16x4.4_NV_AVX2 , 2Q4 , VecProductQuantizer_NoVec<16,2,4,4,0,0,uint8_t,__m128i,__m256i>)

DVP(VPQ_12x6.6.4_NVT    , 2t6 , VecProductQuantizer_NoVecTable<12,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_12x6.5.5_NVT    , 2ta , VecProductQuantizer_NoVecTable<12,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_12x5.5.5_NVT    , 2tb , VecProductQuantizer_NoVecTable<12,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_16x4.4_NVT      , 2t4 , VecProductQuantizer_NoVecTable<16,2,4,4,0,0,uint8_t,__m128i,__m128i>)
DVP(VPQ_16x4.4_NVT_AVX2 , 2T4 , VecProductQuantizer_NoVecTable<16,2,4,4,0,0,uint8_t,__m128i,__m256i>)

DVP(VPQ_12x6.6.4_NQ    , 2s6 , VecProductQuantizer_NoVecNoQuant<12,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_12x6.6.4_NQ    , 2sa , VecProductQuantizer_NoVecNoQuant<12,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_12x6.6.4_NQ    , 2sb , VecProductQuantizer_NoVecNoQuant<12,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_16x4.4_NQ      , 2s4 , VecProductQuantizer_NoVecNoQuant<16,2,4,4,0,0,uint8_t,__m128i,__m128i>)
DVP(VPQ_16x4.4_NQ_AVX2 , 2S4 , VecProductQuantizer_NoVecNoQuant<16,2,4,4,0,0,uint8_t,__m128i,__m256i>)

DVP(VPQ_12x6.6.4_AVX512, 2v6 , VecProductQuantizer_XYZ_AVX512_unsigned<12,6,6,4>)
DVP(VPQ_12x6.5.5_AVX512, 2va , VecProductQuantizer_XYZ_AVX512_unsigned<12,6,5,5>)
DVP(VPQ_12x5.5.5_AVX512, 2vb , VecProductQuantizer_XYZ_AVX512_unsigned<12,5,5,5>)
DVP(VPQ_16x4.4_SSE  , 2v4 , VecProductQuantizer_4_SSE128_unsigned<16>)
DVP(VPQ_16x4.4_AVX512 , 2W4 , VecProductQuantizer_4_AVX512_unsigned<16>)
DVP(VPQ_16x4.4_AVX2 , 2V4 , VecProductQuantizer_4_AVX256_unsigned<16>)

DVP(VPQ_8x8.8_AVX512, 2vc, VecProductQuantizer_88_AVX512_unsigned<8>)
DVP(VPQ_8x8_AVX512, 2v8, VecProductQuantizer_8_AVX512_unsigned<8>)

// Signed variants
DVP(VPQ_12x6.6.4_AVX512_s, 2vS , VecProductQuantizer_XYZ_AVX512<12,6,6,4>)
DVP(VPQ_12x6.5.5_AVX512_s, 2vA , VecProductQuantizer_XYZ_AVX512<12,6,5,5>)
DVP(VPQ_12x5.5.5_AVX512_s, 2vB , VecProductQuantizer_XYZ_AVX512<12,5,5,5>)
DVP(VPQ_16x4.4_SSE_s  , 2vF , VecProductQuantizer_4_SSE128<16>)
DVP(VPQ_16x4.4_AVX512_s , 2WF , VecProductQuantizer_4_AVX512<16>)
DVP(VPQ_16x4.4_AVX2_s , 2VF , VecProductQuantizer_4_AVX256<16>)

DVP(VPQ_8x8.8_AVX512_s, 2vC, VecProductQuantizer_88_AVX512<8>)
DVP(VPQ_8x8_AVX512_s, 2vH, VecProductQuantizer_8_AVX512<8>)

// with 16-bit distance variants
DVP(VPQ_16x4.4_NV16      , 2P4 , VecProductQuantizer_4444_AVX512_unsigned <16>)


// -----------------------------------------------------------------------------
// 128bit quantizers
DVP(VPQ_24x6.6.4_NV    , 4q6 , VecProductQuantizer_NoVec<24,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_24x6.5.5_NV    , 4qa , VecProductQuantizer_NoVec<24,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_24x5.5.5_NV    , 4qb , VecProductQuantizer_NoVec<24,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_32x4.4_NV      , 4q4 , VecProductQuantizer_NoVec<32,2,4,4,0,0,uint8_t,__m128i,__m128i>)
DVP(VPQ_32x4.4_NV_AVX2 , 4Q4 , VecProductQuantizer_NoVec<32,2,4,4,0,0,uint8_t,__m128i,__m256i>)

DVP(VPQ_24x6.6.4_NVT    , 4t6 , VecProductQuantizer_NoVecTable<24,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_24x6.5.5_NVT    , 4ta , VecProductQuantizer_NoVecTable<24,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_24x5.5.5_NVT    , 4tb , VecProductQuantizer_NoVecTable<24,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_32x4.4_NVT      , 4t4 , VecProductQuantizer_NoVecTable<32,2,4,4,0,0,uint8_t,__m128i,__m128i>)
DVP(VPQ_32x4.4_NVT_AVX2 , 4T4 , VecProductQuantizer_NoVecTable<32,2,4,4,0,0,uint8_t,__m128i,__m256i>)

DVP(VPQ_24x6.6.4_NQ    , 4s6 , VecProductQuantizer_NoVecNoQuant<24,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_24x6.6.4_NQ    , 4sa , VecProductQuantizer_NoVecNoQuant<24,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_24x6.6.4_NQ    , 4sb , VecProductQuantizer_NoVecNoQuant<24,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_32x4.4_NQ      , 4s4 , VecProductQuantizer_NoVecNoQuant<32,2,4,4,0,0,uint8_t,__m128i,__m128i>)
DVP(VPQ_32x4.4_NQ_AVX2 , 4S4 , VecProductQuantizer_NoVecNoQuant<32,2,4,4,0,0,uint8_t,__m128i,__m256i>)

DVP(VPQ_24x6.6.4_AVX512, 4v6 , VecProductQuantizer_XYZ_AVX512_unsigned<24,6,6,4>)
DVP(VPQ_24x6.5.5_AVX512, 4va , VecProductQuantizer_XYZ_AVX512_unsigned<24,6,5,5>)
DVP(VPQ_24x5.5.5_AVX512, 4vb , VecProductQuantizer_XYZ_AVX512_unsigned<24,5,5,5>)
DVP(VPQ_32x4.4_SSE  , 4v4 , VecProductQuantizer_4_SSE128_unsigned<32>)
DVP(VPQ_32x4.4_AVX512 , 4W4 , VecProductQuantizer_4_AVX512_unsigned<32>)
DVP(VPQ_32x4.4_AVX2 , 4V4 , VecProductQuantizer_4_AVX256_unsigned<32>)

DVP(VPQ_16x8.8_AVX512, 4vc, VecProductQuantizer_88_AVX512_unsigned<16>)
DVP(VPQ_16x8_AVX512, 4v8, VecProductQuantizer_8_AVX512_unsigned<16>)


// Signed variants
DVP(VPQ_24x6.6.4_AVX512_s, 4vS , VecProductQuantizer_XYZ_AVX512<24,6,6,4>)
DVP(VPQ_24x6.5.5_AVX512_s, 4vA , VecProductQuantizer_XYZ_AVX512<24,6,5,5>)
DVP(VPQ_24x5.5.5_AVX512_s, 4vB , VecProductQuantizer_XYZ_AVX512<24,5,5,5>)
DVP(VPQ_32x4.4_SSE_s  , 4vF , VecProductQuantizer_4_SSE128<32>)
DVP(VPQ_32x4.4_AVX512_s , 4WF , VecProductQuantizer_4_AVX512<32>)
DVP(VPQ_32x4.4_AVX2_s , 4VF , VecProductQuantizer_4_AVX256<32>)

DVP(VPQ_16x8.8_AVX512_s, 4vC, VecProductQuantizer_88_AVX512<16>)
DVP(VPQ_16x8_AVX512_s, 4vH, VecProductQuantizer_8_AVX512<16>)

// with 16-bit distance variants
DVP(VPQ_32x4.4_NV16      , 4P4 , VecProductQuantizer_4444_AVX512_unsigned <32>)

// -----------------------------------------------------------------------------
// 256 bit codes
DVP(VPQ_48x6.6.4_NQ    , 8s6 , VecProductQuantizer_NoVecNoQuant<48,3,6,6,4,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_48x6.6.4_NQ    , 8sa , VecProductQuantizer_NoVecNoQuant<48,3,6,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_48x6.6.4_NQ    , 8sb , VecProductQuantizer_NoVecNoQuant<48,3,5,5,5,0,uint16_t,__m512i,__m512i>)
DVP(VPQ_64x4.4_NQ      , 8s4 , VecProductQuantizer_NoVecNoQuant<64,2,4,4,0,0,uint8_t,__m128i,__m128i>)

DVP(VPQ_48x6.6.4_AVX512, 8v6 , VecProductQuantizer_XYZ_AVX512_unsigned<48,6,6,4>)
DVP(VPQ_48x6.5.5_AVX512, 8va , VecProductQuantizer_XYZ_AVX512_unsigned<48,6,5,5>)
DVP(VPQ_48x5.5.5_AVX512, 8vb , VecProductQuantizer_XYZ_AVX512_unsigned<48,5,5,5>)
DVP(VPQ_64x4.4_SSE  , 8v4 , VecProductQuantizer_4_SSE128_unsigned<64>)
DVP(VPQ_64x4.4_AVX512 , 8W4 , VecProductQuantizer_4_AVX512_unsigned<64>)
DVP(VPQ_64x4.4_AVX2 , 8V4 , VecProductQuantizer_4_AVX256_unsigned<64>)

DVP(VPQ_32x8.8_AVX512, 8vc, VecProductQuantizer_88_AVX512_unsigned<32>)
DVP(VPQ_32x8_AVX512, 8v8, VecProductQuantizer_8_AVX512_unsigned<32>)

// Signed variants
DVP(VPQ_48x6.6.4_AVX512_s, 8vS , VecProductQuantizer_XYZ_AVX512<48,6,6,4>)
DVP(VPQ_48x6.5.5_AVX512_s, 8vA , VecProductQuantizer_XYZ_AVX512<48,6,5,5>)
DVP(VPQ_48x5.5.5_AVX512_s, 8vB , VecProductQuantizer_XYZ_AVX512<48,5,5,5>)
DVP(VPQ_64x4.4_SSE_s  , 8vF , VecProductQuantizer_4_SSE128<64>)
DVP(VPQ_64x4.4_AVX512_s , 8WF , VecProductQuantizer_4_AVX512<64>)
DVP(VPQ_64x4.4_AVX2_s , 8VF , VecProductQuantizer_4_AVX256<64>)

DVP(VPQ_32x8.8_AVX512_s, 8vC, VecProductQuantizer_88_AVX512<32>)
DVP(VPQ_32x8_AVX512_s, 8vH, VecProductQuantizer_8_AVX512<32>)

// with 16-bit distance variants
DVP(VPQ_64x4.4_NV16      , 8P4 , VecProductQuantizer_4444_AVX512_unsigned <64>)


// Regular (Non vectorized for test and comparison)
DVP(VPQ_8x8_NQ  , 2s8 , VecProductQuantizer_NoVecNoQuant<8,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_4x8_NQ  , 1s8 , VecProductQuantizer_NoVecNoQuant<4,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_16x8_NQ , 4s8 , VecProductQuantizer_NoVecNoQuant<16,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_32x8_NQ , 8s8 , VecProductQuantizer_NoVecNoQuant<32,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)

DVP(VPQ_8x8_NV  , 2q8 , VecProductQuantizer_NoVec<8,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_4x8_NV  , 1q8 , VecProductQuantizer_NoVec<4,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_16x8_NV , 4q8 , VecProductQuantizer_NoVec<16,1,8,0,0,0,uint8_t,uint8_t,uint8_t>)

DVP(VPQ_8x7_NQ  , 2s7 , VecProductQuantizer_NoVecNoQuant<8,1,7,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_4x7_NQ  , 1s7 , VecProductQuantizer_NoVecNoQuant<4,1,7,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_16x7_NQ , 4s7 , VecProductQuantizer_NoVecNoQuant<16,1,7,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_8x7_NV  , 2q7 , VecProductQuantizer_NoVec<8,1,7,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_4x7_NV  , 1q7 , VecProductQuantizer_NoVec<4,1,7,0,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_16x7_NV , 4q7 , VecProductQuantizer_NoVec<16,1,7,0,0,0,uint8_t,uint8_t,uint8_t>)

DVP(VPQ_8x7.1_NQ  , 2sd , VecProductQuantizer_NoVecNoQuant<8,2,7,1,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_4x7.1_NQ  , 1sd , VecProductQuantizer_NoVecNoQuant<4,2,7,1,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_16x7.1_NQ , 4sd , VecProductQuantizer_NoVecNoQuant<16,2,7,1,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_8x7.1_NV  , 2qd , VecProductQuantizer_NoVec<8,2,7,1,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_4x7.1_NV  , 1qd , VecProductQuantizer_NoVec<4,2,7,1,0,0,uint8_t,uint8_t,uint8_t>)
DVP(VPQ_16x7.1_NV , 4qd , VecProductQuantizer_NoVec<16,2,7,1,0,0,uint8_t,uint8_t,uint8_t>)

// Additional
DVP(VPQ_30x6.6.4_AVX512, 5v6 , VecProductQuantizer_XYZ_AVX512_unsigned<30,6,6,4>)
DVP(VPQ_30x6.5.5_AVX512, 5va , VecProductQuantizer_XYZ_AVX512_unsigned<30,6,5,5>)
DVP(VPQ_30x5.5.5_AVX512, 5vb , VecProductQuantizer_XYZ_AVX512_unsigned<30,5,5,5>)

// Additional (signed, qadc, longer variants)
DVP(VPQ_16x4.4_AVX2_qadc , 2Vq , VecProductQuantizer_4_AVX256_qadc<16>)
DVP(VPQ_32x4.4_AVX2_qadc , 4Vq , VecProductQuantizer_4_AVX256_qadc<32>)
DVP(VPQ_64x4.4_AVX2_qadc , 8Vq , VecProductQuantizer_4_AVX256_qadc<64>)

// Variants based on Bolt Quantization (Quantization that minimize distance table distortion, learned at training time)
DVP(VPQ_16x4.4_NV_BOLT      , 2qr , VecProductQuantizer_BoltNoVec<16,__m128i,__m256i>)
DVP(VPQ_32x4.4_NV_BOLT      , 4qr , VecProductQuantizer_BoltNoVec<32,__m128i,__m256i>)
DVP(VPQ_64x4.4_NV_BOLT      , 8qr , VecProductQuantizer_BoltNoVec<64,__m128i,__m256i>)

DVP(VPQ_16x4.4_AVX2_BOLT16      , 2Vr , VecProductQuantizer_4_AVX256_Bolt16<16>)
DVP(VPQ_32x4.4_AVX2_BOLT16      , 4Vr , VecProductQuantizer_4_AVX256_Bolt16<32>)
DVP(VPQ_64x4.4_AVX2_BOLT16      , 8Vr , VecProductQuantizer_4_AVX256_Bolt16<64>)

DVP(VPQ_16x4.4_AVX2_BOLT8      , 2VR , VecProductQuantizer_4_AVX256_Bolt8<16>)
DVP(VPQ_32x4.4_AVX2_BOLT8      , 4VR , VecProductQuantizer_4_AVX256_Bolt8<32>)
DVP(VPQ_64x4.4_AVX2_BOLT8      , 8VR , VecProductQuantizer_4_AVX256_Bolt8<64>)



} // namespace faiss


#define EXPAND_VPQTEST(F) \
	F(VPQ_2q6) || \
	F(VPQ_2qa) || \
	F(VPQ_2qb) || \
	F(VPQ_2q4) || \
	F(VPQ_2Q4) || \
	F(VPQ_2s6) || \
	F(VPQ_2sa) || \
	F(VPQ_2sb) || \
	F(VPQ_2s4) || \
	F(VPQ_2S4) || \
	F(VPQ_2t6) || \
	F(VPQ_2ta) || \
	F(VPQ_2tb) || \
	F(VPQ_2t4) || \
	F(VPQ_2T4) || \
	F(VPQ_2v6) || \
	F(VPQ_2va) || \
	F(VPQ_2vb) || \
	F(VPQ_2v4) || \
	F(VPQ_2V4) || \
	F(VPQ_2W4) || \
	F(VPQ_4q6) || \
	F(VPQ_4qa) || \
	F(VPQ_4qb) || \
	F(VPQ_4q4) || \
	F(VPQ_4Q4) || \
	F(VPQ_4s6) || \
	F(VPQ_4sa) || \
	F(VPQ_4sb) || \
	F(VPQ_4s4) || \
	F(VPQ_4S4) || \
	F(VPQ_4t6) || \
	F(VPQ_4ta) || \
	F(VPQ_4tb) || \
	F(VPQ_4t4) || \
	F(VPQ_4T4) || \
	F(VPQ_4v6) || \
	F(VPQ_4va) || \
	F(VPQ_4vb) || \
	F(VPQ_4v4) || \
	F(VPQ_4V4) || \
	F(VPQ_4W4) || \
	F(VPQ_5v6) || \
	F(VPQ_5va) || \
	F(VPQ_5vb) || \
	F(VPQ_2s8) || \
	F(VPQ_2s7) || \
	F(VPQ_1s7) || \
	F(VPQ_4s7) || \
	F(VPQ_2q7) || \
	F(VPQ_1q7) || \
	F(VPQ_4q7) || \
	F(VPQ_2vc) || \
	F(VPQ_4vc) || \
	F(VPQ_2v8) || \
	F(VPQ_4v8)


#endif

