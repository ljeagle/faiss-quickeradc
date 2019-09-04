# Copyright (c) 2018-present, Thomson Licensing, SAS.
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# Modifications related the introduction of Quicker ADC (Vectorized Product Quantization)
# are licensed under the Clear BSD license found in the LICENSE file in the root directory
# of this source tree.
#
# The rest of the source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree 


#!/usr/bin/env python2

import time
import numpy as np
import math
import faiss

#################################################################
# Small I/O functions
#################################################################

def round_sigfigs(num, sig_figs):
    """Round to specified number of sigfigs.

    >>> round_sigfigs(0, sig_figs=4)
    0
    >>> int(round_sigfigs(12345, sig_figs=2))
    12000
    >>> int(round_sigfigs(-12345, sig_figs=2))
    -12000
    >>> int(round_sigfigs(1, sig_figs=2))
    1
    >>> '{0:.3}'.format(round_sigfigs(3.1415, sig_figs=2))
    '3.1'
    >>> '{0:.3}'.format(round_sigfigs(-3.1415, sig_figs=2))
    '-3.1'
    >>> '{0:.5}'.format(round_sigfigs(0.00098765, sig_figs=2))
    '0.00099'
    >>> '{0:.6}'.format(round_sigfigs(0.00098765, sig_figs=3))
    '0.000988'
    """
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0  # Can't take the log of 0

def to_precision(x,p):
    #return  ("%."+str(p)+"f") % x
    return str(round_sigfigs(x,p))

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#################################################################
#  Main program
#################################################################


def evaluate(dataset, index_builder):


    xt = fvecs_read(dataset+"/learn.fvecs")
    xb = fvecs_read(dataset+"/base.fvecs")
    xq = fvecs_read(dataset+"/query.fvecs")
    #xt = xt[:32000]
    #xb = xb[:32000]
    #xq = xb[:320]
    nq, d = xq.shape

    #print xq.shape
    gt = ivecs_read(dataset+"/groundtruth.ivecs")

    index = index_builder(d)

    nt = 1
    faiss.omp_set_num_threads(1)

    index.verbose = False;
    index.search_type = faiss.IndexPQ.ST_PQ


    index.train(xt)
    index.add(xb)
    
    t0 = time.time()
    D, I = index.search(xq, 100)
    #print(I)
    t1 = time.time()

    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)
    recall_at_100 =  (I[:, :100] == gt[:, :1]).sum() / float(nq)
    time_pq = (t1-t0)
    return (recall_at_1,recall_at_100, (t1-t0)*1000.0 / nq)       
    #print "%s\t %7.3f ms per query, R@1 %.4f   R@100  %.4f"   % (s,
    #    (t1 - t0) * 1000.0 / nq * nt, recall_at_1, recall_at_100)

## q: quantized / V : vectorized quantized / s : scalar
## v4: u8 , vQ: s8, vq: qadc_s8, s4: float
## qu: u16 , qs: s16,
# index = {
#     "s8" : {
#         8 : faiss.IndexVPQ_2VQ,
#         32: faiss.IndexVPQ_8VQ,
#     },
#     "u8" : {
#         8 : faiss.IndexVPQ_2V4,
#         32: faiss.IndexVPQ_8V4,
#     },
#     "S8" : {
#         8 : faiss.IndexVPQ_2qS,
#     },
#     "U8" : {
#         8 : faiss.IndexVPQ_2q4,
#     },
#     "u16" : {
#         8 : faiss.IndexVPQ_2qu,
#         32: faiss.IndexVPQ_8qu,
#     },
#     "s16" : {
#         8 : faiss.IndexVPQ_2qs,
#         32: faiss.IndexVPQ_8qs,
#     },
#     "float" : {
#         8 : faiss.IndexVPQ_2s4,
#         32: faiss.IndexVPQ_8s4,
#     },
#     "qadc" : {
#         8 : faiss.IndexVPQ_2Vq,
#         32: faiss.IndexVPQ_8Vq,
#     },
#     "bolt" : {
#         8 : faiss.IndexVPQ_2b4,
#         32: faiss.IndexVPQ_8b4,
#     },
#     "boltV16" : {
#         8 : faiss.IndexVPQ_2B4,
#         32: faiss.IndexVPQ_8B4,
#     },
#     "boltV8" : {
#         8 : faiss.IndexVPQ_2C4,
#         32: faiss.IndexVPQ_8C4,
#     }
# }


#for quant in index:
#for quant in ["boltV16","boltV8","u8","u16", "s8", "s16", "float", "qadc"]:
#    print quant, 
#    for dataset in ["sift1M"]: # GIST1M
#        for length in [8,32]:
#            (r1,r100,t) = evaluate(dataset,index[quant][length])
#            print "("+str(length)+")"+str(r1)+"/"+str(r100)+" & ",
#    print ""


listEval = [
    #{"title": "mx{4,4} SSE", "i": "pshufb", "q": "int8", "const": {8: faiss.IndexVPQ_2vF, 16: faiss.IndexVPQ_4vF , 32 : faiss.IndexVPQ_8vF }},
    #{"title": "mx{4,4} SSE", "i": "pshufb", "q": "uint8", "const": {8: faiss.IndexVPQ_2v4, 16: faiss.IndexVPQ_4v4 , 32 : faiss.IndexVPQ_8v4 }},
    #{"title": "mx{4,4} AVX", "i": "pshufb", "q": "int8", "const": {8: faiss.IndexVPQ_2VF, 16: faiss.IndexVPQ_4VF , 32 : faiss.IndexVPQ_8VF }},
    #{"title": "mx{4,4} AVX", "i": "pshufb", "q": "uint8", "const": {8: faiss.IndexVPQ_2V4, 16: faiss.IndexVPQ_4V4 , 32 : faiss.IndexVPQ_8V4 }},
    #{"title": "mx{4,4} AVX512 BW", "i": "pshufb", "q": "int8", "const": {8: faiss.IndexVPQ_2WF, 16: faiss.IndexVPQ_4WF , 32 : faiss.IndexVPQ_8WF }},
    #{"title": "mx{4,4} AVX512 BW", "i": "pshufb", "q": "uint8", "const": {8: faiss.IndexVPQ_2W4, 16: faiss.IndexVPQ_4W4 , 32 : faiss.IndexVPQ_8W4 }},
    #{"title": "mx{4,4,4,4} AVX512", "i": "vpermw", "q": "uint16", "const": {8: faiss.IndexVPQ_2P4, 16: faiss.IndexVPQ_4P4 , 32 : faiss.IndexVPQ_8P4 }},
    #{"title": "mx{4,4} QADC", "i": "pshufb", "q": "int8", "const": {8: faiss.IndexVPQ_2Vq, 16: faiss.IndexVPQ_4Vq , 32 : faiss.IndexVPQ_8Vq }},
    #{"title": "mx{4,4} Bolt16", "i": "pshufb", "q": "uint16", "const": {8: faiss.IndexVPQ_2Vr, 16: faiss.IndexVPQ_4Vr , 32 : faiss.IndexVPQ_8Vr }},
    #{"title": "mx{4,4} Bolt8", "i": "pshufb", "q": "uint8", "const": {8: faiss.IndexVPQ_2VR, 16: faiss.IndexVPQ_4VR , 32 : faiss.IndexVPQ_8VR }},
    #{"title": "mx{4,4}", "i": "", "q": "float", "const": {8: faiss.IndexVPQ_2s4, 16: faiss.IndexVPQ_4s4 , 32 : faiss.IndexVPQ_8s4 }},

    #{"title": "mx{6,6,4} AVX512 BW", "i": "vpermi2w", "q": "int16", "const": {8: faiss.IndexVPQ_2vS, 16: faiss.IndexVPQ_4vS , 32 : faiss.IndexVPQ_8vS }},
    #{"title": "mx{6,6,4} AVX512 BW", "i": "vpermi2w", "q": "uint16", "const": {8: faiss.IndexVPQ_2v6, 16: faiss.IndexVPQ_4v6 , 32 : faiss.IndexVPQ_8v6 }},
    #{"title": "mx{6,6,4}", "i": "", "q": "float", "const": {8: faiss.IndexVPQ_2s6, 16: faiss.IndexVPQ_4s6 , 32 : faiss.IndexVPQ_8s6 }},

    #{"title": "mx{6,5,5} AVX512 BW", "i": "vpermi2w", "q": "int16", "const": {8: faiss.IndexVPQ_2vA, 16: faiss.IndexVPQ_4vA , 32 : faiss.IndexVPQ_8vA }},
    #{"title": "mx{6,5,5} AVX512 BW", "i": "vpermi2w", "q": "uint16", "const": {8: faiss.IndexVPQ_2va, 16: faiss.IndexVPQ_4va , 32 : faiss.IndexVPQ_8va }},
    #{"title": "mx{6,5,5}", "i": "", "q": "float", "const": {8: faiss.IndexVPQ_2sa, 16: faiss.IndexVPQ_4sa , 32 : faiss.IndexVPQ_8sa }},

    #{"title": "mx{5,5,5} AVX512 BW", "i": "vpermw", "q": "int16", "const": {8: faiss.IndexVPQ_2vB, 16: faiss.IndexVPQ_4vB , 32 : faiss.IndexVPQ_8vB }},
    #{"title": "mx{5,5,5} AVX512 BW", "i": "vpermw", "q": "uint16", "const": {8: faiss.IndexVPQ_2vb, 16: faiss.IndexVPQ_4vb , 32 : faiss.IndexVPQ_8vb }},
    #{"title": "mx{5,5,5}", "i": "", "q": "float", "const": {8: faiss.IndexVPQ_2sb, 16: faiss.IndexVPQ_4sb , 32 : faiss.IndexVPQ_8sb }},

    #{"title": "mx{8,8} AVX512 BW", "i": "vpermi2w", "q": "int16", "const": {8: faiss.IndexVPQ_2vC, 16: faiss.IndexVPQ_4vC , 32 : faiss.IndexVPQ_8vC }},
    #{"title": "mx{8,8} AVX512 BW", "i": "vpermi2w", "q": "uint16", "const": {8: faiss.IndexVPQ_2vc, 16: faiss.IndexVPQ_4vc , 32 : faiss.IndexVPQ_8vc }},
    #{"title": "mx{8} AVX512 VBMI", "i": "vpermi2b", "q": "int8", "const": {8: faiss.IndexVPQ_2v8, 16: faiss.IndexVPQ_4v8 , 32 : faiss.IndexVPQ_8v8 }},
    {"title": "mx{8} AVX512 VBMI", "i": "vpermi2b", "q": "uint8", "const": {8: faiss.IndexVPQ_2vH, 16: faiss.IndexVPQ_4vH , 32 : faiss.IndexVPQ_8vH }},
    #{"title": "mx{8}", "i": "", "q": "float", "const": {8: faiss.IndexVPQ_2s8, 16: faiss.IndexVPQ_4s8 , 32 : faiss.IndexVPQ_8s8 }},
]

#for quant in index:
for quant in listEval:
    print quant["title"], "& \\simd{", quant["i"] ,"} & \\dtype{", quant["q"],'}',  
    for dataset in ["sift1M", "deep1M"]: # GIST1M
        for length in [8,16,32]:
            (r1,r100,t) =  evaluate(dataset,quant["const"][length])
            print "\\result{"+to_precision(r1*100.0,3)+"}{"+to_precision(r100*100.0,3)+"}{"+to_precision(t,3) +"} & ",
    print ""

#]