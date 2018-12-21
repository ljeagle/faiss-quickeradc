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

import faiss

#################################################################
# Small I/O functions
#################################################################


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#################################################################
#  Main program
#################################################################

print "load data"

xt = fvecs_read("sift1M/sift_learn.fvecs")
xb = fvecs_read("sift1M/sift_base.fvecs")
xq = fvecs_read("sift1M/sift_query.fvecs")

#xt = xt[:32000]
#xb = xb[:32000]
#xq = xb[:320]

nq, d = xq.shape

print "load GT"

gt = ivecs_read("sift1M/sift_groundtruth.ivecs")


print "Vector dimensionality" , d

def evaluate(s, index):

    nt = 1
    faiss.omp_set_num_threads(1)


    # index with 16 subquantizers, 8 bit each
    index.verbose = True
    index.search_type = faiss.IndexPQ.ST_PQ

    print(s)
    print "train"

    index.train(xt)

    print "add vectors to index"

    index.add(xb)

    print "bench"


    #print "PQ baseline",
    
    t0 = time.time()
    D, I = index.search(xq, 100)
    #print(I)
    t1 = time.time()

    recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)
    recall_at_100 = (I[:, :100] == gt[:, :1]).sum() / float(nq)
            
    print "\t %7.3f ms per query, R@1 %.4f   R@100  %.4f"   % (
        (t1 - t0) * 1000.0 / nq * nt, recall_at_1, recall_at_100)


evaluate("VPQ_2v4", faiss.IndexVPQ_2v4(d))
evaluate("VPQ_2V4", faiss.IndexVPQ_2V4(d))
#evaluate("VPQ_2W4", faiss.IndexVPQ_2W4(d))
evaluate("VPQ_2s4", faiss.IndexVPQ_2s4(d))

evaluate("VPQ_2va", faiss.IndexVPQ_2va(d))
evaluate("VPQ_2sa", faiss.IndexVPQ_2sa(d))

evaluate("VPQ_2v6", faiss.IndexVPQ_2v6(d))
evaluate("VPQ_2s6", faiss.IndexVPQ_2s6(d))

evaluate("VPQ_2vb", faiss.IndexVPQ_2vb(d))
evaluate("VPQ_2sb", faiss.IndexVPQ_2sb(d))

evaluate("VPQ_2s8", faiss.IndexVPQ_2s8(d))
evaluate("VPQ_2q8", faiss.IndexVPQ_2q8(d))

evaluate("VPQ_2s7", faiss.IndexVPQ_2s7(d))
evaluate("VPQ_2q7", faiss.IndexVPQ_2q7(d))



evaluate("VPQ_4v4", faiss.IndexVPQ_4v4(d))
evaluate("VPQ_4V4", faiss.IndexVPQ_4V4(d))
evaluate("VPQ_4W4", faiss.IndexVPQ_4W4(d))
evaluate("VPQ_4s4", faiss.IndexVPQ_4s4(d))

evaluate("VPQ_4va", faiss.IndexVPQ_4va(d))
evaluate("VPQ_4sa", faiss.IndexVPQ_4sa(d))

evaluate("VPQ_4v6", faiss.IndexVPQ_4v6(d))
evaluate("VPQ_4s6", faiss.IndexVPQ_4s6(d))

evaluate("VPQ_4vb", faiss.IndexVPQ_4vb(d))
evaluate("VPQ_4sb", faiss.IndexVPQ_4sb(d))

evaluate("VPQ_4s8", faiss.IndexVPQ_4s8(d))
evaluate("VPQ_4q8", faiss.IndexVPQ_4q8(d))

evaluate("VPQ_4s7", faiss.IndexVPQ_4s7(d))
evaluate("VPQ_4q7", faiss.IndexVPQ_4q7(d))


