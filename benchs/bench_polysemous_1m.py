# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

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

def evaluate(dataset, poly, width):

    xt = fvecs_read(dataset+"/learn.fvecs")
    xb = fvecs_read(dataset+"/base.fvecs")
    xq = fvecs_read(dataset+"/query.fvecs")
    #xt = xt[:32000]
    #xb = xb[:32000]
    #xq = xb[:320]
    nq, d = xq.shape

    #print xq.shape
    gt = ivecs_read(dataset+"/groundtruth.ivecs")



    # index with 16 subquantizers, 8 bit each
    index = faiss.IndexPQ(d, width, 8)

    if(poly):
        index.search_type = faiss.IndexPQ.ST_polysemous
        index.do_polysemous_training = True
    else:
        index.do_polysemous_training = False
        index.search_type = faiss.IndexPQ.ST_PQ
    index.verbose=False

    
    index.train(xt)


    index.add(xb)

    nt = 1
    faiss.omp_set_num_threads(1)




    if(poly and width == 8):
        htList =[0,17,19,21,23,25]
    elif(poly and width == 16):
        htList =[0,43,45,47,49,51,53,55,57,59]
    elif(poly and width == 32):
        htList =[0,78,81,84,87,90,93,96,99,102,105,108,111,114,117]
    else:
        htList = [0]

    for ht in htList:
        index.polysemous_ht = ht
        t0 = time.time()
        D, I = index.search(xq, 100)
        t1 = time.time()
        recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)
        recall_at_100 = (I[:, :100] == gt[:, :1]).sum() / float(nq)
        if(poly):
             print "Poly ",
        else:
             print "PQ ",
        print width,"x8 tau=", ht," ",      
        print "\t %7.3f ms per query, R@1 %.4f   R@100  %.4f"   % (
                (t1 - t0) * 1000.0 / nq * nt, recall_at_1, recall_at_100)

evaluate("sift1M",True,8)
evaluate("sift1M",True,16)
evaluate("sift1M",True,32)
evaluate("deep1M",True,8)
evaluate("deep1M",True,16)
evaluate("deep1M",True,32)
evaluate("sift1M",False,8)
evaluate("sift1M",False,16)
evaluate("sift1M",False,32)
evaluate("deep1M",False,8)
evaluate("deep1M",False,16)
evaluate("deep1M",False,32)
