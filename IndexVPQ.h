/**
 * Copyright (c) 2018-present, Thomson Licensing, SAS.
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * Modifications related the introduction of Quicker ADC (Vectorized Product Quantization)
 * are licensed under the Clear BSD license found in the LICENSE file in the root directory
 * of this source tree.
 *
 * The rest of the source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef FAISS_INDEX_VPQ_H
#define FAISS_INDEX_VPQ_H

#include <stdint.h>

#include <vector>

#include "Index.h"
#include "VecProductQuantizer.h"
#include <boost/align/aligned_allocator.hpp>


namespace faiss {

/// statistics are robust to internal threading, but not if
/// IndexPQ::search is called by multiple threads
struct IndexVPQStats {
    size_t nq;       // nb of queries run
    size_t ncode;    // nb of codes visited

    IndexVPQStats () {reset (); }
    void reset ();
};

extern IndexVPQStats indexVPQ_stats;

struct AbstractIndexVPQ {
	int initial_scan_estim_param;

	 virtual ~AbstractIndexVPQ() = default;
};


/** Index based on a product quantizer. Stored vectors are
 * approximated by PQ codes. */
template<class T_VPQ>
struct IndexVPQ: Index, AbstractIndexVPQ {

	typedef T_VPQ VPQ_t;

    /// The product quantizer used to encode the vectors
    T_VPQ pq;

	typedef typename T_VPQ::group groupt;

    /// Codes. Size ntotal * pq.code_size
    std::vector<groupt, boost::alignment::aligned_allocator<groupt, 64>> codes;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexVPQ (int d,                    ///< dimensionality of the input vectors
             MetricType metric = METRIC_L2):
            	 Index(d, metric), pq(d)
             {
        is_trained = false;
        search_type = ST_PQ;
        this->initial_scan_estim_param=4;
    }

    IndexVPQ () {
        metric_type = METRIC_L2;
        is_trained = false;
        search_type = ST_PQ;
    }

    void train(idx_t n, const float* x) override {
    	pq.train(n, x);
        is_trained = true;
    }

    void add(idx_t n, const float* x) override {
        FAISS_THROW_IF_NOT (is_trained);
        codes.resize (pq.nb_groups(n + ntotal));
        pq.encode_multiple(x, codes.data(), ntotal, n);
        ntotal += n;
    }

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override {

    	FAISS_THROW_IF_NOT (is_trained);
        if (search_type == ST_PQ) {  // Simple PQ search

            if (metric_type == METRIC_L2) {
                float_maxheap_array_t res = {
                    size_t(n), size_t(k), labels, distances };
                pq.search (x, n, codes.data(), ntotal, &res, true, this->initial_scan_estim_param);
            } else {
                float_minheap_array_t res = {
                    size_t(n), size_t(k), labels, distances };
                pq.search_ip (x, n, codes.data(), ntotal, &res, true);
            }
            indexVPQ_stats.nq += n;
            indexVPQ_stats.ncode += n * ntotal;

        } else if(search_type == ST_SDC){ // code-to-code distances
        	 std::vector<groupt, boost::alignment::aligned_allocator<groupt, 64>> q_codes_v;
        	 q_codes_v.resize(pq.nb_groups(n));
             groupt * q_codes = q_codes_v.data();

            pq.encode_multiple(x, q_codes, 0, n);

            float_maxheap_array_t res = {
                    size_t(n),  size_t(k), labels, distances};

            pq.search_sdc (q_codes, n, codes.data(), ntotal, &res, true);

            indexVPQ_stats.nq += n;
            indexVPQ_stats.ncode += n * ntotal;
        }
    }

    void reset() override {
        codes.clear();
        ntotal = 0;
    }

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override {
        FAISS_THROW_IF_NOT (ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
        for (idx_t i = 0; i < ni; i++) {
            pq.decode (codes.data(), recons + i * d,i);
        }
    }

    void reconstruct(idx_t key, float* recons) const override {
        FAISS_THROW_IF_NOT (key >= 0 && key < ntotal);
        pq.decode (codes.data(), recons, key);
    }

    /// how to perform the search in search_core
    enum Search_type_t {
        ST_PQ,             ///< asymmetric product quantizer (default)
        ST_SDC,            ///< symmetric product quantizer (SDC)
    };

    Search_type_t search_type;


};

template <class T>
inline std::string fourcc_vpq(const IndexVPQ<T>* n){return "j"+cc_vpq((T*)NULL);}

} // namespace faiss



#endif
