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

#ifndef FAISS_INDEX_IVFVPQ_H
#define FAISS_INDEX_IVFVPQ_H

//#define PARALLEL_IVFVPQ 1


#include <vector>
#include <omp.h>

#include "IndexIVF.h"
#include "IndexVPQ.h"
#include "IndexPQ.h"
// include IVFPQ.h to access stat object
#include "IndexIVFPQ.h"




#include <boost/align/aligned_allocator.hpp>


namespace faiss {

static uint64_t get_cycles () {
#ifdef  __x86_64__
    uint32_t high, low;
    asm volatile("rdtsc \n\t"
                 : "=a" (low),
                   "=d" (high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

#define TIC t0 = get_cycles()
#define TOC get_cycles () - t0

struct AbstractIndexIVFVPQ {
    int initial_scan_estim_param = 4;

	 virtual ~AbstractIndexIVFVPQ() = default;
};

struct IVFVPQSearchParameters: IVFSearchParameters {
    size_t scan_table_threshold;   ///< use table computation or on-the-fly?
    int initial_scan_estim_param;   ///< When building a top-k, how many vectors as a multiple of k should be scanned to estimate distance quantizers
    size_t scan_table_threshold_simd;
	~IVFVPQSearchParameters () {}
};

/** Inverted file with Product Quantizer encoding. Each residual
 * vector is encoded as a product quantizer code.
 */
template<class T_VPQ>
struct IndexIVFVPQ: IndexIVF, AbstractIndexIVFVPQ {
	typedef T_VPQ VPQ_t;
    bool by_residual;              ///< Encode residual or plain vector?
    int use_precomputed_table;     ///< if by_residual, build precompute tables

#ifdef ECLIPSE
	typedef VecProductQuantizer_4_AVX256<16> T_VPQ;
#endif
    T_VPQ pq;           ///< produces the codes

    // search-time parameters
    size_t scan_table_threshold=0;   ///< use table computation or on-the-fly?
    size_t scan_table_threshold_simd=16;   ///< use table computation or simd?


    /// if use_precompute_table
    /// size nlist * pq.M * pq.ksub
    std::vector <float> precomputed_table;


	typedef typename T_VPQ::group groupt;

	using vec_vec_group = std::vector<std::vector<groupt, boost::alignment::aligned_allocator<groupt, 64>>>;
    using vec_size = std::vector<size_t>;
    using vec_vec_ids = std::vector<std::vector<long>>;

    /// Inverted lists of groups of codes per partition, of total count per partition
    vec_vec_group group_codes;
    vec_size count_codes;

    // Note that  std::vector < std::vector<uint8_t> > codes;  is left unused (empty lists) in our implementation due to different memory layout.

    IndexIVFVPQ (Index * quantizer, size_t d, size_t nlist) :
    	    IndexIVF (quantizer, d, nlist, 0, METRIC_L2),
    	    pq (d)
			//group_codes(),
			//count_codes()
    	{
    	    code_size = 0;
    	    invlists->code_size = 0;
    	    is_trained = false;
    	    by_residual = true;
    	    use_precomputed_table = 0;
    	    //scan_table_threshold_simd = 16; Must be initialized somewhere else (for deserialization)
    	    //scan_table_threshold = 0;
    	    group_codes.resize (nlist);
    	    count_codes.resize (nlist);
    	    maintain_direct_map=false; // Force direct map to false
    	    this->initial_scan_estim_param=4; // Default value, overwritten by auto-tune
    	}

    void add_with_ids_internal (idx_t n, const float * x, const long *xids) {
         FAISS_THROW_IF_NOT (is_trained);
         double t0 = getmillisecs ();
         const long * idx;
         ScopeDeleter<long> del_idx;


         long * idx0 = new long [n];
         del_idx.set (idx0);
         quantizer->assign (n, x, idx0);
         idx = idx0;

         double t1 = getmillisecs ();
         std::vector<groupt, boost::alignment::aligned_allocator<groupt, 64>> xcodes;
         std::vector<float> residuals;
         xcodes.resize(pq.nb_groups(n));

         const float * to_encode;

         if (by_residual) {
         	residuals.resize(n*d);
             for (size_t i = 0; i < n; i++) {
                 if (idx[i] < 0)
                     memset (residuals.data() + i * d, 0, sizeof(*residuals.data()) * d);
                 else
                     quantizer->compute_residual(x + i * d, residuals.data() + i * d, idx[i]);
             }
             to_encode = residuals.data();
         } else {
             to_encode = x;
         }
         pq.encode_multiple(to_encode, xcodes.data(),0,n);

         double t2 = getmillisecs ();
         size_t n_ignore = 0;
         for (size_t i = 0; i < n; i++) {
             idx_t key = idx[i];
             if (key < 0) {
                 n_ignore ++;
                 continue;
             }
             idx_t id = xids ? xids[i] : ntotal + i;
             size_t offset = invlists->add_entry (key, id, NULL /* CODE IS STORED IN GROUP_CODES NOT IN INVLISTS */);
             pq.append_codes(group_codes[key],&count_codes[key],xcodes.data(),i,1);
             if (maintain_direct_map){
             	direct_map.push_back (key << 32 | offset);
             }
         }


         double t3 = getmillisecs ();
         if(verbose) {
             char comment[100] = {0};
             if (n_ignore > 0)
                 snprintf (comment, 100, "(%ld vectors ignored)", n_ignore);
             printf(" add_core times: %.3f %.3f %.3f %s\n",
                    t1 - t0, t2 - t1, t3 - t2, comment);
         }
         ntotal += n;
     }

    void add_with_ids(idx_t n, const float *x,  const long * xids= nullptr) {
            idx_t bs = 262144;
            if (n > bs) {
                for (idx_t i0 = 0; i0 < n; i0 += bs) {
                    idx_t i1 = std::min(i0 + bs, n);
                    if (verbose) {
                        printf("IndexIVFPQ::add_core_o: adding %ld:%ld / %ld\n",
                               i0, i1, n);
                    }
                    add_with_ids_internal(i1 - i0, x + i0 * d,
                                xids ? xids + i0 : nullptr);
                }
                return;
            }

    }

    /// trains the product quantizer
    void train_residual(idx_t n, const float* x) override {
    	const float * x_in = x;

    	    x = fvecs_maybe_subsample (
    	         d, (size_t*)&n, pq.cp.max_points_per_centroid * pq.ksub_total/pq.M,
    	         x, verbose, pq.cp.seed);

    	    ScopeDeleter<float> del_x (x_in == x ? nullptr : x);

    	    const float *trainset;
    	    ScopeDeleter<float> del_residuals;
    	    if (by_residual) {
    	        if(verbose) printf("computing residuals\n");
    	        idx_t * assign = new idx_t [n]; // assignement to coarse centroids
    	        ScopeDeleter<idx_t> del (assign);
    	        quantizer->assign (n, x, assign);
    	        float *residuals = new float [n * d];
    	        del_residuals.set (residuals);
    	        for (idx_t i = 0; i < n; i++)
    	           quantizer->compute_residual (x + i * d, residuals+i*d, assign[i]);

    	        trainset = residuals;
    	    } else {
    	        trainset = x;
    	    }
    	    if (verbose)
    	        printf ("training %zdx%zd product quantizer on %ld vectors in %dD\n",
    	                pq.M, pq.ksub_total, n, d);
    	    pq.verbose = verbose;
    	    pq.train (n, trainset);

    	    if (by_residual) {
    	        precompute_table ();
    	    }
    }


    void reconstruct_from_offset (long list_no, long offset,
                                  float* recons) const override {
        const groupt * code = group_codes[list_no].data();

        if (by_residual) {
          std::vector<float> centroid(d);
          quantizer->reconstruct (list_no, centroid.data());

          pq.decode (code, recons,offset);
          for (int i = 0; i < d; ++i) {
            recons[i] += centroid[i];
          }
        } else {
          pq.decode (code, recons,offset);
        }
    }


    void merge_from (IndexIVF &other, idx_t add_id) override {
    	FAISS_THROW_MSG("Not implemented");
    }





    /** Precomputed tables for residuals
     *
     * During IVFPQ search with by_residual, we compute
     *
     *     d = || x - y_C - y_R ||^2
     *
     * where x is the query vector, y_C the coarse centroid, y_R the
     * refined PQ centroid. The expression can be decomposed as:
     *
     *    d = || x - y_C ||^2 + || y_R ||^2 + 2 * (y_C|y_R) - 2 * (x|y_R)
     *        ---------------   ---------------------------       -------
     *             term 1                 term 2                   term 3
     *
     * When using multiprobe, we use the following decomposition:
     * - term 1 is the distance to the coarse centroid, that is computed
     *   during the 1st stage search.
     * - term 2 can be precomputed, as it does not involve x. However,
     *   because of the PQ, it needs nlist * M * ksub storage. This is why
     *   use_precomputed_table is off by default
     * - term 3 is the classical non-residual distance table.
     *
     * Since y_R defined by a product quantizer, it is split across
     * subvectors and stored separately for each subvector. If the coarse
     * quantizer is a MultiIndexQuantizer then the table can be stored
     * more compactly.
     *
     * At search time, the tables for term 2 and term 3 are added up. This
     * is faster when the length of the lists is > ksub * M.
     */
    void precompute_table () {
    	//FIXME update precompute
    	if (use_precomputed_table == 0) { // then choose the type of table
    	        if (quantizer->metric_type == METRIC_INNER_PRODUCT) {
    	            fprintf(stderr, "IndexIVFPQ::precompute_table: WARN precomputed "
    	                    "tables not needed for inner product quantizers\n");
    	            return;
    	        }
    	        const MultiIndexQuantizer *miq =
    	            dynamic_cast<const MultiIndexQuantizer *> (quantizer);
    	        if (miq && pq.M % miq->pq.M == 0)
    	            use_precomputed_table = 2;
    	        else
    	            use_precomputed_table = 1;
    	    } // otherwise assume user has set appropriate flag on input

    	    if (verbose) {
    	        printf ("precomputing IVFPQ tables type %d\n",
    	                use_precomputed_table);
    	    }

    	    // squared norms of the PQ centroids
    	    std::vector<float> r_norms (pq.ksub_total, NAN);
    	    for (int m = 0; m < pq.M; m++)
    	        for (int j = 0; j < pq.ksub[m]; j++)
    	            r_norms [pq.ksub_offset[m] + j] =
    	                fvec_norm_L2sqr (pq.get_centroids (m, j), pq.dsub[m]);

    	    if (use_precomputed_table == 1) {

    	        precomputed_table.resize (nlist * pq.ksub_total);
    	        std::vector<float> centroid (d);

    	        for (size_t i = 0; i < nlist; i++) {
    	            quantizer->reconstruct (i, centroid.data());

    	            float *tab = &precomputed_table[i * pq.ksub_total];
    	            pq.compute_inner_prod_table (centroid.data(), tab);
    	            fvec_madd (pq.ksub_total, r_norms.data(), 2.0, tab, tab);
    	        }
    	    } else if (use_precomputed_table == 2) {
    	        const MultiIndexQuantizer *miq =
    	           dynamic_cast<const MultiIndexQuantizer *> (quantizer);
    	        FAISS_THROW_IF_NOT (miq);
    	        const ProductQuantizer &cpq = miq->pq;
    	        FAISS_THROW_IF_NOT (pq.M % cpq.M == 0);
    	        FAISS_THROW_IF_NOT (pq.ksub_total % cpq.M == 0);

    	        precomputed_table.resize(cpq.ksub * pq.ksub_total);

    	        // reorder PQ centroid table
    	        std::vector<float> centroids (d * cpq.ksub, NAN);

    	        for (int m = 0; m < cpq.M; m++) {
    	            for (size_t i = 0; i < cpq.ksub; i++) {
    	                memcpy (centroids.data() + i * d + m * cpq.dsub,
    	                        cpq.get_centroids (m, i),
    	                        sizeof (*centroids.data()) * cpq.dsub);
    	            }
    	        }

    	        pq.compute_inner_prod_tables (cpq.ksub, centroids.data (),
    	                                      precomputed_table.data ());

    	        for (size_t i = 0; i < cpq.ksub; i++) {
    	            float *tab = &precomputed_table[i * pq.ksub_total];
    	            fvec_madd (pq.ksub_total, r_norms.data(), 2.0, tab, tab);
    	        }

    	    }

    }

    IndexIVFVPQ () {
        // initialize some runtime values
        use_precomputed_table = 0;
        scan_table_threshold = 0;
        by_residual=0;
    }


    /** QueryTables manages the various ways of searching an
     * IndexIVFPQ. The code contains a lot of branches, depending on:
     * - metric_type: are we computing L2 or Inner product similarity?
     * - by_residual: do we encode raw vectors or residuals?
     * - use_precomputed_table: are x_R|x_C tables precomputed?
     */
    /*****************************************************
     * Scaning the codes.
     * The scanning functions call their favorite precompute_*
     * function to precompute the tables they need.
     *****************************************************/
    template <typename IDType>
    struct QueryTables {

        /*****************************************************
         * General data from the IVFPQ
         *****************************************************/

        const IndexIVFVPQ<T_VPQ> & ivfpq;

        // copied from IndexIVFPQ for easier access
        int d;
        const T_VPQ & pq;
        MetricType metric_type;
        bool by_residual;
        int use_precomputed_table;

        // pre-allocated data buffers
        float * sim_table, * sim_table_2;
        float * residual_vec, *decoded_vec;

        // single data buffer
        std::vector<float> mem;

        // for table pointers
        std::vector<const float *> sim_table_ptrs;

        const groupt * __restrict list_codes;
        const IDType * list_ids;
        size_t list_size;
        size_t already_scanned;
        size_t codes_needed_to_build_quantizer;
        size_t n_simd_eval;

        explicit QueryTables (const  IndexIVFVPQ<T_VPQ> & ivfpq, int k, const IVFSearchParameters *params):
            ivfpq(ivfpq),
            d(ivfpq.d),
            pq (ivfpq.pq),
            metric_type (ivfpq.metric_type),
            by_residual (ivfpq.by_residual),
            use_precomputed_table (ivfpq.use_precomputed_table),
			list_codes(NULL),
			list_ids(NULL),
			list_size(0),
			already_scanned(0),
			codes_needed_to_build_quantizer(ivfpq.initial_scan_estim_param*k)
        {
            mem.resize (pq.ksub_total * 2 + d *2);
            sim_table = mem.data();
            sim_table_2 = sim_table + pq.ksub_total;
            residual_vec = sim_table_2 + pq.ksub_total;
            decoded_vec = residual_vec + d;

            init_list_cycles = 0;
            sim_table_ptrs.resize(pq.M);

            if (auto ivfvpq_params =
                      dynamic_cast<const IVFVPQSearchParameters *>(params)) {
                      codes_needed_to_build_quantizer = k*ivfvpq_params->initial_scan_estim_param;
                  }

            key=0;
            coarse_dis=0.0;
            qi=NULL;
            n_simd_eval = 0;

        }

        /*****************************************************
         * What we do when query is known
         *****************************************************/

        // field specific to query
        const float * qi;

        // query-specific intialization
        void init_query (const float * qi) {
            this->qi = qi;
			this->already_scanned=0;
            if (metric_type == METRIC_INNER_PRODUCT)
                init_query_IP ();
            else
                init_query_L2 ();
        }

        void init_query_IP () {
            // precompute some tables specific to the query qi
            pq.compute_inner_prod_table (qi, sim_table);
            // we compute negated inner products for use with the maxheap
            for (int i = 0; i < pq.ksub_total; i++) {
                sim_table[i] = - sim_table[i];
            }
        }

        void init_query_L2 () {
            if (!by_residual) {
                pq.compute_distance_table (qi, sim_table);
            } else if (use_precomputed_table) {
                pq.compute_inner_prod_table (qi, sim_table_2);
            }
        }

        /*****************************************************
         * When inverted list is known: prepare computations
         *****************************************************/

        // fields specific to list
        Index::idx_t key;
        float coarse_dis;
        //std::vector<uint8_t> q_code;

        uint64_t init_list_cycles;

        /// once we know the query and the centroid, we can prepare the
        /// sim_table that will be used for accumulation
        /// and dis0, the initial value
        float precompute_list_tables () {
            float dis0 = 0;
            uint64_t t0; TIC;
            if (by_residual) {
                if (metric_type == METRIC_INNER_PRODUCT)
                    dis0 = precompute_list_tables_IP ();
                else
                    dis0 = precompute_list_tables_L2 ();
            }
            init_list_cycles += TOC;
            return dis0;
         }

        float precompute_list_table_pointers () {
            float dis0 = 0;
            uint64_t t0; TIC;
            if (by_residual) {
                if (metric_type == METRIC_INNER_PRODUCT)
                  FAISS_THROW_MSG ("not implemented");
                else
                  dis0 = precompute_list_table_pointers_L2 ();
            }
            init_list_cycles += TOC;
            return dis0;
         }

        /*****************************************************
         * compute tables for inner prod
         *****************************************************/

        float precompute_list_tables_IP ()
        {
            // prepare the sim_table that will be used for accumulation
            // and dis0, the initial value
            ivfpq.quantizer->reconstruct (key, decoded_vec);
            // decoded_vec = centroid
            float dis0 = -fvec_inner_product (qi, decoded_vec, d);

            return dis0;
        }


        /*****************************************************
         * compute tables for L2 distance
         *****************************************************/

        float precompute_list_tables_L2 ()
        {
            float dis0 = 0;

            if (use_precomputed_table == 0) {
                ivfpq.quantizer->compute_residual (qi, residual_vec, key);
                pq.compute_distance_table (residual_vec, sim_table);
            } else if (use_precomputed_table == 1) {
                dis0 = coarse_dis;

                fvec_madd (pq.ksub_total,
                           &ivfpq.precomputed_table [key * pq.ksub_total],
                           -2.0, sim_table_2,
                           sim_table);
            } else if (use_precomputed_table == 2) {
                dis0 = coarse_dis;

                const MultiIndexQuantizer *miq =
                    dynamic_cast<const MultiIndexQuantizer *> (ivfpq.quantizer);
                FAISS_THROW_IF_NOT (miq);
                const ProductQuantizer &cpq = miq->pq;
               // int Mf = pq.M / cpq.M;

                const float *qtab = sim_table_2; // query-specific table
                float *ltab = sim_table; // (output) list-specific table

                long k = key;
                for (int cm = 0; cm < cpq.M; cm++) {
                    // compute PQ index
                    int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                    k >>= cpq.nbits;

                    // get corresponding table
                    const float *pc = &ivfpq.precomputed_table[ki*pq.ksub_total + cm*pq.ksub_total/cpq.M];
                    //   [(ki * pq.M + cm * Mf) * pq.ksub];


					// sum up with query-specific table
					fvec_madd (pq.ksub_total / cpq.M,
							   pc,
							   -2.0, qtab,
							   ltab);
					ltab += pq.ksub_total / cpq.M;
					qtab += pq.ksub_total / cpq.M;


                }
            }

            return dis0;
        }

        float precompute_list_table_pointers_L2 ()
        {
            float dis0 = 0;

            if (use_precomputed_table == 1) {
                dis0 = coarse_dis;

                const float * s = &ivfpq.precomputed_table [key * pq.ksub_total];
                for (int m = 0; m < pq.M; m++) {
                    sim_table_ptrs [m] = &s[pq.ksub_offset[m]];
                }
            } else if (use_precomputed_table == 2) {
                dis0 = coarse_dis;

                const MultiIndexQuantizer *miq =
                    dynamic_cast<const MultiIndexQuantizer *> (ivfpq.quantizer);
                FAISS_THROW_IF_NOT (miq);
                const ProductQuantizer &cpq = miq->pq;
                int Mf = pq.M / cpq.M;

                long k = key;
                int m0 = 0;
                for (int cm = 0; cm < cpq.M; cm++) {
                    int ki = k & ((uint64_t(1) << cpq.nbits) - 1);
                    k >>= cpq.nbits;

                    const float *pc = &ivfpq.precomputed_table[ki*pq.ksub_total + cm*pq.ksub_total/cpq.M];
                    //const float *pc = &ivfpq.precomputed_table
                    //    [(ki * pq.M + cm * Mf) * pq.ksub];

                    for (int m = m0; m < m0 + Mf; m++) {
                        sim_table_ptrs [m] = &pc[pq.ksub_offset[m]];
                    }
                    m0 += Mf;
                }
            } else {
              FAISS_THROW_MSG ("need precomputed tables");
            }


            return dis0;
        }


        /// list_specific intialization
        void init_list (Index::idx_t key, float coarse_dis,
                        size_t list_size_in, const IDType *list_ids_in,
                        const groupt *list_codes_in) {
            this->key = key;
            this->coarse_dis = coarse_dis;
            list_size = list_size_in;
            list_codes = list_codes_in;
            list_ids = list_ids_in;
        }

        /*****************************************************
         * Scaning the codes: simple PQ scan.
         *****************************************************/

        /// version of the scan where we use precomputed tables
        void scan_list_with_table (
                 size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
        {
            float dis0 = precompute_list_tables ();
            pq.lookup_and_update_heap(list_size, 0, list_codes,sim_table,k, heap_sim, heap_ids, dis0,
            		key, list_ids, store_pairs);
            	already_scanned += list_size;
        }


        void scan_list_with_table_simd (
                 size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
        {
            float dis0 = precompute_list_tables ();

            size_t scanned_with_nonquantized_distances = 0;

            if(already_scanned < codes_needed_to_build_quantizer){
            	scanned_with_nonquantized_distances = std::min(codes_needed_to_build_quantizer-already_scanned,list_size);
            	pq.lookup_and_update_heap(scanned_with_nonquantized_distances, 0, list_codes,sim_table,k, heap_sim, heap_ids, dis0,
				key, list_ids, store_pairs);
            	already_scanned += scanned_with_nonquantized_distances;
            }

            if(already_scanned >= codes_needed_to_build_quantizer){


    			std::vector<typename T_VPQ::QuantTableLane,boost::alignment::aligned_allocator<typename T_VPQ::QuantTableLane, 64>> mm_dis_tables;
    			mm_dis_tables.resize(pq.dt_lanes_total);

    			typename T_VPQ::VPQQuant* qmax = pq.quantize_tables(sim_table, mm_dis_tables.data(), maxheap_worst_value(k, heap_sim));

    			/* Test if the quantizer was built, i.e. if we need to scan the inverted list */
    			if(qmax != nullptr){
    				pq.lookup_and_update_heap_simd(list_size-scanned_with_nonquantized_distances, scanned_with_nonquantized_distances, list_codes, sim_table, mm_dis_tables.data(),
    						qmax, k, heap_sim, heap_ids, dis0,
							key, list_ids, store_pairs);
    				delete qmax;
    			}

            	n_simd_eval += list_size-scanned_with_nonquantized_distances;
            	already_scanned += list_size-scanned_with_nonquantized_distances;
            }else{
            	//printf("already scanned: %d\n",(int)already_scanned);
            	//printf("needed: %d\n",(int)codes_needed_to_build_quantizer);
            	//printf("list_size: %d\n",(int)list_size);
            	//printf("scanned_nq: %d\n",(int)scanned_with_nonquantized_distances);
            	//printf("Number of entries remaining %d\n",(int)(list_size-scanned_with_nonquantized_distances));
            	FAISS_THROW_IF_NOT_MSG(0 == list_size-scanned_with_nonquantized_distances,"Remaining to scan");
            }

        }

        /// tables are not precomputed, but pointers are provided to the
        /// relevant X_c|x_r tables
        void scan_list_with_pointer (
                 size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
        {

            float dis0 = precompute_list_table_pointers ();

            for (size_t j = 0; j < list_size; j++) {

                float dis = dis0;

                for (size_t m = 0; m < pq.M; m++) {
                    unsigned ci = pq.get_code_component(list_codes, j, m);
                    dis += sim_table_ptrs [m][ci] - 2 * sim_table_2[pq.ksub_offset[m]+ci];
                }

                if (dis < heap_sim[0]) {
                    maxheap_pop (k, heap_sim, heap_ids);
                    long id = store_pairs ? (key << 32 | j) : list_ids[j];
                    maxheap_push (k, heap_sim, heap_ids, dis, id);
                }
            }
            already_scanned += list_size;

        }

        /// nothing is precomputed: access residuals on-the-fly
        void scan_on_the_fly_dist (
                 size_t k, float * heap_sim, long * heap_ids, bool store_pairs)
        {

            if (by_residual && use_precomputed_table) {
                scan_list_with_pointer (k, heap_sim, heap_ids, store_pairs);
                return;
            }

            const float *dvec;
            float dis0 = 0;

            if (by_residual) {
                if (metric_type == METRIC_INNER_PRODUCT) {
                    ivfpq.quantizer->reconstruct (key, residual_vec);
                    dis0 = fvec_inner_product (residual_vec, qi, d);
                } else {
                    ivfpq.quantizer->compute_residual (qi, residual_vec, key);
                }
                dvec = residual_vec;
            } else {
                dvec = qi;
                dis0 = 0;
            }

            for (size_t j = 0; j < list_size; j++) {

                pq.decode (list_codes, decoded_vec, j);

                float dis;
                if (metric_type == METRIC_INNER_PRODUCT) {
                    dis = -dis0 - fvec_inner_product (decoded_vec, qi, d);
                } else {
                    dis = fvec_L2sqr (decoded_vec, dvec, d);
                }

                if (dis < heap_sim[0]) {
                    maxheap_pop (k, heap_sim, heap_ids);
                    long id = store_pairs ? (key << 32 | j) : list_ids[j];
                    maxheap_push (k, heap_sim, heap_ids, dis, id);
                }
            }
            already_scanned += list_size;
        }



    };

    void search_preassigned (idx_t nx, const float *qx, idx_t k,
                                const idx_t *keys,
                                const float *coarse_dis,
                                float *distances, idx_t *labels,
                                bool store_pairs,
								const IVFSearchParameters *params=nullptr
    							) const override {
       	   float_maxheap_array_t res = {
       	        size_t(nx), size_t(k),
       	        labels, distances
       	    };

       	#pragma omp parallel
       	    {
       	        QueryTables<long> qt (*this,k,params);
       	        size_t stats_nlist = 0;
       	        size_t stats_ncode = 0;
       	        uint64_t init_query_cycles = 0;
       	        uint64_t scan_cycles = 0;
       	        uint64_t heap_cycles = 0;
       	        long local_nprobe = params ? params->nprobe : nprobe;
       	        long local_max_codes = params ? params->max_codes : max_codes;
       	        //uint64_t local_scan_table_threshold_simd= params ? params-> scan_table_threshold_simd :  scan_table_threshold_simd;
       	        //uint64_t local_scan_table_threshold= params ? params-> scan_table_threshold :  scan_table_threshold;

       	#pragma omp for
       	        for (size_t i = 0; i < nx; i++) {
       	            const float *qi = qx + i * d;
       	            const long * keysi = keys + i * nprobe;
       	            const float *coarse_dis_i = coarse_dis + i * nprobe;
       	            float * heap_sim = res.get_val (i);
       	            long * heap_ids = res.get_ids (i);

       	            uint64_t t0;
       	            TIC;
       	            maxheap_heapify (k, heap_sim, heap_ids);
       	            heap_cycles += TOC;

       	            TIC;
       	            qt.init_query (qi);
       	            init_query_cycles += TOC;

       	            size_t nscan = 0;


       	            for (size_t ik = 0; ik < local_nprobe; ik++) {
       	                long key = keysi[ik];  /* select the list  */
       	                if (key < 0) {
       	                    // not enough centroids for multiprobe
       	                    continue;
       	                }
       	                FAISS_THROW_IF_NOT_FMT (
       	                    key < (long) nlist,
       	                    "Invalid key=%ld  at ik=%ld nlist=%ld\n",
       	                    key, ik, nlist);

       	                size_t list_size = invlists->list_size (key);
       	                stats_nlist ++;
       	                nscan += list_size;

       	                if (list_size == 0) continue;

       	                qt.init_list (key, coarse_dis_i[ik],
       	                              list_size,  InvertedLists::ScopedIds (invlists, key).get(),
       	                              group_codes[key].data());

       	                TIC;
       	                if(list_size > scan_table_threshold_simd) {
       	                	qt.scan_list_with_table_simd (k, heap_sim, heap_ids, store_pairs);
       	                }else if(list_size > scan_table_threshold) {
       	                	qt.scan_list_with_table (k, heap_sim, heap_ids, store_pairs);
       	                } else {
       	                	qt.scan_on_the_fly_dist (k, heap_sim, heap_ids, store_pairs);
       	                }
       	                scan_cycles += TOC;

       	                if (local_max_codes && nscan >= local_max_codes) break;
       	            }
       	            stats_ncode += nscan;
       	            TIC;
       	            maxheap_reorder (k, heap_sim, heap_ids);

       	            if (metric_type == METRIC_INNER_PRODUCT) {
       	                for (size_t j = 0; j < k; j++)
       	                    heap_sim[j] = -heap_sim[j];
       	            }
       	            heap_cycles += TOC;
       	        }

       	#pragma omp critical
       	        {
       	            indexIVFPQ_stats.nlist += stats_nlist;
       	            indexIVFPQ_stats.ncode += stats_ncode;

       	            indexIVFPQ_stats.n_hamming_pass += qt.n_simd_eval;

       	            indexIVFPQ_stats.init_query_cycles += init_query_cycles;
       	            indexIVFPQ_stats.init_list_cycles += qt.init_list_cycles;
       	            indexIVFPQ_stats.scan_cycles += scan_cycles - qt.init_list_cycles;
       	            indexIVFPQ_stats.heap_cycles += heap_cycles;
       	        }

       	    }
       	    indexIVFPQ_stats.nq += nx;
       	}

};


// Reuse statistic structure from IVFPQ


template <class T>
inline std::string fourcc_vpq(const IndexIVFVPQ<T>* n){return "J"+cc_vpq((T*)NULL);}


} // namespace faiss




#endif
