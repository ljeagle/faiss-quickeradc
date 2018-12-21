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

/* Copyright 2004-present Facebook. All Rights Reserved.
   Index based on product quantiztion.
*/

#include "IndexVPQ.h"


#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cmath>

#include <algorithm>

#include "FaissAssert.h"

namespace faiss {



/*****************************************
 * Stats of IndexPQ codes
 ******************************************/


void IndexVPQStats::reset()
{
    nq = ncode  = 0;
}

IndexVPQStats indexVPQ_stats;





} // END namespace faiss
