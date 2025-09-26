/*

Note that 5090 is of blackwell architecture, so I am not quite sure if it supports tma
or not.

Ref link:


https://github.com/triton-lang/triton/blob/22b1a44d659b099313c7920fe70da9045a70f7d8/python/tutorials/08-grouped-gemm.py#L43

https://www.youtube.com/watch?v=hQ9GPnV0-50&list=WL&index=30&t=1387s

https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/cute/tutorial/hopper/wgmma_sm90.cu


*/

#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/cluster_launch.hpp"

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;