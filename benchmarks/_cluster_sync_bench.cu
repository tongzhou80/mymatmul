// Microbenchmark: cost of one cluster.sync() vs __syncthreads() vs empty loop.
//
// Runs N_ITERS of the barrier in a tight loop, uses clock64() to measure cycles,
// outputs (avg cycles per iter) per CTA.
//
// Three kernels:
//   empty_loop      — pure loop overhead baseline
//   syncthreads_loop — __syncthreads (intra-CTA)
//   cluster_sync_loop — cluster.sync() with __cluster_dims__(1, 2, 1)
//
// Per-iter cost is reported as median across CTAs.

#include <stdint.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#ifndef N_ITERS
#define N_ITERS 1000
#endif

// Empty loop with a dependency to prevent DCE
extern "C" __global__ __launch_bounds__(128, 1)
void empty_loop(unsigned long long* out_cycles, int n_iters) {
    int x = threadIdx.x;
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        x = x ^ i;
        asm volatile("" : "+r"(x) :: "memory");  // prevent loop optimization
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        const int linear = blockIdx.y * gridDim.x + blockIdx.x;
        out_cycles[linear] = (t1 - t0);
    }
}

extern "C" __global__ __launch_bounds__(128, 1)
void syncthreads_loop(unsigned long long* out_cycles, int n_iters) {
    int x = threadIdx.x;
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        __syncthreads();
        x = x ^ i;
        asm volatile("" : "+r"(x) :: "memory");
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        const int linear = blockIdx.y * gridDim.x + blockIdx.x;
        out_cycles[linear] = (t1 - t0);
    }
}

extern "C" __global__ __cluster_dims__(1, 2, 1) __launch_bounds__(128, 1)
void cluster_sync_loop(unsigned long long* out_cycles, int n_iters) {
    auto cluster = cg::this_cluster();
    int x = threadIdx.x;
    unsigned long long t0 = clock64();
    #pragma unroll 1
    for (int i = 0; i < n_iters; i++) {
        cluster.sync();
        x = x ^ i;
        asm volatile("" : "+r"(x) :: "memory");
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        const int linear = blockIdx.y * gridDim.x + blockIdx.x;
        out_cycles[linear] = (t1 - t0);
    }
}
