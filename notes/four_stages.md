This repo is intended for optimizing GEMM (or matmul) step by step. 
We do not distinguish GEMM from matmul in this documentation as their computations are exactly the same with GEMM adds a scaling factor and an offset when writing back 
the results. The underlying optimization principles are identical. Matrix muliplication is used for illustration in this work.
 
# Terminology
Consider matrix muliplication (matmul) as 

C = A x B

We note the dimensionality of the matrices as:

A: M x K
B: K x N
C: M x N

Sometimes the M, K, N dimensions are referred to as i, k, j as well, if the context is more code-oriented. A naive implementation of matmul is often expressed as 

```python
for i in range(M):
    for j in range(N):
        C[i,j] = 0
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j] 
```

We will start with a naive "outer i-j parallelized" version where we launch 2-D grid with 2-D thread layout, and each thread loops over the entire k-dim to 
compute one output element. In CUDA style this would be
```
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;

if (i < M && j < N) {
    float acc = 0f;
    for (int k = 0; k < K; k++) {
	acc += ....;
    }
    C[i*N + j] = acc;
}
``` 
with a square thread layout, e.g. 16x16 thread block luanch.

We then progress the optimizations in 4 stages:

# Stage 1: Memory Coalescing
The first optimization at this stage is to map the x-dim (thread layout dimension) to `j` instead of `i`, as `x-dim` is the fastest changing dimension in CUDA (a common
confusing design to CUDA beginners!) so it should be mapped to the `j` dimension for more contiguous memory access patterns.

```
int j = blockDim.x * blockIdx.x + threadIdx.x;
int i = blockDim.y * blockIdx.y + threadIdx.y;
```

The next optimization in this phase is to make the thread layout x-dim a multiple of 32, the size of a warp, and the unit of thread execution in NV GPUs. This part does
not touch the CUDA kernel, and just changes the thread block layout from 16x16 to 32x8. Or 32x4 for a smaller thread block. The size of a thread block determines the GPU
occupancy but for this current implementation which is very simple, either 32x4 or 32x8 should have a similar occupancy.

This is known as "global memory coalescecing" in GPU terminology, where basically threads from the same warp access contiguous memory locations which can best utilize the 
global memory bandwidth.

# Stage 2: Shared Memory Tiling - Tradeoff Between Arith. Intensity and Occupancy
The kernels in stage 1 are easy to understand but they do not exploit any data reuse. Matmul itself actually has plenty of data reuse opportunity - at a $O(N^3)$ complexity
it loads only $O(N^2)$ data elements (in case of a square matrix). Being able to reuse data inside either shared memory or registers would significantly reduce the number
of global memory accesses, which is tens of thousands of times slower than the arithemic units (computes).

The simplest form of shared memory tiling is to still make each thread compute just one single output element, but reorganize the computation into 3 distinct phases:

- Phase 1: Threads cooperatively load a tile of A and a tile of B from global memory to on-chip shared memory.
- Phase 2: Each thread loops over a tile of the K dimension, and accumulate the partial result.
- Phase 3: Phase 1 and 2 alternates in an outer loop which loops over the K tiles. Once this outer loop finishes, each thread writes back to its assigned output elemenet.

We now need to determine the tile sizes of each dimension M, N, and K, which are often called BM, BN, and BK. To map the thread layout exactly to one output tile, BM will 
be 8 and BN will be 32 (suppose the thread layout is 32x8, and remember x-dim corresponds to the N dimension), and a common starting point for BK is 32 (warp size). 

Putting it all together, the CUDA pseudocode is

```cuda
__shared__ A_tile[BM][BK];
__shared__ B_tile[BK][BN];

int j = blockIdx.x * BN + threadIdx.x;
int i = blockIdx.y * BM + threadIdx.y;

float acc = 0f;
 
// Outer loop over K-tiles
for (int k = 0; k < K; k += BK) {
    // Threads load a BMxBK tile from A
    ...

    // Threads load a BKxBN tile from B
    ...


    // Sync is necessary because a thread block can have multiple warps
    __syncthreads(); 

    // Compute using data from shared memory
    for (int kk = 0; kk < BK; kk += 1) {
        acc += ...
    }
}

// Write back
C[i*N + j] = acc;
``` 

For kernel luanch, we still use a 32x8 thread layout.

While this implementation alone won't achieve super high performance, it clearly shows the structure of shared memory tiling, and lays a foundation to explore increasing arith. intensity, as well as how that affects occupancy.

The idea of increaing arith. intensity is simply doing more compute (FMA) per memory load. We can achieve this because matmul inherently contains data reuse opportunities! Now we have two knobs to play with - the tile sizes (BM, BN, BK) and the thread count and layout e.g. 32x8, or 32x4 or 16x16 etc. Tile sizes determine compute per global memory load while thread layout (and 
the register reuse that comes with it) determines computer per shared memory load. In general, larger tile sizes give us more data reuse in shared memory while more work per thread give 
us more data reuse in registers. But here's the thing - using two many registers per thread or too much shared memory per thread block could lower the occupancy, so there's a tradeoff 
here. 

The simplest next step to explore data reuse is to keep the same tile sizes (BM=8, BN=32, BK=32), but change the thread layout to 32x4. This achives simple register reuse while shared memory usage stays the same. Alternatively, we can increase the tile sizes to BM=16, BN=32, BK=32 while keeping thread layout as 32x8 is. At this stage, we will do quantitiave analysis
of arith. intensity vs occupancy. We will also introduce two additional symbols for per-thread micro-tile: TM and TN which represent the output tile that each thread computes. 

We now show how the arithmetic intensity is calculated.

Case 1: BM=8, BN=32, BK=32, threads=32x4 => TM=2, TN=1

Each thread now computes two output element (two vertical elements), the total # of output elements computed per CTA is 8 * 32.

TileA: 8 * 32
TileB: 32 * 32
FMAs per tile: 8 * 32 * 32

Here BK is a common factor that can be eliminated.

So # of FMA per global memory load is 8 * 32 / (8 + 32) = 6.4, # of FMA per shared memory load is 2 / (2+1) = 0.66.

Basically the formula for global memory arith. intensity is BM * BN / (BM + BN) and for shared memory arith. intensity is TM * TN / (TM + TN).

Case 2: BM=16, BN=32, BK=32, threads=32x8 => TM=2, TN=1

TM and TN are identical to case 1 so the # of FMA per shared memory load is the same. And the global memory arith. intensity is (16 * 32) / (16 + 32) = 10.66!
Wow a boost from previous 6.4! Why is that? Because 16x32 is more square-like than 8x32, and as we shall see later, a more square-like tile gives higher arith. intensity.

However, now it comes to the tradeoff between arith. intensity and occupancy. Although case 2 achieve higher arithmetic intensity but it does use a 2x as large thread 
block with 256 threads and uses more shared memory (`16*32 + 32*32` vs case 1's `8*32 + 32*32`). But the increase in shared memory usage is less than 2x so the warp occupancy
should be at least as good as case 1, or even better!  




 
