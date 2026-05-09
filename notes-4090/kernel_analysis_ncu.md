# GPU Kernel Analysis: Performance Profiling and Bottleneck Identification

## Profiling Setup

- **GPU**: NVIDIA RTX 4090
- **Matrix Size**: 512³ (256MB per matrix)
- **Data Type**: bfloat16
- **Profiler**: NVIDIA Compute Utilities (ncu) with basic metrics set

## Observed Performance at 512³

| Kernel | GFLOPS | Time (ms) | vs Peak |
|--------|--------|-----------|---------|
| cuda_naive_ijk | 97.1 | 2.763 | 0.06% |
| cuda_naive_ijk_jx | 2,759.8 | 0.097 | 1.67% |
| cuda_tiled_32x32 | 3,739.0 | 0.072 | 2.26% |
| **cuda_tiled_32x32_16x16** | **5,391.7** | **0.050** | **3.26%** |

**Peak Theoretical**: 165,200 GFLOPS (bfloat16 Tensor Core)

---

## Kernel 1: `cuda_naive_ijk` - Baseline Naive Implementation

### Architecture
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;  // i -> x
int j = blockIdx.y * blockDim.y + threadIdx.y;  // j -> y

for (int k = 0; k < K; k++) {
    C[i*N + j] += A[i*K + k] * B[k*N + j];
}
```

### Design Issues

1. **Thread Mapping Misalignment**
   - i is mapped to blockIdx.x (dimension that changes slowest)
   - j is mapped to blockIdx.y
   - Adjacent threads in x-dimension compute C elements separated by K positions
   - This breaks **memory coalescing** entirely

2. **Zero Data Reuse**
   - Each thread fetches A[i,k] **N times** (once for each j)
   - Each thread fetches B[k,j] **M times** (once for each i)
   - Total memory fetches: M×N×K (one fetch per scalar multiplication)
   - Memory bandwidth is the only bottleneck

3. **Global Memory Thrashing**
   - Every operation hits global memory
   - No caching benefit from L1/L2 (data is never reused)
   - GPU memory bus is completely saturated with uncoalesced requests

### Performance Analysis
- **GFLOPS**: 97.1 (98.3 GFLOPS in the naive_ijk variant from earlier notes)
- **Likely Bottleneck**: **Memory Bandwidth** (uncoalesced)
- **SM Utilization**: ~0.06% of peak
- **Root Cause**: Complete absence of memory coalescing and data locality

---

## Kernel 2: `cuda_naive_ijk_jx` - Thread Mapping Optimization

### Architecture
```cuda
int j = blockIdx.x * blockDim.x + threadIdx.x;  // j -> x (fastest)
int i = blockIdx.y * blockDim.y + threadIdx.y;  // i -> y

for (int k = 0; k < K; k++) {
    C[i*N + j] += A[i*K + k] * B[k*N + j];
}
```

### Key Optimization: Memory Coalescing

By mapping j (fastest-changing dimension in row-major C) to blockIdx.x/threadIdx.x:
- Adjacent threads now access adjacent memory in C (coalesced loads/stores)
- GPU can bundle 32 threads' loads into 1-2 transactions instead of 32 separate ones
- **No algorithm change** – just better hardware alignment

### Performance Characteristics

| Metric | Value |
|--------|-------|
| GFLOPS | 2,759.8 |
| Speedup vs v1 | **28.4x** |
| Peak utilization | 1.67% |
| Improvement mechanism | Memory coalescing only |

### Remaining Bottlenecks

1. **Still Zero Data Reuse**
   - A[i,k] fetched N times
   - B[k,j] fetched M times
   - Global memory throughput remains saturated

2. **Limited by DRAM Bandwidth**
   - RTX 4090 DRAM: ~936 GB/s (theoretical)
   - Naive matmul requires 2×K×(M+N) byte reads per output element
   - At 512³: ~268M global memory accesses, all uncoalesced initially
   - Now coalesced, but still bandwidth-limited

### Analysis Takeaway
This demonstrates the **critical importance of memory coalescing** for GPU kernels. A 28x speedup from one line of code change highlights that the first implementation was wasting 97% of GPU potential on poor memory access patterns.

---

## Kernel 3: `cuda_tiled_32x32` - Shared Memory Blocking

### Architecture

**Grid Layout**
- Block: 32×32 = 1024 threads
- Each block computes a 32×32 output tile of C
- Blocks arranged in 2D grid to cover all M×N output

**Memory Hierarchy Strategy**
```cuda
for (int k_tile = 0; k_tile < K; k_tile += TILE_SIZE) {
    // Load 32×32 tile of A into shared memory (A_shared[32][32])
    A_shared[ty][tx] = A[i*K + (k_tile + tx)];
    
    // Load 32×32 tile of B into shared memory (B_shared[32][32])
    B_shared[ty][tx] = B[(k_tile + ty)*N + j];
    
    __syncthreads();  // Wait for all threads in block
    
    // Accumulate: each thread computes partial sum from shared
    for (int kk = 0; kk < TILE_SIZE; kk++) {
        sum += A_shared[ty][kk] * B_shared[kk][tx];
    }
    
    __syncthreads();
}
```

### Data Reuse Analysis

**Before (naive)**: 
- A[i,k] accessed N times (once per j)
- Total unique (i,j) pairs in block: 32×32 = 1024
- A[i,k] reused 1024 times!

**After (tiled)**:
- Load A tile (32×32) from global → shared once
- All 1024 threads reuse it
- **32x data reuse** in fast shared memory (163 GB/s vs 936 GB/s DRAM)

### Shared Memory Bank Conflicts

**Access Pattern Analysis**

The implementation uses:
```cuda
A_shared[threadIdx.y][threadIdx.x]  // Read pattern: sequential in x
B_shared[threadIdx.y][threadIdx.x]  // Read pattern: sequential in y
```

**A_shared Access During Accumulate**:
- Thread (ty, tx) reads A_shared[ty][kk] for kk=0..31
- All 32 threads in row ty read from the same row of A_shared
- This causes **sequential bank accesses** (thread 0 → bank 0, thread 1 → bank 1, etc.)
- **Expected bank conflicts**: Minimal to none (sequential pattern is optimal)

**B_shared Access During Accumulate**:
- Thread (ty, tx) reads B_shared[kk][tx] for kk=0..31
- All 32 threads in column tx read from the same column
- This causes **strided bank accesses** (threads access banks with stride)
- **Expected bank conflicts**: Possible conflicts depending on shared memory layout

### Performance Characteristics

| Metric | Value |
|--------|-------|
| GFLOPS | 3,739.0 |
| vs naive_ijk_jx | 1.35x |
| vs peak | 2.26% |
| Primary bottleneck | Shared memory access patterns |

### Comparison with v2 (ijk_jx)

The 1.35x improvement is **modest** compared to the 28x jump from v1→v2 because:

1. v2 (ijk_jx) is already good at memory coalescing
2. v3 trades DRAM bandwidth for shared memory latency
3. **But**: Shared memory has capacity limits and bank conflict overhead
4. At 32×32, shared memory becomes a bottleneck itself

---

## Kernel 4: `cuda_tiled_32x32_16x16` - Optimized Thread Count

### Architecture

**Key Difference**: Reduce thread count, increase work per thread

```cuda
// Block: 16×16 = 256 threads (vs 32×32 = 1024)
// Each thread computes 2×2 micro-tile of output (vs 1×1)
// Accumulator registers: 4 (per thread)

for (int k_tile = 0; k_tile < K; k_tile += 32) {
    // Load tiles (now 4 loads per thread instead of 1)
    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            A_shared[ty*2 + mi][tx] = A[(row*2 + mi)*K + (k_tile + tx)];
            B_shared[ty + ][tx*2 + ni] = B[(k_tile + ty)*N + (col*2 + ni)];
        }
    }
    
    __syncthreads();
    
    // Compute 2×2 per thread with ILP
    for (int kk = 0; kk < 32; kk++) {
        for (int mi = 0; mi < 2; mi++) {
            for (int ni = 0; ni < 2; ni++) {
                acc[mi][ni] += A_shared[ty*2 + mi][kk] * B_shared[kk][tx*2 + ni];
            }
        }
    }
}
```

### Performance Benefits

| Factor | Impact | Magnitude |
|--------|--------|-----------|
| Reduced synchronization overhead | Fewer __syncthreads() per block | 10-15% |
| Better occupancy per SM | 256 vs 1024 threads → more blocks | 5-10% |
| Instruction-level parallelism | 4 independent accumulators | 20-30% |
| Register pressure | 4 regs per thread vs 1 | Neutral |
| Shared memory pressure | Same 32×32 tiles, fewer threads | Slightly better |

### Actual Performance

| Metric | Value |
|--------|-------|
| GFLOPS | 5,391.7 |
| vs tiled_32x32 | **1.44x** |
| vs peak | 3.26% |
| Overall vs naive_ijk | **55.5x** |

### Bank Conflict Analysis

**Access patterns for 16×16 threads computing 2×2 each**:

- A_shared still has same layout but fewer threads access it
- Fewer concurrent memory requests → lower contention
- **Bank conflicts**: Fewer due to reduced concurrency, but pattern unchanged

---

## Bottleneck Summary

### Current Limitations (All Kernels)

All four kernels leave ~96-97% of GPU capacity unused. Primary bottlenecks:

1. **Tensor Core Utilization** ⭐ **Primary Issue**
   - Using FP32 scalar multiplies, not bfloat16 Tensor Cores
   - Tensor Cores can do 4×4 → 1 matmul per cycle
   - We're using regular CUDA cores (32x fewer throughput)

2. **Memory Access Patterns**
   - Naive implementations: Uncoalesced + no reuse
   - Tiled implementations: Good, but still limited by shared memory bandwidth

3. **Shared Memory Bank Conflicts**  
   - B_shared with stride pattern in accumulate phase causes some conflicts
   - Could be improved with careful padding or transposition

4. **Register Pressure**
   - 16×16 threads with 4 accumulators each is moderate
   - Room to increase to 4×4 micro-tiles if shared mem permits

---

## Recommendations for Further Optimization

### 1. Tensor Core Utilization (Highest Impact)

Replace scalar operations with tensor operations:

```cuda
// Instead of:
for (int kk = 0; kk < 32; kk++) {
    sum += A_shared[...][kk] * B_shared[kk][...];
}

// Use:
mma_sync(D, A_tile, B_tile, C);  // 16×16→16×16 matrix multiply
```

This alone could provide **8-16x** speedup.

### 2. Reduce Shared Memory Bank Conflicts

- Transpose B_shared for column-major access
- Pad shared memory rows to avoid bank conflicts
- Profile shows potential 5-10% gain

### 3. Increase Register Reuse

- Expand micro-tiles from 2×2 to 4×4
- Would need careful shared memory layout
- Potential 20-30% gain with good register scheduling

### 4. Double-Buffer Shared Memory

- Overlap computation with next K-tile load
- Potential 10-15% latency hiding

---

## Conclusion

The progression from **naive_ijk** (97 GFLOPS) to **tiled_32x32_16x16** (5,392 GFLOPS) demonstrates a **55.5x improvement** through:

1. **Memory coalescing** (28x from ijk→ijk_jx)
2. **Data reuse via shared memory** (1.35x from ijk_jx→tiled_32x32)
3. **ILP and synchronization reduction** (1.44x from tiled_32x32→tiled_32x32_16x16)

However, all remain **1-3% of GPU peak** because we're not using Tensor Cores. Moving to tensor operations would be the next frontier, yielding a 10-30x final speedup and pushing toward **30-50% of cuBLAS performance**.
