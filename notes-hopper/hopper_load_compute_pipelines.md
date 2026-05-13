On hopper, both loading data into SMEM and running mmas on SMEM data are asynchronous. 
There are multiple possible pipeline designs.

We use the following term: LD(tile_id, slot_id), and MMA(slot_it) to represent loading data into shared memory and running compute, respectively.

# 2 Buffers
Design 1:

```
LD(0, 0)

iter 0:
  LD(1, 1)
  smem_wait(1)  // wait until LD(0, 0) finishes
  syncthreads()
  MMA(0)
  mma_wait(0)
  syncthreads()  // necessary because LD(2, 0) writes to slot 0

iter 1:
  LD(2, 0)
  smem_wait(1)  // wait until LD(1, 1) finishes
  MMA(1)
  syncthreads()
```

No other possible designs for 2 buffers.

# 3 Buffers
Design 1:

```
LD(0, 0)
LD(1, 1)

iter 0:
  LD(2, 2)
  smem_wait(2)  // wait until LD(0, 0) finishes
  syncthreads()
  MMA(0)
  syncthreads()  // necessary because LD(2, 0) writes to slot 0

iter 1:
  LD(3, 0)
  smem_wait(2)  // wait until LD(1, 1) finishes
  MMA(1)
  syncthreads()

iter 2:
  LD(4, 1)
  smem_wait(2)  // wait until (2, 2) finishes
  MMA(2)
  syncthreads()

```

Now each smem_wait waits data from not 1 iteratioin ago, but 2 iterations ago - leaving more time for async smem load to finish.


Design 2:

```
LD(0, 0)
LD(1, 1)

iter 0:
  LD(2, 2)
  smem_wait(2)  // wait until LD(0, 0) finishes
  syncthreads()
  MMA(0)
  syncthreads()  // necessary because LD(2, 0) writes to slot 0

iter 1:
  LD(3, 0)
  smem_wait(2)  // wait until LD(1, 1) finishes
  MMA(1)
  syncthreads()

iter 2:
  LD(4, 1)
  smem_wait(2)  // wait until (2, 2) finishes
  MMA(2)
  syncthreads()

```