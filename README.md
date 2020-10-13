malloc() is a function for dynamic memory allocation in C.

```c
test_malloc():
Testing performance of 100 memory copy operations
between CPU memory allocated with malloc().
```

```c
test_cuda_malloc():
Testing performance of 100 memory copy operations
between CPU memory allocated with malloc() and
GPU memory allocated with cudaMalloc(). Because
memory allocated with malloc() is pageable memory,
it will first be copied to a page-locked `staging`
area, before being transferring to GPU by DMA.
Note however that allocating too much pinned memory
can cause system slowdown, or even crash due to
lack of usable memory.
```

```c
test_cuda_host_alloc():
Testing performance of 100 memory copy operations
between CPU memory allocated with cudaHostAlloc()
and GPU memory allocated with cudaMalloc(). Memory
allocated with cudaHostAlloc() is page-locked
(pinned), which means the memory can be directly
copied by DMA into the GPU.
```

```bash
# OUTPUT
CPU malloc -> CPU malloc: 92.0 ms

CPU malloc -> GPU cudaMalloc: 152.8 ms
CPU malloc <- GPU cudaMalloc: 184.7 ms

CPU cudaHostAlloc -> GPU cudaMalloc: 87.0 ms
CPU cudaHostAlloc <- GPU cudaMalloc: 86.3 ms
```

See [main.cu] for code, [main.ipynb] for notebook.

[main.cu]: main.cu
[main.ipynb]: https://colab.research.google.com/drive/1g0OANPhXuV-iyGwcaLrTMMv0JDQvw6Gd?usp=sharing


### references

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
