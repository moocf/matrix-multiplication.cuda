#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "support.h"


// Testing performance of 100 memory copy operations
// between CPU memory allocated with malloc() and
// GPU memory allocated with cudaMalloc(). Because
// memory allocated with malloc() is pageable memory,
// it will first be copied to a page-locked `staging`
// area, before being transferring to GPU by DMA.
// Note however that allocating too much pinned memory
// can cause system slowdown, or even crash due to
// lack of usable memory.
float test_cuda_malloc(int size, bool up) {
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );

  void *a, *aD;
  a = malloc(size);
  TRY( cudaMalloc(&aD, size) );
  TRY( cudaEventRecord(start, 0) );

  for (int i=0; i<100; i++) {
    if (up) TRY( cudaMemcpy(aD, a, size, cudaMemcpyHostToDevice) );
    else TRY( cudaMemcpy(a, aD, size, cudaMemcpyDeviceToHost) );
  }

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(aD) );
  free(a);
  return duration;
}
