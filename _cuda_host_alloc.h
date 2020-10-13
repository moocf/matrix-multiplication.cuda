#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "support.h"


// Testing performance of 100 memory copy operations
// between CPU memory allocated with cudaHostAlloc()
// and GPU memory allocated with cudaMalloc(). Memory
// allocated with cudaHostAlloc() is page-locked
// (pinned), which means the memory can be directly
// copied by DMA into the GPU.
float test_cuda_host_alloc(int size, bool up) {
  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );

  void *a, *aD;
  TRY( cudaHostAlloc(&a, size, cudaHostAllocDefault) );
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
  TRY( cudaFreeHost(a) );
  return duration;
}
