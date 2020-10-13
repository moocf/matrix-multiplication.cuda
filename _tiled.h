#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "support.h"


__global__ void kernel_tiled(float *a, float *x, float *y, int XR, int XC, int YC) {
  DEFINE(tx, ty, bx, by, BX, BY);
  extern __shared__ float *shared;
  float *as = &shared[0];
  float *xs = &shared[1*BY*BX];
  float *ys = &shared[2*BY*BX];
  
  int r = by*BY + ty;
  int c = bx*BX + tx;
  GET2D(as, ty, tx, BX) = 0;

  for (int i=0; i<XC; i+=BX) {
    __syncthreads();
    GET2D(xs, ty, tx, BX) = GET2D(x, r, c+i, XC);
    GET2D(ys, ty, tx, BX) = GET2D(y, r+i, c, YC);
    __syncthreads();
    for (int j=0; j<BX; j++)
      GET2D(as, ty, tx, BX) += GET2D(xs, ty, j, BX) * GET2D(ys, j, tx, BX);
  }
  __syncthreads();
  GET2D(a, r, c, YC) = GET2D(as, ty, tx, BX);
}


float test_tiled(float *a, float *x, float *y, int XR, int XC, int YC) {
  int A1 = XR * YC * sizeof(float);
  int X1 = XR * XC * sizeof(float);
  int Y1 = XC * YC * sizeof(float);

  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );

  float *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, Y1) );

  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, Y1, cudaMemcpyHostToDevice) );

  dim3 threads(16, 16);
  dim3 blocks(CEILDIV(XR, 16), CEILDIV(YC, 16));
  size_t shared = 3 * threads.x * threads.y * sizeof(float);
  kernel_tiled<<<blocks, threads, shared>>>(aD, xD, yD, XR, XC, YC);

  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop) );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return duration;
}
