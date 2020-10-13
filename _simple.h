#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "support.h"


__global__ void kernel_simple(float *a, float *x, float *y, int XR, int XC, int YC) {
  DEFINE(tx, ty, bx, by, BX, BY);
  int r = by*BY + ty;
  int c = bx*BX + tx;
  
  float s = 0;
  for (int i=0; i<XC; i++)
    s += GET2D(x, r, i, XC) * GET2D(y, i, c, YC);
  GET2D(a, r, c, YC) = s;
}


float test_simple(float *a, float *x, float *y, int XR, int XC, int YC) {
  size_t A1 = XR * YC * sizeof(float);
  size_t X1 = XR * XC * sizeof(float);
  size_t Y1 = XC * YC * sizeof(float);

  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );
  TRY( cudaEventRecord(start, 0) );

  float *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, Y1) );

  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, Y1, cudaMemcpyHostToDevice) );

  dim3 threads(16, 16);
  dim3 blocks(CEILDIV(XR, 16), CEILDIV(YC, 16));
  kernel_simple<<<blocks, threads>>>(aD, xD, yD, XR, XC, YC);

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
