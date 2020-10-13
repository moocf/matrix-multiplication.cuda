#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "support.h"
#include "_malloc.h"
#include "_cuda_malloc.h"
#include "_cuda_host_alloc.h"


void matrix_product(float *a, float *x, float *y, int xr, int xc, int yc) {
  for (int r=0; r<xr; r++) {
    for (int c=0; c<yc; c++) {
      int s = 0;
      for (int i=0; i<xc; i++)
        s += x[r*xc + i] * y[i*yc + c];
      a[r*yc + c] = s;
    }
  }
}


float test_host(float *a, float *x, float *y, int xr, int xc, int yc) {
  clock_t start = clock();
  matrix_product(a, x, y, xr, xc, yc);
  clock_t stop = clock();
  float duration = (float) (stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
}


__global__ void kernel_simple(float *a, float *x, float *y, int xr, int xc, int yc) {
  int r = blockIdx.y*blockDim.y + threadIdx.y;
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int s = 0;
  for (int i=0; i<xc; i++)
    s += x[r*xc+ i] * y[i*yc + c];
  a[r*yc + c] = s;
}


float test_simple(float *a, float *x, float *y, int xr, int xc, int yc) {
  int A1 = xr * yc * sizeof(float);
  int X1 = xr * xc * sizeof(float);
  int Y1 = xc * yc * sizeof(float);

  cudaEvent_t start, stop;
  TRY( cudaEventCreate(&start) );
  TRY( cudaEventCreate(&stop) );

  void *aD, *xD, *yD;
  TRY( cudaMalloc(&aD, A1) );
  TRY( cudaMalloc(&xD, X1) );
  TRY( cudaMalloc(&yD, Y1) );

  TRY( cudaMemcpy(xD, x, X1, cudaMemcpyHostToDevice) );
  TRY( cudaMemcpy(yD, y, Y1, cudaMemcpyHostToDevice) );

  dim3 threads(16, 16);
  dim3 blocks(CEILDIV(xr, 16), CEILDIV(yc, 16));
  kernel_simple<<<blocks, threads>>>(aD, xD, yD, xr, xc, yc);

  TRY( cudaMemcpy(a, aD, A1, cudaMemcpyDeviceToHost) );

  float duration;
  TRY( cudaEventRecord(stop, 0) );
  TRY( cudaEventSynchronize(stop); );
  TRY( cudaEventElapsedTime(&duration, start, stop) );

  TRY( cudaEventDestroy(start) );
  TRY( cudaEventDestroy(stop) );
  TRY( cudaFree(yD) );
  TRY( cudaFree(xD) );
  TRY( cudaFree(aD) );
  return duration;
}


int main() {
  int size = 10 * 1024 * 1024;

  printf("CPU malloc -> CPU malloc: %3.1f ms\n",
    test_malloc(size));
  printf("\n");

  printf("CPU malloc -> GPU cudaMalloc: %3.1f ms\n",
    test_cuda_malloc(size, 1));
  printf("CPU malloc <- GPU cudaMalloc: %3.1f ms\n",
    test_cuda_malloc(size, 0));
  printf("\n");

  printf("CPU cudaHostAlloc -> GPU cudaMalloc: %3.1f ms\n",
    test_cuda_host_alloc(size, 1));
  printf("CPU cudaHostAlloc <- GPU cudaMalloc: %3.1f ms\n",
    test_cuda_host_alloc(size, 0));
  return 0;
}
