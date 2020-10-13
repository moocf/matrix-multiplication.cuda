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
