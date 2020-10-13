#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "_host.h"
#include "_simple.h"
#include "_tiled.h"
#include "main.h"


int main() {
  int XR = 8192;
  int XC = 8192;
  int YC = 8192;
  
  int A1 = XR * YC * sizeof(float);
  int X1 = XR * XC * sizeof(float);
  int Y1 = XC * YC * sizeof(float);

  float *a = (float*) malloc(A1);
  float *x = (float*) malloc(X1);
  float *y = (float*) malloc(Y1);
  
  float *exp = (float*) malloc(A1);
  test_populate(exp, x, y, XR, XC, YC);

  printf("CPU matrix multiplication ...\n");
  test_print(exp, a, XR, YC, test_host(a, x, y, XR, XC, YC));

  printf("GPU matrix multiplication, simple ...\n");
  test_print(exp, a, XR, YC, test_simple(a, x, y, XR, XC, YC));

  printf("CPU matrix multiplication, tiled ...\n");
  test_print(exp, a, XR, YC, test_tiled(a, x, y, XR, XC, YC));
  return 0;
}
