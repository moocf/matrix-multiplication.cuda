#pragma once
#include <time.h>
#include "main.h"


float test_host(float* a, float* x, float* y, int XR, int XC, int YC) {
  clock_t start = clock();
  matrix_product(a, x, y, XR, XC, YC);
  clock_t stop = clock();
  float duration = (float) (stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
}
