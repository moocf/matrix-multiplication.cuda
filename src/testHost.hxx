#pragma once
#include <ctime>
#include "test.hxx"

using std::clock_t;
using std::clock;


float testHost(float* a, float* x, float* y, int XR, int XC, int YC) {
  clock_t start = clock();
  matrixProduct(a, x, y, XR, XC, YC);
  clock_t stop = clock();
  float duration = (float)(stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
}
