#pragma once
#include <cstdio>
#include <cstring>
#include "_main.hxx"

using std::printf;
using std::memcmp;


// Calculates matrix product on CPU.
void matrixProduct(float* a, float* x, float* y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<YC; c++) {
      float s = 0;
      for (int i=0; i<XC; i++)
        s += x[r*XC + i] * y[i*YC + c];
      a[r*YC + c] = s;
    }
  }
}


void testPopulate(float *x, float *y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<XC; c++)
      GET2D(x, r, c, XC) = (float) r*XC + c;
  }
  for (int r=0; r<XC; r++) {
    for (int c = 0; c < YC; c++)
      GET2D(y, r, c, YC) = (float) r*YC + c;
  }
}


void testPrint(float *exp, float *ans, int R, int C, float duration) {
  printf("Execution time: %.1f ms\n", duration);
  printf("Matrix element sum: %.5g\n", sum(ans, R*C));
  if (exp) {
    int cmp = memcmp(exp, ans, R*C * sizeof(float));
    if (cmp != 0) printf("Result doesnt match!\n");
  }
  printf("\n");
}
