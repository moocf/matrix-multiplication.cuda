#pragma once
#include <string.h>
#include <stdio.h>
#include "support.h"


// Calculates matrix product on CPU.
void matrix_product(float* a, float* x, float* y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<YC; c++) {
      float s = 0;
      for (int i=0; i<XC; i++)
        s += x[r*XC + i] * y[i*YC + c];
      a[r*YC + c] = s;
    }
  }
}


void test_populate(float *x, float *y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<XC; c++)
      GET2D(x, r, c, XC) = (r*XC + c)*1e-8f;
  }
  for (int r=0; r<XC; r++) {
    for (int c = 0; c < YC; c++)
      GET2D(y, r, c, YC) = (r*YC + c)*1e-8f;
  }
}


void test_print(float *exp, float *ans, int R, int C, float duration) {
  printf("Execution time: %.1f ms\n", duration);
  printf("Matrix element sum: %.5g\n", SUM_ARRAY(ans, R*C));
  if (exp) {
    int cmp = memcmp(exp, ans, R*C * sizeof(float));
    if (cmp != 0) printf("Result doesnt match!\n");
  }
  printf("\n");
}
