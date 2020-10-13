#pragma once
#include <string.h>
#include <stdio.h>


// Calculates matrix product on CPU.
void matrix_product(float* a, float* x, float* y, int XR, int XC, int YC) {
  for (int r=0; r<XR; r++) {
    for (int c=0; c<YC; c++) {
      int s = 0;
      for (int i=0; i<XC; i++)
        s += x[r*XC + i] * y[i*YC + c];
      a[r*YC + c] = s;
    }
  }
}


void test_print(float *exp, float *ans, int R, int C, float duration) {
  printf("Execution time: %3.1f ms\n", duration);
  if (exp) {
    int cmp = memcmp(exp, ans, R * C * sizeof(float));
    if (cmp != 0) printf("Result is invalid!\n");
  }
  printf("\n");
}
