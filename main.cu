#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "src/main.hxx"

using std::free;
using std::malloc;
using std::memcpy;
using std::printf;


int main() {
  int XR = 1024;
  int XC = 1024;
  int YC = 1024;

  size_t A1 = XR * YC * sizeof(float);
  size_t X1 = XR * XC * sizeof(float);
  size_t Y1 = XC * YC * sizeof(float);

  float *a = (float*) malloc(A1);
  float *x = (float*) malloc(X1);
  float *y = (float*) malloc(Y1);

  float *exp = (float*) malloc(A1);
  testPopulate(x, y, XR, XC, YC);

  printf("CPU matrix multiplication ...\n");
  testPrint(NULL, a, XR, YC, testHost(a, x, y, XR, XC, YC));
  memcpy(exp, a, A1);

  printf("GPU matrix multiplication, simple ...\n");
  testPrint(exp, a, XR, YC, testSimple(a, x, y, XR, XC, YC));

  printf("CPU matrix multiplication, tiled ...\n");
  testPrint(exp, a, XR, YC, testTiled(a, x, y, XR, XC, YC));
  return 0;
}
