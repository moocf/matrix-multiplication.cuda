#pragma once
#include <stdlib.h>
#include <string.h>
#include <time.h>


// Testing performance of 100 memory copy operations
// between CPU memory allocated with malloc().
float test_malloc(int size) {
  void *a = malloc(size);
  void *b = malloc(size);
  clock_t start = clock();

  for (int i=0; i<100; i++)
    memcpy(b, a, size);

  clock_t stop = clock();
  float duration = (float) (stop - start) / CLOCKS_PER_SEC;
  return duration * 1000;
}
