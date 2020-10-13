The resultant of two matrices, formed by calculating
dot-product of respective components is called matrix
multiplication.

```c
test_host():
Testing matrix multiplication on the CPU.
A = X . Y
X: 1024 x 1024
Y: 1024 x 1024
```

```c
test_simple():
Testing simple matrix multiplication on the GPU.
This is the standard CPU algorithm adapted to use
GPU blocks amd threads.
```

```c
test_tiled():
Testing tiled matrix multiplication on the GPU.
Each thread-block computes the resultant of a small
square sub-matrix by fetching parts of input matrices
and storing partial results in the resultant sub-matrix.
This is repeated for all sub-matrices until full
matrix multiplication is calculated.
```

```bash
# OUTPUT
CPU matrix multiplication ...
Execution time: 3078.0 ms

GPU matrix multiplication, simple ...
Execution time: 23.5 ms

CPU matrix multiplication, tiled ...
Execution time: 32.8 ms
```

See [main.cu] for code, [main.ipynb] for notebook.

[main.cu]: main.cu
[main.ipynb]: https://colab.research.google.com/drive/14LZMQ_uI2nSLTNpnwaGcQ7O6LzK604qv?usp=sharing


### references

- [Running a parallel matrix multiplication program using CUDA on FutureGrid :: Indiana University](https://kb.iu.edu/d/bcgu)
