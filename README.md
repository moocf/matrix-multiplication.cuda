The resultant of two matrices, formed by calculating
dot-product of respective components is called matrix
multiplication.

```c
testHost():
Testing matrix multiplication on the CPU.
A = X . Y
X: 1024 x 1024
Y: 1024 x 1024
```

```c
testSimple():
Testing simple matrix multiplication on the GPU.
This is the standard CPU algorithm adapted to use
GPU blocks amd threads.
```

```c
testTiled():
Testing tiled matrix multiplication on the GPU.
Each thread-block computes the resultant of a small
square sub-matrix by fetching parts of input matrices
and storing partial results in the resultant sub-matrix.
This is repeated for all sub-matrices until full
matrix multiplication is calculated.
```

```bash
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# CPU matrix multiplication ...
# Execution time: 3880.0 ms
# Matrix element sum: 2.9528e+20
#
# GPU matrix multiplication, simple ...
# Execution time: 7.3 ms
# Matrix element sum: 2.9528e+20
# Result doesnt match exactly!
#
# GPU matrix multiplication, tiled ...
# Execution time: 4.3 ms
# Matrix element sum: 2.9528e+20
# Result doesnt match exactly!
```

See [main.cu] for code.

[main.cu]: main.cu

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://gist.github.com/wolfram77/72c51e494eaaea1c21a9c4021ad0f320)

![](https://ga-beacon.deno.dev/G-G1E8HNDZYY:v51jklKGTLmC3LAZ4rJbIQ/github.com/moocf/matrix-multiplication.cuda)
