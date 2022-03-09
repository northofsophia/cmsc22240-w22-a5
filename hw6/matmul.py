from numba import jit, cuda 
import numpy as np 
import cupy as cp
# to measure exec time 
from timeit import default_timer as timer 
import math

# matrix multiplication on CPU 
@jit()
def cpu(A, B, C):
  """Perform matrix multiplication of C = A * B
  """
  for i in range(len(A)):
    for j in range(len(B[0])):
      tmp = A[i][0] * B[0][j]
      for k in range(1, len(B)):
        tmp += A[i][k] * B[k][j]
      C[i][j] = tmp

# matrix multiplication on GPU 
@cuda.jit
def gpu(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if i < C.shape[0] and j < C.shape[1]:
        tmp = C.dtype.type(0)
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i][j] = tmp

if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    # dtype = getattr(np, argv[2])
    dtype = np.float32
    assert issubclass(dtype, np.number)
    a = np.random.rand(n, n).astype(dtype)
    b = np.random.rand(n, n).astype(dtype)
    c = np.zeros((n,n)).astype(dtype)
    c2 = np.zeros((n,n)).astype(dtype)
    d_a = cuda.device_array_like(a)
    d_b = cuda.device_array_like(b)
    d_c = cuda.device_array_like(c)
    cp_c = cp.ndarray(shape=c.shape, dtype=c.dtype)

    for i in range(5):
        if n<=2000:
            start = timer()
            cpu(a,b,c)
            fin = timer()
            print("CPU: %.4f"%(fin-start))

        start = timer()
        np.matmul(a,b,out=c)
        fin = timer()
        print("CPU-np: %.4f"%(fin-start))
 
        if n<=2000:
            blockDim = 32, 32
            gridDim = math.ceil(n/blockDim[0]), math.ceil(n/blockDim[1])
            cuda.synchronize()
            start = timer()
            # Transfer the arrays to the GPU
            cuda.to_device(a, to=d_a)
            cuda.to_device(b, to=d_b)
            cuda.synchronize()
            mid1=timer()
            gpu[gridDim, blockDim](d_a,d_b,d_c)
            cuda.synchronize()
            mid2=timer()
            d_c.copy_to_host(c2) 
            cuda.synchronize()
            fin = timer()
            print("GPU: %.4f %.4f %.4f %.4f"%(fin-start, mid1-start, mid2-mid1, fin-mid2))

        cp.cuda.stream.get_current_stream().synchronize()
        start = timer()
        # Transfer the arrays to the GPU
        cuda.to_device(a, to=d_a) # Use numba to copy array onto GPU, as I didn't find a Cupy API that separates copy from memory allocation.
        cuda.to_device(b, to=d_b)
        cuda.synchronize()
        cp_a = cp.asarray(d_a) # Cupy claims such conversion is zero-copy. ref: https://docs.cupy.dev/en/stable/user_guide/interoperability.html#numba
        cp_b = cp.asarray(d_b)
        cp.cuda.stream.get_current_stream().synchronize()
        mid1=timer()
        cp.matmul(cp_a, cp_b, out=cp_c)
        cp.cuda.stream.get_current_stream().synchronize()
        mid2=timer()
        cp_c.get(out=c2) 
        cp.cuda.stream.get_current_stream().synchronize()
        fin = timer()
        print("GPU-cp: %.4f %.4f %.4f %.4f"%(fin-start, mid1-start, mid2-mid1, fin-mid2))

        print()
