from numba import jit, cuda 
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer 
import math

@jit()
def optimized_cpu(X, A):
  """copy array X into A and do bubble sort
  """
  for i in range(len(X)):
    A[i]=X[i]
  for k in range(len(A)-1):
    for i in range(len(A)-k-1):
      if A[i]<A[i+1]:
        tmp=A[i]
        A[i]=A[i+1]
        A[i+1]=tmp

@cuda.jit
def gpu(X, A):
    """copy array X into A and do bubble sort
    """
    if cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x == 0:
        for i in range(len(X)):
            A[i]=X[i]
        for k in range(len(A)-1):
            for i in range(len(A)-k-1):
                if A[i]<A[i+1]:
                    tmp=A[i]
                    A[i]=A[i+1]
                    A[i+1]=tmp

if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    # dtype = getattr(np, argv[2])
    dtype = np.float64
    assert issubclass(dtype, np.number)
    a = np.random.rand(n).astype(dtype)
    c = np.zeros((n,)).astype(dtype)
    c2 = np.zeros((n,)).astype(dtype)
    d_a = cuda.device_array_like(a)
    d_c = cuda.device_array_like(c)

    for i in range(5):
        start = timer()
        optimized_cpu(a, c)
        fin = timer()
        print("CPU: %.4f"%(fin-start))
 
        blockDim = 1
        gridDim = 1
        cuda.synchronize()
        start = timer()
        # Transfer the arrays to the GPU
        cuda.to_device(a, to=d_a)
        cuda.synchronize()
        mid1=timer()
        gpu[gridDim, blockDim](d_a, d_c)
        cuda.synchronize()
        mid2=timer()
        d_c.copy_to_host(c2) 
        cuda.synchronize()
        fin = timer()
        print("GPU: %.4f %.4f %.4f %.4f"%(fin-start, mid1-start, mid2-mid1, fin-mid2))

        print()
