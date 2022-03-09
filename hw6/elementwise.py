from numba import jit, cuda, prange
import numpy as np 
# to measure exec time 
from timeit import default_timer as timer 
import math

@jit(parallel=True)
def cpu(w, t, p, result):
  """Perform element-wise operation: result = (w*t+p)
     Note that w and p are 1-D arrays of identical length, so is result. t is scalar
  """
  for i in prange(len(w)):
    result[i] = (w[i]*t+p[i])

@cuda.jit
def gpu(w, t, p, result):
    """Perform element-wise operation: result = (w*t+p)
       Note that w and p are 1-D arrays of identical length, so is result. t is scalar
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if i < w.shape[0]:
        result[i] = (w[i]*t+p[i])

if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    # dtype = getattr(np, argv[2])
    dtype = np.float64
    assert issubclass(dtype, np.number)
    w = np.random.rand(n).astype(dtype)
    p = np.random.rand(n).astype(dtype)
    result = np.zeros((n,)).astype(dtype)
    result2 = np.zeros((n,)).astype(dtype)
    d_w = cuda.device_array_like(w)
    d_p = cuda.device_array_like(p)
    d_r = cuda.device_array_like(result)

    for i in range(5):
        t=np.random.random()

        start = timer()
        cpu(w, t, p, result)
        fin = timer()
        print("CPU: %.4f"%(fin-start))

        blockDim = 1024
        gridDim = math.ceil(n/blockDim)
        cuda.synchronize()
        start = timer()
        # Transfer the arrays to the GPU
        cuda.to_device(w, to=d_w)
        cuda.to_device(p, to=d_p)
        cuda.synchronize()
        mid1=timer()
        gpu[gridDim, blockDim](d_w,t,d_p,d_r)
        cuda.synchronize()
        mid2=timer()
        d_r.copy_to_host(result2) 
        cuda.synchronize()
        fin = timer()
        print("GPU: %.4f %.4f %.4f %.4f"%(fin-start, mid1-start, mid2-mid1, fin-mid2))

        print()
