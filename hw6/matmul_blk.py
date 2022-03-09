from numba import jit, cuda 
import numpy as np 
import cupy as cp
# to measure exec time 
from timeit import default_timer as timer 
import math

# matrix multiplication on GPU 
@cuda.jit
def gpu(A, B, C, i0, j0, iblk, jblk):
    """Perform matrix multiplication of C = A * B
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if i < iblk and j < jblk:
        i=i0+i
        j=j0+j
        tmp = C.dtype.type(0)
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i][j] = tmp

if __name__=="__main__":
    from sys import argv
    n = int(argv[1])
    IBLK=int(argv[2])
    JBLK=int(argv[2])
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
            blockDim = math.ceil(IBLK/math.ceil(IBLK/32)), math.ceil(JBLK/math.ceil(JBLK/32))
            gridDim = math.ceil(IBLK/blockDim[0]), math.ceil(JBLK/blockDim[1])
            cuda.synchronize()
            start = timer()
            # Transfer the arrays to the GPU
            cuda.to_device(a, to=d_a)
            cuda.to_device(b, to=d_b)
            cuda.synchronize()
            mid1=timer()
            for ii in range(math.ceil(n/IBLK)):
              for jj in range(math.ceil(n/JBLK)):
                i0,j0=ii*IBLK,jj*JBLK
                gpu[gridDim, blockDim](d_a,d_b,d_c,i0,j0,min(n-i0,IBLK),min(n-j0,JBLK))
            cuda.synchronize()
            mid2=timer()
            d_c.copy_to_host(c2) 
            cuda.synchronize()
            fin = timer()
            print("GPU: %.4f %.4f %.4f %.4f"%(fin-start, mid1-start, mid2-mid1, fin-mid2))
