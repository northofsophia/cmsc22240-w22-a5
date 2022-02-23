import argparse
import time
from mpi4py import MPI

# The input file contains a 100,000x10 matrix.
# The matrix we are going to use is the first n lines of it, named A.
# And we are going to compute transpose(A)*A in outer product fashion.
# The end result should be a 10x10 matrix.
# Complete the functions below as directed.


# In this function, you seperate matrix into k submatrices and return the array
# The last matrix can have different dimension than (n//k)x10
# Note: you should not hardcode the dimension 10.
# Inputs:
#   matrix => nx10 matrix
#   k      => rows per submatrix
# Output:
#   k number of (n//k)x10 submatrices
def create_submatrices(matrix, k):
    submatrices = []
    # TODO: complete this function
    print("Remove me when finish!")
    return submatrices


# In this function, you calculate the matrix multiplication of transpose(matrix)*matrix
# Note: you should not hardcode the dimension 10.
# Hint: you should not need to actually transpose the matrix, just index it differently
# Input:
#   matrix => [[...], [...], ...]
# Output:
#   A 10x10 partial matrix
def multiply_rows(matrix):
    partial_matrix = []
    # TODO: complete this function
    print("Remove me when finish!")
    return partial_matrix


# In this function, you sum up 2 partial matrices
# Note: you should not hardcode the dimension 10.
# Input:
#   Two 10x10 partial matrices
# Output:
#   A 10x10 summed/reduced matrix
def sum_partial_matrices(part_mat_a, part_mat_b):
    if len(part_mat_a) == 0:
        return part_mat_b
    # TODO: complete this function
    print("Remove me when finish!")
    return res


def mpi_mm():
    if rank == 0:
        submats = create_submatrices(matrix, int(pargs.k))
    else:
        submats = None
    submat = comm.scatter(submats, root=0)
    partmat = multiply_rows(submat)
    resmat = comm.reduce(partmat, op=mat_redu_op)
    return resmat


def debug_print(*args):
    if pargs.debug:
        print(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_in', default='matrix_data.txt')
    parser.add_argument('-o', '--data_out', default='mm_out.txt')
    parser.add_argument('-n')
    parser.add_argument('-k')
    parser.add_argument('--debug', action="store_true")
    pargs = parser.parse_args()

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mat_redu_op = MPI.Op.Create(lambda a, b, dt: sum_partial_matrices(a, b), commute=True)
    debug_print('Rank =', rank, 'Node name =', MPI.Get_processor_name())

    matrix = []
    if rank == 0:
        # Read matrix up to Nx10
        with open(pargs.data_in, 'r') as f:
            for i, ln in enumerate(f):
                if i >= int(pargs.n):
                    break
                matrix.append([float(j) for j in ln.split(' ') if j.strip()])
        # Master
        print("=== COMPUTATION STARTS ===")
        print("==> Input file:", pargs.data_in)
        print("==> n =", pargs.n)
        print("==> k =", pargs.k)

    # Wait all processes are ready
    comm.barrier()
    t1 = time.time()

    # Compute
    res_mat = mpi_mm()

    # Wait all processes to finish
    comm.barrier()

    if rank == 0:
        t2 = time.time()
        print("=== COMPUTATION COMPLETE ===")
        # Write results to file
        with open(pargs.data_out, 'w') as f:
            for row in res_mat:
                for elem in row:
                    f.write('%.6f ' % elem)
                f.write('\n')
        print("==> Result matrix written to:", pargs.data_out)
        print('==> Time elapsed = %.4f s' % (t2 - t1))
        print("=== PROGRAM ENDS ===")
