import argparse
import time
from mpi4py import MPI

# The input file contains a 100,000x10 matrix.
# The matrix we are going to use is the first n lines of it, named A.
# And we are going to compute transpose(A)*A in outer product fashion.
# The end result should be a 10x10 matrix.
# Complete the functions below as directed.


# In this function, you seperate matrix into k submatrices and return the array
# The last matrix can be smaller than (n//k)x10
# Inputs:
#   matrix => nx10 matrix
#   k      => rows per submatrix
# Output:
#   List of (n//k)x10 submatrices
def create_submatrices(matrix, k):
    submatrices = []
    # TODO: complete this function
    return submatrices


# In this function, you calculate the matrix multiplication of transpose(matrix)*matrix
# Hint: you should not need to actually transpose the matrix, just index it differently
# Input:
#   matrix => [[...], [...], ...]
# Output:
#   A 10x10 partial matrix
def multiply_rows(matrix):
    partial_matrix = []
    # TODO: complete this function
    return partial_matrix


# In this function, you sum up all partial matrices to get the result
# Input:
#   List of 10x10 partial matrices
# Output:
#   A 10x10 summed/reduced matrix
def sum_partial_matrices(partial_matrices):
    if len(partial_matrices) == 0:
        return []
    # TODO: complete this function


def master_op():
    submatrices = create_submatrices(matrix, int(pargs.k))

    # send sub-matrices
    debug_print('Rank =', rank, 'Master', 'Sending')
    for i, subm in enumerate(submatrices):
        comm.send(subm, dest=mapper_ranks[i % len(mapper_ranks)], tag=1)

    # end signal for mappers
    debug_print('Rank =', rank, 'Master', 'Signal mappers end')
    for i in mapper_ranks:
        comm.send([[0]], dest=i, tag=0)

    # wait mappers to finish
    debug_print('Rank =', rank, 'Master', 'Wait all mappers to complete')
    for i in mapper_ranks:
        comm.recv(source=i, tag=4)

    # end signal for reducers
    debug_print('Rank =', rank, 'Master', 'Signal reducers end')
    for i in reducer_ranks:
        comm.send([[0]], dest=i, tag=0)

    # receive & sum
    debug_print('Rank =', rank, 'Master', 'Reducing')
    all_partm = []
    for i in reducer_ranks:
        partm = comm.recv(source=i, tag=3)
        if len(partm) > 0:
            all_partm.append(partm)
    res = sum_partial_matrices(all_partm)
    debug_print('Rank =', rank, 'Master', 'DONE')
    return res


def mapper_op():
    cnt = 0
    status = MPI.Status()
    while True:
        debug_print('Rank =', rank, 'Mapper', 'Receiving')
        subm = comm.recv(source=0, status=status)
        if status.Get_tag() == 0:
            # end signal
            comm.send([[0]], dest=0, tag=4)
            debug_print('Rank =', rank, 'Mapper', 'DONE')
            return
        elif status.Get_tag() == 1:
            debug_print('Rank =', rank, 'Mapper', 'Multiplying')
            # send for reduction
            partm = multiply_rows(subm)
            comm.send(partm, dest=reducer_ranks[cnt % len(reducer_ranks)], tag=2)
            cnt += 1


def reducer_op():
    status = MPI.Status()
    all_partm = []
    while True:
        debug_print('Rank =', rank, 'Reducer', 'Receiving')
        partm = comm.recv(status=status)
        if status.Get_tag() == 0:
            break
        elif status.Get_tag() == 2:
            debug_print('Rank =', rank, 'Reducer', 'Collecting')
            all_partm.append(partm)

    # do reduction
    debug_print('Rank =', rank, 'Reducer', 'Reducing')
    partm_sum = sum_partial_matrices(all_partm)
    # send the results back to master
    comm.send(partm_sum, dest=0, tag=3)
    debug_print('Rank =', rank, 'Reducer', 'DONE')
    return


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

    mapper_ranks = range(5, 16)
    reducer_ranks = range(1, 5)

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read matrix up to Nx10
    matrix = []
    with open(pargs.data_in, 'r') as f:
        for i, ln in enumerate(f):
            if i >= int(pargs.n):
                break
            matrix.append([float(j) for j in ln.split(' ') if j.strip()])

    debug_print('Rank =', rank, 'Node name =', MPI.Get_processor_name())
    if rank == 0:
        # Master
        print("=== COMPUTATION STARTS ===")
        print("==> Input file:", pargs.data_in)
        print("==> n =", pargs.n)
        print("==> k =", pargs.k)
        t1 = time.time()
        out_mat = master_op()
        t2 = time.time()
        print("=== COMPUTATION COMPLETE ===")
        # Write results to file
        with open(pargs.data_out, 'w') as f:
            for row in out_mat:
                for elem in row:
                    f.write('%.6f ' % elem)
                f.write('\n')
        print("==> Result matrix written to:", pargs.data_out)
        print('==> Time elapsed = %.4f s' % (t2 - t1))
        print("=== PROGRAM ENDS ===")
    elif rank in reducer_ranks:
        # Reducers
        reducer_op()
    elif rank in mapper_ranks:
        # Mappers
        mapper_op()
