#!/bin/bash
salloc -p general -J CMSC22240_A5_MN --time=0:00:15 -N 4 -n 4 --exclusive mpirun -n 4 /usr/bin/python3 mm_mpi.py --multi_node -n $1 -k $2
