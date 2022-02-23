#!/bin/bash
salloc -p general -J CMCS22240_A5_MNT --time=0:00:30 -N 4 -n 32 mpirun -n 32 /usr/bin/python3 mm_mpi.py -n $1 -k $2 --multi_node
