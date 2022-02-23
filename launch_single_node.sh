#!/bin/bash
salloc -p general -J CMSC22240_A5_SN --time=0:00:15 -N 1 -n 16 --exclusive mpirun -n 16 /usr/bin/python3 mm_mpi.py -n $1 -k $2
