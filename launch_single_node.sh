#!/bin/bash
salloc -p general -J CMSC22240_A5_SN --time=0:00:30 -N 1 -n $2 mpirun -n $2 /usr/bin/python3 mm_mpi.py -n $1 -k $2
