#!/bin/bash
salloc -p general -J CMSC22240_A5_SN --time=0:00:30 -N 1 -n 16 mpirun -n 16 /usr/bin/python3 mm_mpi.py -n $1 -k $2
