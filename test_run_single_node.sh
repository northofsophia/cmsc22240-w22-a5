#!/bin/bash
salloc -p debug -J CMCS22240_A5_SNT --time=0:00:15 -N 1 -n $2 mpirun -n $2 /usr/bin/python3 mm_mpi.py -n $1 -k $2
