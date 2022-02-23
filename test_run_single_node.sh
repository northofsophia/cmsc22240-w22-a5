#!/bin/bash
salloc -p debug -J CMCS22240_A5_SNT --time=0:00:30 -N 1 -n 16 mpirun -n 16 /usr/bin/python3 mm_mpi.py -n $1 -k $2
