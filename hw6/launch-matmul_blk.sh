srun -u -t 10 -p pascal -G 1 -c 4 --mem-per-cpu=2048 python3 matmul_blk.py 2000 $1
