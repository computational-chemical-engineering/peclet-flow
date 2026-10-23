#!/bin/bash
# usage: runall.sh BIN TAG LEVEL TOL THREADS  -> runs/<TAG>/{log.dat,err.log,fields_N*.bin}; prints summary
B=$(realpath $1); D=runs/$2; mkdir -p $D; cd $D
OMP_NUM_THREADS=$5 OMP_PROC_BIND=false $B $3 $4 > log.dat 2> err.log; rc=$?
cd ../..; echo "$2 rc=$rc $(python post.py $D)"
