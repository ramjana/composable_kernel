#!/bin/bash

PRECISION=  ##--half

if [ $# -ge 1 ] ; then
    NREPEAT=$1
else
    NREPEAT=1
fi

Operation=4

LENGTHS=64,4,280,82

## for generic validation
for op in $Operation; do
    for use_idx in 0 1; do
        set -x
        ./bin/ckProfiler reduce $PRECISION -D 64,4,280,82 -R 0 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 4,64,280,82 -R 0 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 280,4,64,82 -R 0 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 64,4,280,82  -R 0,1,2 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 4,64,280,82  -R 0,1,2 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 64,280,82,4  -R 0,1,2 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 700,8192  -R 1 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 700,1024  -R 1 -O $op $CTYPE -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 700,4  -R 1 -O $op $CTYPE -v 1 1 $NREPEAT
        set +x
    done
done

## for performance evaluation (resnet50 NHWC => C)
for op in $Operation; do
    for use_idx in 0 1; do
        set -x
        ./bin/ckProfiler reduce $PRECISION -D 256,14,14,1024 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,28,28,128 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,58,58,128 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,7,7,2048 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,14,14,256 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,30,30,256 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,56,56,256 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,16,16,512 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,28,28,512 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,7,7,512 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,56,56,64 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 256,230,230,3 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,14,14,1024 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,28,28,128 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,58,58,128 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,7,7,2048 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,14,14,256 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,30,30,256 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,56,56,256 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,16,16,512 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,28,28,512 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,7,7,512 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        ./bin/ckProfiler reduce $PRECISION -D 128,56,56,64 -R 0,1,2 -O $op -I $use_idx  -v 1 1 $NREPEAT
        set +x
    done
done 

