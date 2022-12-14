#!/bin/bash

if [ "$1" == "--cloud" ]; then
    source install.sh --cuda
fi

if [ "$2" == "--expe" ]; then
    python experiment.py --iter=1000 > out 2> err
    rm err
    python parse.py > output
fi

if [ "$2" == "--idx" ]; then
    for i in 1 2 4 8 16 32
    do
        OMP_NUM_THREADS=$i python experiment.py --iter=1000 > out$i 2> err
    done
    rm err
    python parse.py > output$3
    for i in 1 2 4 8 16 32
    do
        rm out$i
    done
fi
