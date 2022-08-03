#!/bin/bash

if [ "$1" == "--cloud" ]; then
    source install.sh
fi
python experiment.py --iter=1000 > out 2> err
rm err
python parse.py > output