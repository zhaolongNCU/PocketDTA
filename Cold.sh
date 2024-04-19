#!/bin/bash
for seed in 0 1 2 3 4; do
    python main.py --use-cold-spilt True --cold Drug --r $seed
done
for seed in 0 1 2 3 4; do
    python main.py --use-cold-spilt True --cold target_key --r $seed
done
for seed in 0 1 2 3 4; do
    python main.py --use-cold-spilt True --r $seed
done