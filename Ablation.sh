#!/bin/bash
for seed in 0 1 2 3 4; do
    python main.py --use-drug-seq False --r $seed
done
for seed in 0 1 2 3 4; do
    python main.py --use-drug-struc False --use-target-struc False --r $seed
done
for seed in 0 1 2 3 4; do
    python main.py --use-target-seq False --r $seed
done
for seed in 0 1 2 3 4; do
    python main.py --use-drug-struc False --r $seed
done