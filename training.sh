#!/bin/bash
for seed in 0 1 2 3 4; do
    python main.py --task Davis --r $seed
done