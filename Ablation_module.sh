#!/bin/bash
for seed in 0 1 2 3 4; do
    python main.py --model PredictorGraphMVPAblation --r $seed
done
for seed in 0 1 2 3 4; do
    python main.py --model PredictorBANAblation --r $seed
done