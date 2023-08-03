#!/bin/sh

mt="mlp"
output="./checkpoints"

venv/bin/python train.py --mapping_type=$mt --out_dir=$output
