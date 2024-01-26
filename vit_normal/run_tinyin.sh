#!/bin/bash

# Function to run a command and check its exit status
run_command() {
    $1
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
    fi
}

# Run each command, allowing the script to continue even if one fails
run_command "python train_tinyin.py --net vit_no_ln_s --n_epochs 100 --size 128 --patch 8 --lr 0.0001"
run_command "python train_tinyin.py --net vit_no_ln_with_init_s --n_epochs 100 --size 128 --patch 8 --lr 0.0001"
run_command "python train_tinyin.py --net vit_no_ln_s --n_epochs 100 --size 128 --patch 8 --lr 0.001"
run_command "python train_tinyin.py --net vit_no_ln_with_init_s --n_epochs 100 --size 128 --patch 8 --lr 0.001"
run_command "python train_tinyin.py --net vit_s --n_epochs 100 --size 128 --patch 8 --lr 0.0001"
run_command "python train_tinyin.py --net vit_ti --n_epochs 100 --size 128 --patch 8 --lr 0.0001"

