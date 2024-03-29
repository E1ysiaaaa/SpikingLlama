#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
lightning run model --accelerator=cuda --devices=7 pretrain.py --devices 7 --train_data_dir data/slimpajama --val_data_dir data/slimpajama --resume True

