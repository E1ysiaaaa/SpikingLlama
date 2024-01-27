#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
lightning run model --accelerator=cuda --devices=8 pretrain.py --devices 8 --train_data_dir data/bookcorpus --val_data_dir data/bookcorpus --resume True

