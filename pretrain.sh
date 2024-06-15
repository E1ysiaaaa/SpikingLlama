#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
fabric run model --accelerator=cuda --devices=6 pretrain.py --devices 6 --train_data_dir /data4/slim_star_combined --val_data_dir /data4/slim_star_combined --resume True

