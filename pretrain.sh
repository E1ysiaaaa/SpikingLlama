#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
fabric run pretrain.py --accelerator=cuda --devices=4 --devices 4 --train_data_dir data/openwebtext_processed/train --val_data_dir data/openwebtext_processed/validation --resume False --main-port 29300

