#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
fabric run model --accelerator=cuda --devices=4 pretrain.py --devices 4 --train_data_dir data/openwebtext_processed/train --val_data_dir data/openwebtext_processed/validation --resume False

