export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/prepare_slimpajama.py --split train --percentage 0.25
python3 scripts/prepare_slimpajama.py --split validation --percentage 0.50
