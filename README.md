# Auto Quantization Encoder
This repository aims to use a fully connected network and a heave side function to implement a 1-bit quantization for all the input vectors before a linear layer in a GPT model. The feasibilty of this method is still under investigation. Our network architecture, training spefications and codes are based on [TinyLlama](https://github.com/jzhang38/TinyLlama).

## Datasets
We basically follow the data preprocessing in TinyLlama, the full SlimPajama dataset and its preprocessed .bin files are stored on `/data4` directory of the 22 server. (.bin files are stored in `/data4/slim_star_combined`) However, currently we use the data in `./data/slimpajama` for training, it is only a subset of the big dataset in `/data4/slim_star_combined`. After data preprocessing is finished, we will use the bigger dataset instead.

The data preprocessing scripts are stored in `scripts`, and to produce the training and validation .bin files from SlimPajama, you can run 

`	`./prepare_data.sh

But since I've done this, it is not necessary now.

## Training
We use the pretrained 1B weights from TinyLlama as our base model, and do a layerwise finetuning and knowledge distillation to obtain the auto quantization encoder for the model. To run the training process, simply run

	./pretrain.sh

The output checkpoints are stored in the `out` directory, along with the pretrained weights from TinyLlama(named `teacher.pth`). 

For details of training and methods, you can refer to the codes in `pretrain.py` and `src/quant_model.py`.

## Evaluation
To evaluate the model, you can use `eval.py`, but be aware to change the model and paths in the python scripts to make sure you are evaluating the model you want. Currently `eval.py` loads the pretrained TinyLlama model and pretrained 1B weights. To do the evaluation, you can run

	export HF_ENDPOINT=https://hf-mirror.com    # huggingface is inaccessible in China.
	# choose the test set you need
	python3 eval.py --model spikellama --tasks=piqa,lambada,arc_easy,winogrande


