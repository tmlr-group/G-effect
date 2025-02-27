# Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond

This is the code for the paper [**Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond**](https://www.arxiv.org/abs/2502.19301)

## Installation

```
conda create -n unlearning python=3.10
conda activate unlearning
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
## Loading the Dataset

To load the dataset, use the following code:

```python
from datasets import load_dataset
dataset = load_dataset("locuslab/TOFU","full")
```

## Finetune your models

The code currently supports `Phi-1.5`, and `Llama2-7b chat` models. But newer models can directly be added in the `model_config.yaml` file. For the unlearning challenege, we fine-tuned `Phi-1.5` for 5 epochs using a maximum learning rate of `2e-5`, and the `Llama2-7b chat` model for the same duration at `1e-5`. Finetuning can be done as follows:

```
master_port=18765
split=full
model=phi #you can choose phi, llama2-7b
lr=2e-5
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port finetune.py --config-name=finetune.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
```

## Forget Models

Before running the code, ensure that you have some pretrained models, which can be achieved by following the guidelines in the original TOFU repository. Additionally, adjust the `model_path` and `save_dir` (default is `icml/`) for the YAML files in the `config/` directory (e.g., `config/forget.yaml` and `config/forget_ge.yaml`).

### Running Baseline Methods:
Execute the script `basher1.py` to run the original unlearning methods with various checkpoints, which automatically saves the results into the folder named `icml`.

```bash
python basher1.py ga --model=llama --cuda_id=3 --setting=forget05 --hyper=2 
```

### Computing G-effects:
Use `basher2.py` to calculate the g-effects for the saved checkpoints, and save the results to a specific file.

```bash
python basher2.py ga --model=llama --cuda_id=3 --setting=forget05 --hyper=2 > ga_ge_log.txt
```

### Current Supported Methods:
- `ga` (Gradient Ascent)
- `npo` 
- `ins_npo` 
- `w_ins_npo` 
- `wga` 
- `rmu_x` (Particular Layer To Be Perturbed, e.g., `rmu_32`, `rmu_21`, `rmu_10`)
- `idk` (referred to as `po`)

### Supported Models:
- `llama`
- `phi`

### Supported Settings:
- `forget01`: Forgetting 1% of the original dataset
- `forget05`: Forgetting 5% of the original dataset
- `forget10`: Forgetting 10% of the original dataset


## Evaluate models
Once you have the model trained, you can generate the PS-series metrics used for evaluation with the following command:

```
cuda_id=0
master_port=18765
model=phi #you can choose phi, llama2-7b
ckpt=baseline/llama2-7b/grad_ascent_1e-05_forget05_8_0.0_250/checkpoint-125  # where the checkpoint is stored
CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$master_port evaluation_everything.py split=${split} model_family=${model} model_path=${ckpt}
```
For TOFU metrics used for evaluation with the following command:

```
cuda_id=0
master_port=18765
model=phi #you can choose phi, llama2-7b
ckpt=baseline/llama2-7b/grad_ascent_1e-05_forget05_8_0.0_250/checkpoint-125  # where the checkpoint is stored
CUDA_VISIBLE_DEVICES=$cuda_id torchrun --nproc_per_node=1 --master_port=$master_port tofu_evaluation.py split=${split} model_family=${model} model_path=${ckpt}
```


### Available forget sets are:

- `forget01`: Forgetting 1% of the original dataset, all entries correspond to a single author.
- `forget05`: Forgetting 5% of the original dataset, all entries correspond to a single author.
- `forget10`: Forgetting 10% of the original dataset, all entries correspond to a single author.

Retain sets corresponding to each forget set are also available, which can be used to train an Oracle model.

## Citing Our Work

If you find our metrics beneficial, please cite our work:
```
@inproceedings{wang2025rethinking,
title={Rethinking LLM Unlearning Objectives: A Gradient Perspective and Go Beyond}, 
author={Qizhou Wang and Jin Peng Zhou and Zhanke Zhou and Saebyeol Shin and Bo Han and Kilian Q Weinberger},
booktitle = {International Conference on Learning Representations},
year = {2025}
}
```

