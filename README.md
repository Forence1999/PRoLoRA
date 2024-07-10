<h1 align="center">
<!-- <img src="./logo.png" width="100" alt="Symbol-LLM" /> -->
<br>
PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA
</h1>



<p align="center">
  <a href="https://arxiv.org/abs/2402.16902"><b>[📄 Paper]</b></a> •
  <a href="https://hub.docker.com/r/forence/open-instruct"><b>[🐳 Docker]</b></a> •
  <a href="https://github.com/Forence1999/open-instruct-1121"><b>[🌐 GitHub]</b></a>
  
</p>


<p align="center">
Repo for "<a href="https://arxiv.org/abs/2402.16902" target="_blank">PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA</a>"
</p>


## 🔥 News

- [2025/07/08] We make the ProLoRA repo public !
- [2024/05/16] 🔥🔥🔥 ProLoRA is accepted by ACL 2024 (main conference) !



## 💡 Abstract
With the rapid scaling of large language models (LLMs), serving numerous LoRAs concurrently has become increasingly impractical, leading to unaffordable costs and necessitating more parameter-efficient finetuning methods. In this work, we introduce Partially Rotationenhanced Low-Rank Adaptation (PRoLoRA), an intra-layer sharing mechanism comprising four essential components: broadcast reduction, rotation enhancement, partially-sharing refinement, and rectified initialization strategy. As a superset of LoRA, PRoLoRA pertains its advantages, and effectively circumvent the drawbacks of peer parameter-sharing methods with superior model capacity, practical feasibility, and broad applicability. Empirical experiments demonstrate the remarkably higher parameter efficiency of PRoLoRA in both specific parameter budget and performance target scenarios, and its scalability to larger LLMs. Notably, with one time less trainable parameters, PRoLoRA still outperforms LoRA on multiple instruction tuning datasets. Subsequently, an ablation study is conducted to validate the necessity of individual components and highlight the superiority of PRoLoRA over three potential variants. Hopefully, the conspicuously higher parameter efficiency can establish PRoLoRA as a resource-friendly alternative to LoRA.

<!-- ## 🚀 -->

## ⚙️ Environment setting
```bash
# Clone the repo to local machine
git clone https://github.com/Forence1999/open-instruct-1121.git
cd open-instruct-1121
```


### 🐳 Docker

We recommend to setup the environment with docker and we have constructed and publish a docker [image](https://hub.docker.com/r/forence/open-instruct) to setup and deploy ProLoRA.

```bash
# Pull the image from dockerhub
docker pull forence/open-instruct:v1

# Start the container, remember to replace <PROJECT_DIR> with path to project directory
docker run \
    --gpus all \
    --name prolora \
    --network=host \
    --ipc=host \
    -v <PROJECT_DIR>:/workspace \
    -it forence/open-instruct:v1 /bin/bash

cd /workspace
```


### 🐍 Conda
```bash
# Create and activate conda environment
conda create -n prolora python=3.11
conda activate prolora

# Install required dependencies
pip install -r requirements.txt
```

## 📜 Datasets
The preparation of datasets is inherited from the work ["How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources"](https://arxiv.org/abs/2306.04751)(Github repo [open-instruct](https://github.com/allenai/open-instruct.git)). Please refer to the original repo to download and process the datasets for both fine-tuning and evaluation with following scripts:

You can use the following script to download all the training data:

```bash
./scripts/prepare_train_data.sh
```

You can use the following script to download all the evaluation data:

```bash
./scripts/prepare_eval_data.sh
```

Benchmark for evaluation includes:

- [MMLU](https://github.com/hendrycks/test)
- [Grade School Math (GSM)](https://github.com/openai/grade-school-math)
- [Big-Bench Hard (BBH)](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main)
- [TydiQA](https://github.com/google-research-datasets/tydiqa)
- [Codex HumanEval](https://github.com/openai/human-eval/tree/master)


## 📃 Experiments

LLaMA series require addtional requests to download. E.g., for LLaMa 1 and 2, please refer to [Hugging Face documentation for LLaMA](https://huggingface.co/docs/transformers/model_doc/llama) for requesting the token for model access.

There are two methods to pass the access token:
1. Pass as a parameter (Recommended)
```bash
# Set the <HF_TOKEN> in the script and pass it as:
--token ${HF_TOKEN}
```
2. Pass through environment variable
```bash
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token(<HF_TOKEN>)"
```

The implementation of ProLoRA is modified from [open-instruct](https://github.com/allenai/open-instruct.git). Here's an example to run fine-tuning on LLaMA2-7B with SuperNI and evaluation on MMLU. The running script is as follows:

```bash
# Before running the following script, please replace the <HF_TOKEN> with your own huggingface token
bash ft_llama2_7b_superni_mmlu.sh <LORA_RANK> <UNSHARED_RANK> <REDUCED_LORA_A_X> <REDUCED_LORA_B_X> <LEARNING_RATE> <SEED> <GPU_ID>
```

Here's a detailed description for each parameter:

Here's a detailed description for each parameter:

- <span style="background-color: rgba(240, 247, 255, 0.65); border-radius: 10px; padding: 2px 8px; display: inline-block; color: black;">**LORA_RANK**</span>: The intrinsic rank of LoRA used for Parameter Efficient Fine-Tuning.

- <span style="background-color: rgba(240, 247, 255, 0.65); border-radius: 10px; padding: 2px 8px; display: inline-block;color: black;">**UNSHARED_RANK**</span>: Among all ranks in LoRA, how many ranks are unshared and preserved.

- <span style="background-color: rgba(240, 247, 255, 0.65); border-radius: 10px; padding: 2px 8px; display: inline-block;color: black;">**REDUCED_LORA_A_X / REDUCED_LORA_B_X**</span>: Multiples of LoRA_A / LoRA_B sharing.

- <span style="background-color: rgba(240, 247, 255, 0.65); border-radius: 10px; padding: 2px 8px; display: inline-block;color: black;">**LEARNING_RATE**</span>: Learning rate.

- <span style="background-color: rgba(240, 247, 255, 0.65); border-radius: 10px; padding: 2px 8px; display: inline-block;color: black;">**SEED**</span>: Random seed.

- <span style="background-color: rgba(240, 247, 255, 0.65); border-radius: 10px; padding: 2px 8px; display: inline-block;color: black;">**GPU_ID**</span>: The id of GPU assigned for the work.



<!-- ## 🔧 Repo Structure
This repo contains the training scripts and the demo deployment. Detailed structure is as follow:
```
.
├── README.md
├── logo.png
├── demo-webui
``` -->

## © Citation
If you find it helpful, please kindly cite the paper.
```
@article{wang2024prolora,
      title={PRoLoRA: Partial Rotation Empowers More Parameter-Efficient LoRA}, 
      author={Sheng Wang and Boyang Xue and Jiacheng Ye and Jiyue Jiang and Liheng Chen and Lingpeng Kong and Chuan Wu},
      year={2024},
      eprint={2402.16902},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.16902}, 
}
```