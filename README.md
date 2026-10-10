## Q-RAG with different rewards

**Q-RAG** is a resource-efficient method for **multi-step retrieval** trained with reinforcement learning directly in the latent space of text-chunk embeddings. Instead of expensive LLM fine-tuning, Q-RAG trains only a lightweight embedder agent using value-based RL (temporal difference learning), keeping the LLM frozen. This repository provides the full training and evaluation code to reproduce the results from the paper.

Supported rewards:
* Exact Match
* LLM-as-a-Judge
* Information Gain
* Group Information Gain
* Semantic Entropy Probes
* Synthetic Semantic Information Gain

## Installation

```bash
# Create conda environment
conda create -n qrag python=3.12 -y
conda activate qrag

# Install dependencies
python -m pip install pip==26.0.1 wheel==0.46.3
pip install vllm==0.18.0
pip install hydra-core==1.3.2 tensorboard==2.20.0 rotary-embedding-torch==0.8.9 pandas==3.0.1 nltk==3.9.4 sortedcontainers==2.4.0 accelerate==1.13.0 datasets==4.8.4
```
Note: vllm pulls in a compatible PyTorch, Transformers, Triton, etc. Install it before the remaining dependencies.

## Configs

Hyperparameters are set in `configs/`. Useful files are:
* `training.yaml` -- top-level training configuration
* `algo/pqn_e5_hotpotqa.yaml`, `algo/pqn_gte.yaml` -- algorithm/agent configs
* `envs/hotpotqa.yaml`, `configs/envs/2wiki.yaml` -- environment configs
* `feedback/defaults.yaml`, `feedback/llm.yaml` -- reward/feedback configs

Before training, set your dataset path in `configs/envs/hotpotqa.yaml` (data_path).

## Training

The examples below show how to train Q-RAG with each reward on **HotPotQA**.

### Start vLLM

Start a vLLM server before training with any LLM-based reward:
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve ~/Qwen/Qwen3-4B \
--served-model-name "Qwen3-4B" \
--host 0.0.0.0 \
--port 9100 \
--tensor-parallel-size 1 \
--gpu-memory-utilization 0.3
```

### Exact Match

```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=llm \
  feedback.llm_answer.reward_type=exact-match
```
Update `configs/feedback/llm.yaml` to specify your `model`, port (in `base_url`), and `api_key`.

### LLM-as-a-Judge

```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=llm \
  feedback.llm_answer.reward_type=llm-judge
```
Update `configs/feedback/llm.yaml` to specify your `model`, port (in `base_url`), and `api_key`.

### Information Gain

```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=gold_shift
```
Modify `configs/feedback/gold_shift.yaml` to specify your `model`, port (in `base_url`), and `api_key`.

### Group Information Gain

Download this file:

https://huggingface.co/datasets/Q-RAG/Clear_2wiki_hotpot/blob/main/hotpot_candidate_train_q0_s1.jsonl

Set its path in `configs/envs/hotpotqa_candidate.yaml`.

Launch training:
```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa_candidate \
  feedback=candidate_beta
```
Update `configs/feedback/candidate_beta.yaml` to specify your `model`, port (in `base_url`), and `api_key`.

### Semantic Entropy Probes

Step 1 -- Train the SEP model:
```bash
CUDA_VISIBLE_DEVICES=0 python train_semantic_entropy_probe.py \
--dataset hotpotqa \
--dataset_path ../Datasets/Hotpotqa \
--model_name ../Qwen/Qwen3-4B \
--split train  \
--max_samples 2000  \
--n_samples 10 \
--temperature 1.0 \
--max_new_tokens 128   \
--position tbg \
--nli_device cuda   \
--output runs/SEP_models/sep_probe_hotpot_2000_qwen3.pkl  \
--save_dataset runs/SEP_models/sep_dataset_hotpot_2000_qwen3.npz
```

Step 2 -- Train Q-RAG with the SEP reward:
```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=defaults \
  feedback.type=sep \
  feedback.sep_probe_path=runs/SEP_models/sep_probe_hotpot_2000_qwen3.pkl
```
Adjust parameters in `configs/feedback/defaults.yaml` as needed.

### Synthetic Semantic Information Gain

```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=defaults \
  feedback.type=info_gain \
  feedback.task=HotPotQA 
```
Adjust parameters in `configs/feedback/defaults.yaml` as needed.

## Evaluation

Evaluation consists of two stages.

Stage 1 -- Evaluate the retriever:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_retriever.py \
  pretrained_path=runs/Oct10_06-53-11_QRAG_HotPotQA
```
Replace pretrained_path with the checkpoint produced by your Q-RAG training run.

Stage 2 -- Evaluate the end-to-end open-QA pipeline:
```bash
CUDA_VISIBLE_DEVICES=0 python eval_llm_openqa.py \
  --llm_name=../Qwen/Qwen3-4B \
  --llm_judge \
  --retriever_logfile=runs/Oct10_06-53-11_QRAG_HotPotQA/eval_seed42_ns50.jsonl
```
Replace `--retriever_logfile` with the log file produced in Stage 1.
