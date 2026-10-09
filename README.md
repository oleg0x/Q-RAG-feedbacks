## Q-RAG with different rewards

**Q-RAG** is a resource-efficient method for **multi-step retrieval** trained with reinforcement learning directly in the latent space of text-chunk embeddings. Instead of expensive LLM fine-tuning, Q-RAG trains only a lightweight embedder agent using value-based RL (temporal difference learning), keeping the LLM frozen. This repository provides the full training and evaluation code to reproduce the results from the paper.

We consider the following rewards:
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
pip install vllm==0.18.0  # pulls compatible PyTorch, Transformers, Triton, etc.
pip install hydra-core==1.3.2 tensorboard==2.20.0 rotary-embedding-torch==0.8.9 pandas==3.0.1 nltk==3.9.4 sortedcontainers==2.4.0 accelerate==1.13.0 datasets==4.8.4
```

## Configs

The hyperparameters are set in `configs/`. Useful files are:
* `training.yaml`
* `algo/pqn_e5_hotpotqa.yaml`, `algo/pqn_gte.yaml`
* `envs/hotpotqa.yaml`, `configs/envs/2wiki.yaml`
* `feedback/defaults.yaml`, `feedback/llm.yaml`

## Training 

Here is how to train Q-RAG with each reward on HotPotQA.
Modify `configs/envs/hotpotqa.yaml` to specify your `data_path`.

### Start vLLM

Before training Q-RAG with different rewards, start vLLM server with the command like this:
```bash
CUDA_VISIBLE_DEVICES=1 vllm serve ~/Qwen/Qwen3-4B \
--served-model-name "Qwen3-4B" \
--host 0.0.0.0 \
--port 9000 \
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
Modify `configs/feedback/llm.yaml` to specify your `model`, port (in `base_url`) and `api_key`.

### LLM-as-a-Judge

```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=llm \
  feedback.llm_answer.reward_type=llm-judge
```
Modify `configs/feedback/llm.yaml` to specify your `model`, port (in `base_url`) and `api_key`.

### Information Gain

```bash
python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=gold_shift \
```
Modify `configs/feedback/gold_shift.yaml` to specify your `model`, port (in `base_url`) and `api_key`.

### Group Information Gain

```bash
python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa_candidate \
  feedback=candidate_beta \
```
Modify `configs/feedback/candidate_beta.yaml` to specify your `model`, port (in `base_url`) and `api_key`.

### Semantic Entropy Probes

Train SEP model first:
```bash
python train_semantic_entropy_probe.py \
--dataset hotpotqa \
--dataset_path ../datasets/hotpotqa \
--model_name Qwen/Qwen3-4B \
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
Then train Q-RAG:
```bash
python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=default \
  feedback.type=sep \
  feedback.sep_probe_path=runs/SEP_models/sep_probe_hotpot_2000_qwen3.pkl \
  feedback.sep_model_name=Qwen3-4B
```
Modify parameters in configs/feedback/defaults.yaml, id needed

### Synthetic Semantic Information Gain

```bash
CUDA_VISIBLE_DEVICES=0 python train_q_rag.py \
  algo=pqn_e5_hotpotqa \
  envs=hotpotqa \
  feedback=default \
  feedback.type=info_gain \
  feedback.task=HotPotQA 
```
Modify parameters in configs/feedback/defaults.yaml, id needed








