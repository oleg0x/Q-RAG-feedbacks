# Q-RAG training package for Full-Wiki

This directory is a fork of the Q-RAG code in which the Q-RAG state tower is
trained with PQN as an iterative dense retriever over all 21,015,324 Wiki-18
passages ("direct search", called line A in config and run names). Data,
reward server, launch commands and evaluation are described in the root
[README.md](../README.md), section 2.

## Entry point

```bash
cd Q-RAG_for_full-wiki
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python train_q_rag_search.py                       # configs/training_fullwiki_search.yaml
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python train_q_rag_search.py --config-name training_fullwiki_rawmix.yaml
```

Use `train_q_rag_search.py`, not `train_q_rag.py`: in direct search the
actions come from a search `s @ M.T` over the whole corpus rather than from a
candidate list in the dataset, so the objects are assembled differently (the
embedding matrix resident on the GPU, the title table for the per-title quota,
the corpus reader, batched environments in `envs/parallel_search_env.py`).

The reader and judge used for the reward are reached through the environment
variables `VLLM_MODEL`, `VLLM_BASE_URL` and `VLLM_API_KEY`; no keys are stored
in the configs.

## Layout

| Path | Contents |
|---|---|
| `train_q_rag_search.py` | training loop for direct search |
| `envs/search_env.py`, `envs/parallel_search_env.py` | episode state, masking before top-k, per-title quota, batched search |
| `envs/action_index.py`, `envs/title_table.py`, `envs/corpus_reader.py` | embedding matrix on the GPU, row → title table, corpus access by byte offsets |
| `rl/agents/pqn.py`, `rl/q_module.py` | PQN with masked soft values, calibration head, temperature schedule |
| `rl/feedback/llm_answer.py` | reward: alias exact match, LLM judge otherwise; retries for the vLLM server |
| `configs/` | hydra configs; `training_fullwiki_search.yaml` and `training_fullwiki_rawmix.yaml` are the two used here |
| `runs/` | config snapshots and TensorBoard logs of the three trained towers (checkpoints are not in git) |
| `gig_pipeline/` | scripts that build the candidate/`q0_s1` training files |
| `tests/` | unit tests of the environment, PQN fixes, reward contract and configs |

Other scripts (`train_q_rag.py`, `eval_*.py`, BABILong/NIAH utilities) are
inherited from the original Q-RAG code for the distractor setting and are not
used by the Full-Wiki experiments. The `requirements*.txt` files in this
directory also come from the original code; the environment actually used is
the root [requirements.txt](../requirements.txt) (`pip install -e '.[train]'`
from the repository root).

## Tests

```bash
cd Q-RAG_for_full-wiki
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/ -q
```

The tests need neither a GPU nor network access. Tests that need the raw
training mix (`train_data/raw/`, rebuilt by `src/build_train_mix.py`) or saved
run payloads are skipped when those files are absent.
