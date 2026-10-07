# Q-RAG in the Full-Wiki setting

This repository contains the code behind the Full-Wiki experiments: a Q-RAG
state tower trained with PQN to act as an **iterative dense retriever over all
21,015,324 passages of Wiki-18**, and the evaluation pipeline that produces the
reported answer exact match (EM).

How retrieval works at inference time (`src/fullwiki_qrag.py search`):

1. The state is `question [SEP] chunk_1 [SEP] ...`; the state tower embeds it.
2. The state vector itself is the query: exact inner-product search over the
   GTE embedding matrix `M` (21,015,324 × 768, fp32, 60.1 GiB, kept on the GPU).
   Already selected chunks and titles that have used up their quota of `N`
   chunks are masked *before* top-k.
3. `Q(s, a) = s · M[a]` over the top-100 candidates; the best chunk is appended
   to the state. The action side is the frozen stock GTE, so the search index
   and the action embeddings are the same matrix.
4. After 6 steps the selected chunks go to the reader (Qwen3-4B served by
   vLLM). The main metric is EM against all accepted answer aliases
   (`em_alias`), the definition used by Search-R1.

Running the same code without a checkpoint gives the untrained tower, i.e.
stock `gte-multilingual-base` doing the same iterative search ("GTE
zero-shot").

Contents:

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [1. Reproducing the inference results](#1-reproducing-the-inference-results)
- [2. Training](#2-training)
- [Tests](#tests)

## Repository layout

```text
.
├── exp.py                     experiment driver: config → run directory → retrieve → judge → score
├── Makefile                   shortcuts: make dry / experiment / test / report
├── configs/                   evaluation configs (YAML) and training config snapshots
│   └── configs_history/       all other evaluation configs, incl. the Search-R1 benchmark configs
├── src/                       pipeline modules
│   ├── fullwiki_qrag.py         offsets, reranking and direct search over Wiki-18
│   ├── answer_judge.py          reader + LLM judge (OpenAI-compatible vLLM client)
│   ├── build_index_wiki_gte.py  GTE embeddings of the corpus (+ FAISS FlatIP index)
│   ├── build_title_table.py     row → title table used by the per-title quota
│   ├── build_searchr1_eval.py   Search-R1 test.parquet → seven evaluation JSONL files
│   ├── build_musique_eval.py    MuSiQue-Ans → HotpotQA schema
│   ├── build_train_mix.py       raw NQ + HotpotQA training mix (optional training variant)
│   ├── runlib.py                run registry: paths, manifests, metrics
│   └── ...                      reports, diagnostics, oracle/variant builders
├── tests/                     pytest suite for the pipeline (no GPU or network needed)
├── runs/                      registry of every evaluation run: manifest.json, metrics.json, cmd.sh
│   ├── <run>/                   runs behind the dev-split table below
│   ├── runs_history/<run>/      all other runs, incl. the Search-R1 benchmark runs
│   └── INDEX.md                 one line per run (generated)
├── train_data/                manifest and holdout of the optional raw training mix
└── Q-RAG_for_full-wiki/       training code (Q-RAG fork: envs/, rl/, hydra configs/)
    ├── train_q_rag_search.py    training entry point for direct search
    ├── configs/                 hydra configs (training_fullwiki_search.yaml, ...)
    └── runs/                    config snapshots and TensorBoard logs of the trained towers
```

Only small files are versioned. Run payloads (retrieval JSONL, reader/judge
outputs), the corpus, the index and the checkpoints are not in git; the
sections below say how to obtain or rebuild each of them.

## Requirements

**Hardware.** All numbers were produced on NVIDIA H200 (141 GB) GPUs. Direct
search keeps the whole embedding matrix (60.1 GiB) on one GPU, so the retrieval
stage needs a single GPU that holds the matrix plus the tower and a batch of
states. Training used about 72 GB of one H200. The reader/judge vLLM server
runs on a separate GPU (a fraction of an H200 is enough for Qwen3-4B).

**Disk.** Wiki-18 corpus 14.4 GB; GTE embedding shards 60.1 GiB; the FAISS
FlatIP file written by the index builder 64.6 GB (direct search does not read
it, so it can be deleted afterwards); offsets and title tables < 0.5 GB; one
training checkpoint 7.3 GB.

**Software.** Python 3.11, PyTorch 2.10 with CUDA, Docker for the vLLM server.
The exact package versions are in [requirements.txt](requirements.txt).

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA build
pip install -e '.[index,dev]'    # evaluation pipeline, index builder, tests
pip install -e '.[train]'        # extra packages for training (section 2)
```

All pipeline stages run with `HF_HUB_OFFLINE=1`, so download the encoder once
while online. The GTE checkpoint uses remote code (`Alibaba-NLP/new-impl`),
which is fetched by the same call:

```bash
python - <<'EOF'
from transformers import AutoModel, AutoTokenizer
for revision in ("9bbca17d9273fd0d03d5725c7a4b0f6b45142062", "main"):
    AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", revision=revision)
    AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", revision=revision,
                              trust_remote_code=True)
EOF
```

Evaluation pins revision `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`; training
loads `main`, which resolved to the same commit when the towers were trained.

**Paths.** Configs refer to external data through the placeholder
`/path/to/data`. Paths inside the repository are relative and resolved from the
repository root (always start `make`/`exp.py` there). Replace the placeholder
once:

```bash
DATA=/your/data/root
grep -rl /path/to/data configs Q-RAG_for_full-wiki/configs | xargs sed -i "s|/path/to/data|$DATA|g"
```

`retrieve.cuda_visible_devices` in each evaluation config pins the GPU used
for retrieval; set it to a free GPU on your machine. Paths that still contain
`/path/to/...` after this step belong to legacy configs and run records and
are not needed for the results below.

## 1. Reproducing the inference results

### 1.1 Data

Expected layout under `$DATA` (the names match the configs):

```text
$DATA/
├── full-wiki/
│   ├── data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl   Wiki-18 corpus
│   └── wiki18-gte/                                                        built in 1.2
├── hotpotqa/hotpot_dev_fullwiki_v1.json                                   HotpotQA dev (fullwiki)
├── 2WikiMultiHopQA/data_ids_april7/dev.json                               2WikiMultiHopQA dev
└── musique/musique_ans_v1.0_dev.jsonl                                     MuSiQue-Ans dev
```

**Corpus.** Wiki-18 in 100-word passages, as distributed with Search-R1
(`PeterJinGo/wiki-18-corpus` on Hugging Face). Despite its name,
`wiki-18.jsonl.gz` is a gzipped tar archive that unpacks into the path above:

```bash
huggingface-cli download PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz --repo-type dataset --local-dir $DATA/full-wiki
tar -xzf $DATA/full-wiki/wiki-18.jsonl.gz -C $DATA/full-wiki
sha256sum $DATA/full-wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
# 43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d  (21,015,324 lines, 14,393,573,105 bytes)
```

Row `i` of the file has `"id": "i"`; FAISS ids, embedding-shard rows and corpus
rows all refer to the same passage, so do not reorder or filter the file.

**Search-R1 benchmarks (main table).** The seven evaluation splits used by
Search-R1 are a single `test.parquet` in `PeterJinGo/nq_hotpotqa_train`. The
converter writes one JSONL per dataset, with the first golden answer in
`answer` and the rest as aliases:

```bash
huggingface-cli download PeterJinGo/nq_hotpotqa_train test.parquet --repo-type dataset \
    --revision b7d80abfee334a7a91cb377544f09180d58b34f6 --local-dir runs/shared/searchr1
python src/build_searchr1_eval.py --input runs/shared/searchr1/test.parquet --output-dir runs/shared/searchr1
```

| File in `runs/shared/searchr1/` | Questions | SHA-256 |
|---|---:|---|
| `test.parquet` (input) | — | `30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc` |
| `nq.jsonl` | 3,610 | `655f62f00b64f0ec49d734a5f60c146aaf1968f420e9f2c7c6701e5cca4e903a` |
| `triviaqa.jsonl` | 11,313 | `7dc518b458bbcc798bbcc35895080cd2cb8e9d176bfe0120e3169a0e4fe5a41e` |
| `popqa.jsonl` | 14,267 | `4eda18303a72c77a5b2f69d4f2f792713eb355969f14a8a58c37695b808ca36f` |
| `hotpotqa.jsonl` | 7,405 | `ea8c6f33107557e49b0f15fe88bf9efff0e12ae116ffb704f444317c2281ccd8` |
| `2wikimultihopqa.jsonl` | 12,576 | `728afd8ef846b2c39211bea4d29bfb0a5c00dad8b8a33d88fb597a5a4e44d2cc` |
| `musique.jsonl` | 2,417 | `c3565d54d3c86adb5f54aec420db14998fb65297266793f1edb11954faaa851b` |
| `bamboogle.jsonl` | 125 | `a5542f8aa57a77d567105a6a7ec92a226dfa097c1ed91cb19457bfdef64661fa` |

**Dev splits (additional table).** The official HotpotQA
`hotpot_dev_fullwiki_v1.json` (7,405 questions, SHA-256
`2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618`), the
2WikiMultiHopQA `dev.json` (12,576; `79f77ae1…dadf5`) and the MuSiQue-Ans dev
file. MuSiQue has to be converted to the HotpotQA schema once:

```bash
python src/build_musique_eval.py --input $DATA/musique/musique_ans_v1.0_dev.jsonl \
    --output runs/shared/musique_ans_dev_hotpot_schema.jsonl
# SHA-256 a2996d330e2a82274282ad39d3664977d80b75f2d1d9db6b9d0584495ba7d5d0, 2,417 lines
```

### 1.2 Embedding matrix, offsets and title table

These are built once per corpus. The embedding step encodes all 21M passages
with `gte-multilingual-base` (fp16 inference, fp32 cache, L2-normalised, 211
shards of 100,000 rows); `--device` takes any list of GPUs and only changes
the build time.

```bash
INDEX=$DATA/full-wiki/wiki18-gte
CORPUS=$DATA/full-wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl

python src/build_index_wiki_gte.py \
  --corpus $CORPUS --output-dir $INDEX \
  --model Alibaba-NLP/gte-multilingual-base \
  --revision 9bbca17d9273fd0d03d5725c7a4b0f6b45142062 \
  --device cuda:0,cuda:1,cuda:2,cuda:3 \
  --model-dtype float16 --cache-dtype float32 \
  --max-length 256 --batch-size 256 --multi-process-chunk-size 1000 \
  --shard-size 100000 --truncation-samples-per-shard 2048 \
  --phase all --index-name gte-multilingual-base.flatip.faiss \
  --verify-samples 128 --seed 42 --trust-remote-code

# byte offsets of corpus rows, so that chunk texts are read without scanning the file
python src/fullwiki_qrag.py prepare --index-dir $INDEX

# row → title id table (int32, 84 MB) used by the per-title quota
python src/build_title_table.py --index-dir $INDEX --output $INDEX/corpus-title-ids.npy
```

The index manifest records the corpus path, size and modification time, and
the offsets check them, so do not move or touch the corpus after this step.

### 1.3 Trained checkpoint

The evaluation configs expect the trained tower at

```text
Q-RAG_for_full-wiki/runs/Aug03_12-44-38_lineA_main/model_best.pt
SHA-256 c8b2212d4b1e04449623a35acabe2ea8abc06d09acc22b6a87e29a2ea5c9db36 (7,333,936,023 bytes)
```

The checkpoint is not stored in git because of its size. Download it from
<ARTIFACTS_URL> into that directory, or train your own tower as described in
section 2 and point `retrieve.checkpoint` of the configs to it. The directory
already contains the training-time `config.yaml`, which the search code reads
next to the checkpoint. The zero-shot rows need no checkpoint.

### 1.4 Reader and judge server

The reader and the judge are the same model, Qwen3-4B (snapshot
`1cfa9a7208912126459214e8b04321603b3df60c`), served by vLLM 0.15.1 on
`127.0.0.1:8010` (the port is set in `judge.base_url` of every config):

```bash
docker run -d --name qrag-vllm-8010 --gpus device=1 --ipc=host \
  -p 127.0.0.1:8010:8010 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:v0.15.1 \
  --model Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name Qwen3-4B --host 0.0.0.0 --port 8010 \
  --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.50 \
  --max-model-len 32768 --max-num-seqs 128 --enable-prefix-caching --trust-remote-code

curl -fsS http://127.0.0.1:8010/v1/models     # must list Qwen3-4B
```

Decoding is fixed by the pipeline: thinking disabled, temperature 0, at most
1,000 tokens for the reader and 100 for the judge, all retrieved chunks in the
context. The judge stage splits the questions into 32 contiguous shards and
queries the server in parallel; this only affects speed.

### 1.5 Running an evaluation

Every evaluation is one config and one command. `make dry` prints the exact
commands without running anything; `make experiment` runs retrieve → judge →
score:

```bash
make dry        CFG=configs/configs_history/sr1_hotpotqa_best.yaml
make experiment CFG=configs/configs_history/sr1_hotpotqa_best.yaml
```

The driver creates `runs/<YYYY-MM-DD>-<slug>/` with `manifest.json`
(parameters, code and input hashes, environment), `cmd.sh` (the exact stage
commands), `log.txt`, `retrieval.jsonl`, `answer_judge.json` and
`metrics.json`, and refuses to overwrite an existing run directory
(use `TAG=...` for a second run on the same day). On one H200 plus the vLLM
server, HotpotQA (7,405 questions) takes about 6 minutes of retrieval and 3
minutes of reading and judging; PopQA (14,267) about 10 + 5 minutes.

`metrics.json` contains `em_alias` (EM against all accepted answers, the
number reported in the main table), `em` and `f1` against the first answer
only, `judge` (LLM-judge accuracy) and, where gold titles exist, title recall
metrics.

### 1.6 Main table: seven Search-R1 benchmarks

EM with answer aliases (%), Qwen3-4B reader, Wiki-18 corpus:

| Method | Configs | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q-RAG trained, 6 steps, N=2 | `sr1_<dataset>_best.yaml` | 33.41 | 57.31 | 39.85 | 37.74 | 32.29 | 11.63 | 28.80 | 34.43 |
| Q-RAG trained, 4 steps, no quota | `sr1_best-steps4-noquota_<dataset>.yaml` | 32.66 | 56.13 | 39.43 | 36.26 | 31.07 | 10.63 | 28.00 | 33.45 |
| GTE zero-shot, 6 steps, N=2 | `sr1_<dataset>_zeroshot.yaml` | 34.40 | 56.83 | 38.07 | 27.05 | 17.40 | 6.25 | 19.20 | 28.46 |

All configs are in `configs/configs_history/`; `<dataset>` is one of
`nq triviaqa popqa hotpotqa 2wiki musique bamboogle`. One trained checkpoint and
one setting are used for all seven datasets. To run a whole row:

```bash
for ds in nq triviaqa popqa hotpotqa 2wiki musique bamboogle; do
  make experiment CFG=configs/configs_history/sr1_${ds}_best.yaml
done
```

The manifests and metrics of the original runs of the first and third rows are
in `runs/runs_history/2026-08-07-sr1-{best,zeroshot}-<dataset>/`. The
`sr1_<dataset>_noctx.yaml` configs give the reader-only lower bound; they reuse
the questions of an `sr1-best` run, so set `retrieve.source_run` to the id of
your run of that config first. The Search-R1 baselines reported next to these
rows were evaluated with the Search-R1 code and are not part of this
repository.

### 1.7 Dev splits (single-answer EM)

The eight top-level configs in `configs/` produce the runs in `runs/<run>/`.
These numbers use EM against the single gold answer (`em` in `metrics.json`):

| Method | Config | HotpotQA dev (7,405) | 2Wiki dev (12,576) | MuSiQue-Ans dev (2,417) |
|---|---|---:|---:|---:|
| Reader without context | `no_retrieval.yaml` | 15.53 | — | — |
| GTE top-6, 2 chunks per title | `gte_only_n2_steps6.yaml` | 29.29 | — | — |
| GTE zero-shot, 6 steps, N=2 | `search_zeroshot_{steps6,2wiki,musique}.yaml` | 27.16 | 15.83 | 4.96 |
| Q-RAG trained, 6 steps, N=2 | `search_best_{hotpotqa,2wiki,musique}.yaml` | **37.70** | **30.31** | **10.14** |

The two reference rows (reader without context, plain GTE top-6) come from an
earlier reranking pipeline: `gte_only_n2_steps6.yaml` reads the action index of
a Q-RAG reranker trained on distractor paragraphs and the original Q-RAG code
(`/path/to/Q-RAG-feedback`), and `no_retrieval.yaml` reuses the questions of
that run. They are not needed for the Q-RAG rows. Absolute EM is not
comparable across datasets: the share of questions whose gold articles exist
in Wiki-18 is 81.67% (HotpotQA), 51.05% (2Wiki) and 72.11% (MuSiQue).

`make report` rebuilds `runs/INDEX.md` and per-dataset tables with paired
McNemar tests from the run directories; it needs the run payloads
(`answer_judge.json`), which are produced by the runs themselves.

## 2. Training

The published tower is run `Aug03_12-44-38_lineA_main` ("line A" in config
and run names): direct search trained on HotpotQA + 2WikiMultiHopQA questions.
Its configuration snapshot is [configs/lineA_main.yaml](configs/lineA_main.yaml)
(also `Q-RAG_for_full-wiki/runs/Aug03_12-44-38_lineA_main/config.yaml`,
together with its TensorBoard logs).

### 2.1 What is trained

- **State tower:** `Alibaba-NLP/gte-multilingual-base` with CLS pooling,
  trained, output not normalised (ranking by inner product does not depend on
  the query norm). A scalar calibration head maps `s · M[a]` to Q-values.
- **Action side:** frozen, bit-identical to stock GTE; action vectors are the
  rows of the same `wiki18-gte` embedding shards.
- **Environment:** 6 steps per question, top-100 candidates from a fresh
  search at every step, per-title quota N=2, masking before top-k.
- **Exploration:** Boltzmann over the tower's own top-100 with a dimensionless
  temperature (logits divided by their spread, linearly 1.0 → 0.3); with
  probability ε = 0.1 the action is drawn from the GTE top-100 of the current
  state instead.
- **Reward:** at the end of the episode the reader answers from the selected
  chunks; reward = 1 if the normalised answer matches the gold answer,
  otherwise the LLM judge's verdict. Failed reader/judge requests are retried;
  transitions that still have no reward are excluded from the loss.
- **Optimisation:** PQN, 30,000 optimiser steps, AdamW lr 5e-6 with 1,000
  warm-up steps, 16 parallel environments, batch 24 × 4 gradient-accumulation
  steps, γ = 0.99.
- **Model selection:** every 500 steps the agent is evaluated on 1,000
  HotpotQA dev questions; `model_best.pt` is the checkpoint with the best
  judge-based reward (step 12,000 for the published run), `model_last.pt` is
  the final one.

The full set of values is in
[Q-RAG_for_full-wiki/configs/training_fullwiki_search.yaml](Q-RAG_for_full-wiki/configs/training_fullwiki_search.yaml)
and the files it includes (`algo/pqn_gte.yaml`, `envs/fullwiki_search.yaml`,
`feedback/llm.yaml`).

### 2.2 Data

Training reuses the corpus, embedding shards, offsets and title table from
section 1.2 (`envs.index.shard_dir`, `envs.corpus`, `envs.offsets`,
`envs.title_table` in `envs/fullwiki_search.yaml`), plus:

- **Training questions** (`envs.train_data_path`):
  `hotpotqa_2wiki_candidate_train_q0_s1.jsonl`, 120,631 HotpotQA and
  2WikiMultiHopQA training questions, SHA-256
  `831b9007161394fdafd11e3ffdff50dc662f528d47c581f3583d24908c5e7311`,
  available from <ARTIFACTS_URL>. It is the `q0_s1` subset produced by
  `Q-RAG_for_full-wiki/gig_pipeline/` (questions the reader cannot answer
  without context but answers from the gold supporting facts). The search
  environment reads only `question`, `answer` and `supporting_facts`; the
  candidate fields used by the distractor setting are ignored.
- **Validation questions** (`envs.eval_data_path`): the directory with
  `hotpot_dev_fullwiki_v1.json` from section 1.1.

`envs/fullwiki_search.yaml` expects the training file at
`$DATA/hotpotqa_2wiki_candidate_train_q0_s1.jsonl` after the placeholder
replacement from [Requirements](#requirements); any other location can be
passed as a hydra override (below).

### 2.3 Reward server

Training queries a reader/judge server through environment variables:

```bash
export VLLM_MODEL=Qwen3-4B
export VLLM_BASE_URL=http://127.0.0.1:8010/v1
export VLLM_API_KEY=                 # empty for a local server
```

The published tower used the same Qwen3-4B snapshot and server settings as
evaluation (section 1.4). Use a separate server for training if evaluations
run at the same time.

### 2.4 Launch

```bash
cd Q-RAG_for_full-wiki
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python train_q_rag_search.py
# with a different location of the training file:
#   python train_q_rag_search.py envs.train_data_path=/other/path/hotpotqa_2wiki_candidate_train_q0_s1.jsonl
```

`train_q_rag_search.py` composes `configs/training_fullwiki_search.yaml` by
default; any other hydra override can be appended in the same way, and
`--config-name <file>` selects another top-level config. Each run writes
`Q-RAG_for_full-wiki/runs/<Mon DD_HH-MM-SS>_QRAG_<task>/` with `config.yaml`,
`tb_logs/`, `model_last.pt` and `model_best.pt`. The published run took about
37 hours on one H200 (about 72 GB of GPU memory) with the reward server on a
second GPU.

To evaluate a new tower, set `retrieve.checkpoint` in the evaluation configs
(e.g. `configs/configs_history/sr1_<dataset>_best.yaml`) to its
`model_best.pt` and run section 1.5.

### 2.5 Optional: raw NQ + HotpotQA mix

`configs/rawmix4b.yaml` and `configs/rawmix8b.yaml` are snapshots of two
additional runs trained on the raw NQ + HotpotQA mix used by Search-R1, with
Qwen3-4B or Qwen3-8B as the reward model (selected only through `VLLM_MODEL` /
`VLLM_BASE_URL`). To reproduce them, put `train.parquet` and `test.parquet`
of `PeterJinGo/nq_hotpotqa_train` (revision
`b7d80abfee334a7a91cb377544f09180d58b34f6`) into `train_data/raw/`, rebuild
the episode weights (holdout examples get weight 0) and train with the raw-mix
config:

```bash
python src/build_train_mix.py --check  # compare the parquet files with train_data/manifest.json
python src/build_train_mix.py          # writes train_data/weights/a1.jsonl
cd Q-RAG_for_full-wiki
python train_q_rag_search.py --config-name training_fullwiki_rawmix.yaml
```

## Tests

```bash
make test                                            # pipeline tests, no GPU or network
cd Q-RAG_for_full-wiki && HF_HUB_OFFLINE=1 python -m pytest tests/   # training package tests
```

In a fresh clone the expected result is `146 passed, 7 skipped` for the
pipeline and `70 passed, 7 skipped` for the training package. Skipped tests
need artifacts that are not in git (saved run payloads, the raw training mix,
or the upstream Q-RAG code used for regression checks of the vendored
prompts); each skip prints the missing file.
