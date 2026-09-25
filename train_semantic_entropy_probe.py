"""
Semantic Entropy Probes (SEP) — Training & Inference Pipeline
=============================================================

Based on: "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs"
          (Kossen et al., arXiv:2406.15927)

Pipeline:
  1. Load questions from HotPotQA, MuSiQue, 2WikiMultiHopQA, or BAbILong datasets
  2. For each question: greedy generation → extract hidden states h_p^l(x)
  3. Sample N=10 answers at T=1.0 → cluster via NLI → compute H_SE(x)
  4. Binarize H_SE with optimal two-group-MSE threshold γ*
  5. Train logistic regression probe on (hidden_state, binary_label) pairs
  6. Export the probe for downstream use (e.g. as a Q-RAG reward signal)

Usage examples:
  # BAbILong QA3
  python train_semantic_entropy_probe.py \
      --dataset babilong \
      --dataset_path ../datasets/babilong/tasks_1-20_v1-2/en-10k/qa3_three-supporting-facts_train.txt \
      --model_name Qwen/Qwen3-4B \
      --max_samples 2000 --output runs/SEP_models/sep_probe_babilong_2000_qwen3_4b_new.pkl

  # HotPotQA
  python train_semantic_entropy_probe.py \
      --dataset hotpotqa \
      --dataset_path ../datasets/hotpotqa \
      --model_name Qwen/Qwen2.5-1.5B-Instruct \
      --max_samples 500 --output runs/SEP_models/sep_probe.pkl

  # MuSiQue (answerable JSONL; same question/context shape as Hotpot eval)
  python train_semantic_entropy_probe.py \
      --dataset musique \
      --dataset_path ../datasets/musique/musique_ans_v1.0_dev.jsonl \
      --musique_n_refs_exactly 2 \
      --model_name Qwen/Qwen3-4B \
      --max_samples 500 --output runs/SEP_models/sep_probe_musique.pkl

  # 2WikiMultiHopQA (HotpotQA-style JSON: train.json / dev.json / test.json)
  python train_semantic_entropy_probe.py \
      --dataset 2wiki \
      --dataset_path ../datasets/2WikiMultiHopQA \
      --split train \
      --model_name Qwen/Qwen3-4B \
      --max_samples 2000 --output runs/SEP_models/sep_probe_2wiki.pkl

  # MATH (ICL: few-shot examples sampled like envs/qa_dataset_adapter + RetrievalMATH)
  python train_semantic_entropy_probe.py \
      --dataset math \
      --dataset_path ../datasets/MATH \
      --math_samples_num 10000 --math_examples_num 1000 \
      --math_n_icl_chunks 2 \
      --model_name Qwen/Qwen3-4B \
      --max_samples 500 --output runs/SEP_models/sep_probe_math.pkl

  # Re-train from a saved (X, se_values) .npz — skips LLM / NLI dataset construction
  python train_semantic_entropy_probe.py \
      --load_dataset runs/SEP_models/sep_probe_musique_2000_last.npz \
      --max_samples 500 \
      --model_name Qwen/Qwen3-4B \
      --layers 31 32 33 34 35 \
      --output runs/SEP_models/sep_probe_musique_500_last.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

# Re-use MuSiQue loader from eval helpers (same chunk / ref semantics as SEP eval).
_SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from scripts.eval_dataset_utils import MusiqueQADataset  # noqa: E402

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from prompts_and_metrics import prompts
from rl.feedback.llm_answer import prepare_examples
from envs.dataloaders.competition_math import RetrievalMATH


# ── Dataset loaders ─────────────────────────────────────────────────────


def _is_bridge_comparison(qtype: Optional[str]) -> bool:
    """True for 2Wiki types like ``bridge comparison``, ``bridge_comparison``, etc."""
    t = (qtype or "").lower().replace("-", " ").replace("_", " ")
    return t == "bridge comparison"


def _2wiki_chunks_and_sf_idx(item: dict) -> Tuple[List[str], List[int]]:
    """Paragraph-level chunks matching ``QADatasetAdapter`` / RL env."""
    sp_title_set = {sup[0] for sup in item.get("supporting_facts", [])}
    chunks: List[str] = []
    sf_idx: List[int] = []
    for idx, (title, sentences) in enumerate(item["context"]):
        chunks.append(title + " " + " ".join(sentences))
        if title in sp_title_set:
            sf_idx.append(idx)
    return chunks, sf_idx


def _sample_partial_context(
    chunks: List[str],
    sf_idx: List[int],
    n_select: int,
    rng: random.Random,
) -> str:
    """Random gold + distractor paragraph subset (mirrors RL partial retrieval)."""
    if n_select <= 0 or not chunks:
        return ""
    sf_set = set(sf_idx)
    distractor_idx = [i for i in range(len(chunks)) if i not in sf_set]
    n_select = min(n_select, len(chunks))
    max_correct = min(len(sf_idx), n_select)
    n_correct = rng.randint(0, max_correct) if max_correct > 0 else 0
    n_wrong = n_select - n_correct
    chosen_correct = rng.sample(sf_idx, n_correct) if n_correct > 0 else []
    n_avail_wrong = min(n_wrong, len(distractor_idx))
    chosen_wrong = (
        rng.sample(distractor_idx, n_avail_wrong) if n_avail_wrong > 0 else []
    )
    selected = sorted(chosen_correct + chosen_wrong)
    return " ".join(chunks[i] for i in selected)


def load_hotpotqa(
    path: str, split: str = "train", max_samples: int = 2000
) -> List[dict]:
    """
    Load HotPotQA JSON and return [{question, context, answer}, ...].
    Expected files: hotpot_train_v1.1.json / hotpot_dev_distractor_v1.json
    """
    file_map = {
        "train": "hotpot_train_v1.1.json",
        "dev": "hotpot_dev_distractor_v1.json",
    }
    filepath = os.path.join(path, file_map[split])
    with open(filepath) as f:
        raw = json.load(f)

    samples = []
    for item in raw[:max_samples]:
        chunks = []
        for title, sentences in item["context"]:
            chunks.append(title + " " + " ".join(sentences))
        samples.append(
            {
                "question": item["question"],
                "context": " ".join(chunks),
                "answer": item["answer"],
            }
        )
    return samples


def load_2wikimultihopqa(
    path: str,
    split: str = "train",
    max_samples: int = 2000,
    skip_bridge_comparison: bool = True,
    seed: int = 100,
    n_context_chunks: int = 2,
    partial_context: bool = True,
) -> List[dict]:
    """
    Load 2WikiMultiHopQA JSON and return [{question, context, answer}, ...].

    Expected files: train.json / dev.json / test.json (same layout as HotpotQA).

    By default, each sample uses a random subset of ``n_context_chunks`` paragraphs
    (gold + distractors), aligned with RL ``max_steps`` and ``eval_sep_reward.py``.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError(f"Unknown split for 2WikiMultihopQA: {split}")
    filepath = os.path.join(path, f"{split}.json")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"2WikiMultihopQA split file not found: {filepath}\n"
        )
    with open(filepath, encoding="utf-8") as f:
        raw = json.load(f)
    if skip_bridge_comparison:
        before = len(raw)
        raw = [item for item in raw if not _is_bridge_comparison(item.get("type"))]
        print(
            f"  2Wiki: skip bridge comparison — {before} → {len(raw)} rows"
        )
    rng = random.Random(seed)
    rng.shuffle(raw)
    samples = []
    for item in raw[:max_samples]:
        chunks, sf_idx = _2wiki_chunks_and_sf_idx(item)
        if partial_context and n_context_chunks > 0:
            context = _sample_partial_context(chunks, sf_idx, n_context_chunks, rng)
        else:
            context = " ".join(chunks)
        samples.append(
            {
                "question": item["question"],
                "context": context,
                "answer": item["answer"],
            }
        )
    return samples


def load_babilong(path: str, max_samples: int = 2000) -> List[dict]:
    """
    Load BAbILong QA txt (e.g. qa3_three-supporting-facts_train.txt).
    Stories are delimited by a line whose id resets to 1.
    Question lines contain a TAB after the question text.
    """
    with open(path) as f:
        lines = f.readlines()

    samples: List[dict] = []
    story_facts: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t")
        tokens = parts[0].split(" ", 1)
        line_id = int(tokens[0])
        text = tokens[1] if len(tokens) > 1 else ""

        if line_id == 1:
            story_facts = []

        if len(parts) >= 2:
            question = text.rstrip()
            answer = parts[1].strip()
            samples.append(
                {
                    "question": question,
                    "context": " ".join(story_facts),
                    "answer": answer,
                }
            )
            if len(samples) >= max_samples:
                break
        else:
            story_facts.append(text)

    return samples


def load_musique(
    jsonl_path: str,
    max_samples: int = 2000,
    n_refs_exactly: int = 2,
) -> List[dict]:
    """
    Load MuSiQue ``musique_ans_*.jsonl`` into
    ``[{question, context, answer}, ...]`` (same keys as HotpotQA loader).

    ``n_refs_exactly``: keep only rows with exactly this many golden reference
    paragraphs in decomposition order; use ``0`` to disable filtering.
    """
    nr = int(n_refs_exactly) if n_refs_exactly else 0
    ds = MusiqueQADataset(
        jsonl_path,
        n_refs_exactly=nr if nr > 0 else None,
    )
    filt = getattr(ds, "_musique_filter", None)
    if filt is not None:
        n, before, after = filt
        print(
            f"  MuSiQue: exactly {n} golden ref(s) — {before} → {after} rows after filter"
        )
    n_take = min(len(ds), max_samples)
    out: List[dict] = []
    for i in range(n_take):
        row = ds[i]
        out.append(
            {
                "question": row["question"],
                "context": " ".join(row["chunks"]),
                "answer": row["answer"],
            }
        )
    return out


def load_math(
    path: str,
    split: str = "train",
    max_samples: int = 2000,
    samples_num: int = 10_000,
    examples_num: int = 1000,
    n_icl_chunks: int = 2,
    seed: int = 42,
) -> List[dict]:
    """
    Load MATH for SEP training with the same ICL setup as RL (``envs=math``).

    - ``RetrievalMATH`` holds disjoint train rows for problems vs few-shot pool
      (see ``configs/envs/math.yaml`` ``samples_num`` / ``examples_num``).
    - Per problem, ``n_icl_chunks`` examples are sampled from the formatted pool
      (``problem #|||# solution`` strings), matching ``prepare_examples`` used in
      ``SEPFeedback`` / ``InfoGainFeedback`` for ICL tasks.
    """
    import random

    ds = RetrievalMATH(path, split, samples_num, examples_num)
    pool = list(ds.get_examples())
    if not pool:
        raise ValueError("MATH example pool is empty — increase math_examples_num")

    rng = random.Random(seed)
    k = max(0, int(n_icl_chunks))
    n_take = min(len(ds), max_samples)
    out: List[dict] = []
    for i in range(n_take):
        row = ds[i]
        picked = rng.sample(pool, min(k, len(pool))) if k > 0 else []
        out.append(
            {
                "question": row["problem"],
                "answer": row["answer"],
                "context": "",
                "examples": prepare_examples(picked),
            }
        )
    return out


# ── Hidden-state extraction ─────────────────────────────────────────────


class HiddenStateExtractor:
    """
    Wraps a HuggingFace causal LM.
    Provides greedy generation with hidden-state capture and
    temperature sampling for semantic-entropy estimation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        torch_dtype=torch.bfloat16,
        *,
        max_gpu_memory: Optional[str] = None,
        load_in_4bit: bool = False,
        compile_model: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kw: dict = {
            "torch_dtype": torch_dtype,
            "device_map": device,
            "trust_remote_code": True,
        }
        if max_gpu_memory is not None:
            # e.g. {"0": "6GiB", "cpu": "256GiB"} — more layers on CPU, less VRAM.
            gpu_ix = 0
            if isinstance(device, str) and device.startswith("cuda:"):
                gpu_ix = int(device.split(":")[-1])
            load_kw["max_memory"] = {gpu_ix: max_gpu_memory, "cpu": "256GiB"}
        if load_in_4bit:
            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.n_layers = self.model.config.num_hidden_layers

        if compile_model:
            try:
                self.model = torch.compile(self.model, dynamic=True)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "torch.compile(dynamic=True) failed (%s); continuing with eager mode.",
                    exc,
                )

    # ── prompt formatting ───────────────────────────────────────────────

    def _build_prompt(
        self,
        question: str,
        context: str = "",
        examples: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> str:
        """HotPotQA/MuSiQue/2Wiki: ``context`` passages. MATH/ICL: ``examples`` few-shot turns."""
        if examples:
            messages = [{"role": "system", "content": prompts.sys_math}]
            for user_input, assistant_output in examples:
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": assistant_output})
            messages.append({"role": "user", "content": question})
        elif context:
            user_msg = (
                "Answer the following question as briefly as possible.\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}"
            )
            messages = [
                {"role": "system", "content": prompts.sys_sep},
                {"role": "user", "content": user_msg},
            ]
        else:
            user_msg = (
                "Answer the following question as briefly as possible.\n\n"
                f"Question: {question}"
            )
            messages = [
                {"role": "system", "content": prompts.sys_sep},
                {"role": "user", "content": user_msg},
            ]
        try:
            # Qwen3: without this, greedy output is often 100% thinking tokens → empty after strip.
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template_kwargs={"enable_thinking": False},
                )
        except Exception:
            text = f"System: You are a helpful assistant.\nUser: {user_msg}\nAssistant:"
        return text

    # ── greedy generation + hidden-state capture ────────────────────────

    @torch.no_grad()
    def extract_hidden_states_and_greedy_answer(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 64,
        layers: Optional[List[int]] = None,
        position: str = "tbg",
        examples: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        1. Greedy-decode an answer.
        2. Run a single forward pass on [prompt ‖ answer] to get all
           hidden states.
        3. Extract & concatenate the states at the requested layers / position.

        Parameters
        ----------
        position : {"tbg", "slt"}
            "tbg" — token-before-generation: last token of the prompt.
            "slt" — second-last token of the full (prompt+answer) sequence,
                    i.e. the token right before EOS / last generated token.

        Returns (hidden_state_vector, answer_text).
        """
        prompt = self._build_prompt(question, context, examples=examples)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        gen_out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            # Override model-level sampling defaults (some checkpoints set these in
            # generation_config), otherwise transformers warns that they are ignored.
            temperature=1.0,
            top_p=1.0,
            top_k=50,
        )
        full_ids = gen_out[0]  # (total_len,)
        answer = self.tokenizer.decode(
            full_ids[prompt_len:], skip_special_tokens=True
        ).strip()

        fwd = self.model(
            input_ids=full_ids.unsqueeze(0), output_hidden_states=True
        )
        # fwd.hidden_states: tuple of (n_layers+1) tensors, each (1, seq, dim)
        # index 0 = embedding layer, index l+1 = transformer layer l

        if layers is None:
            layers = list(range(max(0, self.n_layers - 5), self.n_layers))

        total_len = full_ids.shape[0]
        if position == "tbg":
            pos = prompt_len - 1
        elif position == "slt":
            pos = max(total_len - 2, prompt_len - 1)
        else:
            raise ValueError(f"Unknown position '{position}', use 'tbg' or 'slt'")

        h_vecs = []
        for l in layers:
            h = fwd.hidden_states[l + 1][0, pos, :]
            h_vecs.append(h.cpu().float().numpy())

        return np.concatenate(h_vecs), answer

    # ── stochastic sampling ─────────────────────────────────────────────

    @torch.no_grad()
    def sample_answers(
        self,
        question: str,
        context: str = "",
        n_samples: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 64,
        examples: Optional[Sequence[Tuple[str, str]]] = None,
        sample_batch_size: int = 1,
    ) -> List[str]:
        """Sample *n_samples* answers in micro-batches (limits peak VRAM vs one big batch)."""
        prompt = self._build_prompt(question, context, examples=examples)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        batch_size = max(1, min(int(sample_batch_size), n_samples))
        answers: List[str] = []
        remaining = n_samples
        while remaining > 0:
            k = min(batch_size, remaining)
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=k,
            )
            for i in range(k):
                ans = self.tokenizer.decode(
                    out[i][prompt_len:], skip_special_tokens=True
                ).strip()
                answers.append(ans)
            remaining -= k
            del out
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        return answers


# ── NLI-based semantic clustering ───────────────────────────────────────


class SemanticClusterer:
    """
    Cluster answers with bidirectional NLI (DeBERTa-Large-MNLI; Kossen et al.).

    Implementation note: this uses *greedy sequential* assignment (each answer is
    merged into the first cluster whose representative passes both NLI
    directions).  Fully transitive semantic equivalence is often modeled as
    *connected components* of the pairwise entailment graph (as in InfoReasoner’s
    Algorithm 1 in ``rl/feedback/info_gain_feedback.py``).  The two schemes can
    disagree when entailment is intransitive; keep this in mind when comparing
    H_SE labels to other papers’ cluster definitions.
    """

    NLI_ENTAILMENT_IDX = 2  # DeBERTa MNLI label mapping: 0=contr, 1=neutral, 2=entail

    def __init__(
        self,
        model_name: str = "microsoft/deberta-large-mnli",
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def _batch_entails(
        self, pairs: List[Tuple[str, str]]
    ) -> List[bool]:
        """Run NLI on a batch of (premise, hypothesis) pairs in one forward pass."""
        if not pairs:
            return []
        enc = self.tokenizer(
            [p for p, _ in pairs],
            [h for _, h in pairs],
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        logits = self.model(**enc).logits
        # Hard entailment decision (argmax). InfoReasoner uses softmax threshold τ.
        return [
            row.argmax().item() == self.NLI_ENTAILMENT_IDX for row in logits
        ]

    def cluster(self, answers: List[str]) -> List[List[str]]:
        """
        Greedy clustering with batched NLI: for each new answer, check
        bidirectional entailment against all existing cluster representatives
        in two batched forward passes (A→B then B→A).
        """
        clusters: List[List[str]] = []
        reps: List[str] = []

        for ans in answers:
            if not reps:
                clusters.append([ans])
                reps.append(ans)
                continue

            fwd_pairs = [(ans, r) for r in reps]
            fwd_results = self._batch_entails(fwd_pairs)

            candidates = [i for i, ok in enumerate(fwd_results) if ok]

            if candidates:
                rev_pairs = [(reps[i], ans) for i in candidates]
                rev_results = self._batch_entails(rev_pairs)
                placed = False
                for idx, ok in zip(candidates, rev_results):
                    if ok:
                        clusters[idx].append(ans)
                        placed = True
                        break
                if not placed:
                    clusters.append([ans])
                    reps.append(ans)
            else:
                clusters.append([ans])
                reps.append(ans)

        return clusters


# ── Semantic entropy computation ────────────────────────────────────────


def compute_semantic_entropy(clusters: List[List[str]], n_total: int) -> float:
    """H_SE = −Σ_k  p(C_k) · log p(C_k),  where p(C_k) = |C_k| / N."""
    h = 0.0
    for cl in clusters:
        p = len(cl) / n_total
        if p > 0:
            h -= p * np.log(p + 1e-12)
    return h


def find_optimal_threshold(values: np.ndarray) -> float:
    """
    γ* that minimizes two-group MSE (regression-tree–style best split):
        argmin_γ  Σ_{j: v_j < γ} (v_j − μ_low)² + Σ_{j: v_j ≥ γ} (v_j − μ_high)²
    """
    candidates = np.sort(np.unique(values))
    if len(candidates) <= 1:
        return candidates[0] if len(candidates) else 0.0

    best_gamma, best_cost = candidates[0], np.inf
    for gamma in candidates:
        lo = values[values < gamma]
        hi = values[values >= gamma]
        if lo.size == 0 or hi.size == 0:
            continue
        cost = np.sum((lo - lo.mean()) ** 2) + np.sum((hi - hi.mean()) ** 2)
        if cost < best_cost:
            best_cost = cost
            best_gamma = gamma
    return float(best_gamma)


# ── Semantic Entropy Probe ──────────────────────────────────────────────


class SemanticEntropyProbe:
    """
    Full SEP: dataset construction, probe training, inference.

    Training
    --------
    For each query *x*:
      h(x) = hidden-state vector from a single greedy generation
      H_SE(x) = semantic entropy from N stochastic samples + NLI clustering
      ~H_SE(x) = 1[ H_SE(x) > γ* ]
    Then fit LogisticRegression( h(x) → ~H_SE(x) ).

    Inference
    ---------
    Single greedy forward pass → hidden states → probe → P(high SE).
    """

    def __init__(
        self,
        probe: Optional[LogisticRegression] = None,
        threshold: float = 0.0,
        layers: Optional[List[int]] = None,
        position: str = "tbg",
        model_name: Optional[str] = None,
    ):
        self.probe = probe
        self.threshold = threshold
        self.layers = layers
        self.position = position
        self.model_name = model_name

    # ── dataset construction ────────────────────────────────────────────

    @staticmethod
    def build_dataset(
        samples: List[dict],
        extractor: HiddenStateExtractor,
        clusterer: SemanticClusterer,
        n_samples: int = 10,
        temperature: float = 1.0,
        layers: Optional[List[int]] = None,
        position: str = "tbg",
        max_new_tokens: int = 64,
        sample_batch_size: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns
        -------
        X : (N, D) hidden-state matrix
        se_values : (N,) continuous semantic-entropy values
        greedy_answers : list of greedy-decoded answer strings
        """
        X_rows, se_rows, greedy_answers = [], [], []

        for sample in tqdm(samples, desc="Building SEP dataset"):
            q = sample["question"]
            ctx = sample.get("context", "")
            examples = sample.get("examples")

            h, answer = extractor.extract_hidden_states_and_greedy_answer(
                q,
                ctx,
                max_new_tokens=max_new_tokens,
                layers=layers,
                position=position,
                examples=examples,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            sampled = extractor.sample_answers(
                q,
                ctx,
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                examples=examples,
                sample_batch_size=sample_batch_size,
            )

            clusters = clusterer.cluster(sampled)
            se = compute_semantic_entropy(clusters, len(sampled))

            X_rows.append(h)
            se_rows.append(se)
            greedy_answers.append(answer)

        return np.stack(X_rows), np.asarray(se_rows), greedy_answers

    # ── probe training ──────────────────────────────────────────────────

    @staticmethod
    def train_probe(
        X: np.ndarray,
        se_values: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[LogisticRegression, float, dict]:
        """
        Fit scikit-learn LogisticRegression (L2, LBFGS, default C).

        Returns (probe, γ*, metrics_dict).
        """
        gamma = find_optimal_threshold(se_values)
        y = (se_values > gamma).astype(int)

        print(f"  γ* = {gamma:.4f}")
        print(f"  label counts — low: {(y == 0).sum()},  high: {(y == 1).sum()}")

        if np.unique(y).size < 2:
            gamma = float(np.median(se_values))
            y = (se_values > gamma).astype(int)
            print(f"  (adjusted γ* to median {gamma:.4f} to ensure two classes)")
            print(f"  label counts — low: {(y == 0).sum()},  high: {(y == 1).sum()}")

        stratify = y if np.unique(y).size > 1 else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size,
            random_state=random_state, stratify=stratify,
        )

        probe = LogisticRegression(max_iter=1000)
        probe.fit(X_tr, y_tr)

        y_pred = probe.predict(X_te)
        y_prob = probe.predict_proba(X_te)[:, 1]
        auroc = (
            roc_auc_score(y_te, y_prob) if np.unique(y_te).size > 1 else float("nan")
        )

        metrics = {
            "accuracy": accuracy_score(y_te, y_pred),
            "auroc": auroc,
            "gamma": gamma,
            "n_train": len(X_tr),
            "n_test": len(X_te),
        }
        print(f"  test accuracy : {metrics['accuracy']:.4f}")
        print(f"  test AUROC    : {metrics['auroc']:.4f}")
        return probe, gamma, metrics

    # ── inference ───────────────────────────────────────────────────────

    def predict_proba(self, hidden_state: np.ndarray) -> np.ndarray:
        """Return P(high SE) for one or more hidden-state vectors."""
        if hidden_state.ndim == 1:
            hidden_state = hidden_state.reshape(1, -1)
        return self.probe.predict_proba(hidden_state)[:, 1]

    def predict_from_model(
        self,
        extractor: HiddenStateExtractor,
        question: str,
        context: str = "",
        max_new_tokens: int = 64,
    ) -> Tuple[float, str]:
        """
        End-to-end inference: generate answer + predict P(high SE).
        Returns (probability, answer_text).
        """
        h, answer = extractor.extract_hidden_states_and_greedy_answer(
            question, context,
            max_new_tokens=max_new_tokens,
            layers=self.layers,
            position=self.position,
        )
        prob = float(self.predict_proba(h)[0])
        return prob, answer

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "probe": self.probe,
            "threshold": self.threshold,
            "layers": self.layers,
            "position": self.position,
            "model_name": self.model_name,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"SEP saved → {path}")

    @classmethod
    def load(cls, path: str) -> SemanticEntropyProbe:
        with open(path, "rb") as f:
            data = pickle.load(f)
        data.setdefault("model_name", None)
        return cls(**data)


# ── CLI ─────────────────────────────────────────────────────────────────


def _npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return str(value)


def _resolve_layers(
    cli_layers: Optional[Sequence[int]],
    npz_data: Optional[np.lib.npyio.NpzFile] = None,
    n_model_layers: Optional[int] = None,
) -> List[int]:
    if cli_layers is not None:
        return list(cli_layers)
    if npz_data is not None and "layers" in npz_data:
        return np.asarray(npz_data["layers"]).astype(int).tolist()
    if n_model_layers is not None:
        return list(range(max(0, n_model_layers - 5), n_model_layers))
    raise SystemExit(
        "--layers is required when the .npz file does not contain a 'layers' array"
    )


def _subset_sep_arrays(
    X: np.ndarray,
    se_values: np.ndarray,
    max_samples: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep at most *max_samples* rows (first N, or random without replacement if *seed* set)."""
    if max_samples <= 0:
        raise SystemExit("--max_samples must be positive")
    n = len(X)
    if len(se_values) != n:
        raise SystemExit(
            f"X and se_values length mismatch: {n} vs {len(se_values)}"
        )
    if max_samples >= n:
        return X, se_values
    if seed is not None:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        print(f"  subset: random {max_samples}/{n} rows (seed={seed})")
    else:
        idx = slice(max_samples)
        print(f"  subset: first {max_samples}/{n} rows")
    return X[idx], se_values[idx]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Semantic Entropy Probe (SEP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_argument_group("dataset")
    g.add_argument(
        "--dataset",
        default=None,
        choices=["hotpotqa", "babilong", "musique", "math", "2wiki"],
        help="Which dataset to use (not needed with --load_dataset)",
    )
    g.add_argument(
        "--dataset_path", default=None,
        help="Directory for hotpotqa/2wiki/math, JSONL file for musique, or txt path for babilong",
    )
    g.add_argument(
        "--load_dataset", default=None,
        help="Train from a saved .npz with arrays X and se_values (skips LLM / NLI)",
    )
    g.add_argument(
        "--split",
        default="train",
        help="hotpotqa / 2wiki / math split (ignored for musique/babilong)",
    )
    g.add_argument(
        "--musique_n_refs_exactly",
        type=int,
        default=2,
        help="musique only: keep rows with exactly this many golden refs (0 = no filter)",
    )
    g.add_argument(
        "--skip_bridge_comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2wiki only: drop bridge comparison questions (matches configs/envs/2wiki.yaml)",
    )
    g.add_argument(
        "--wiki_n_context_chunks",
        type=int,
        default=2,
        help="2wiki only: paragraphs per probe sample when partial context is on "
        "(default 2, matches configs/envs/2wiki.yaml max_steps)",
    )
    g.add_argument(
        "--wiki_partial_context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="2wiki only: train on random gold+distractor paragraph subsets "
        "instead of full oracle context",
    )
    g.add_argument(
        "--max_samples",
        type=int,
        default=2000,
        help="Max rows to use. For dataset construction: cap loaded questions. "
        "With --load_dataset: cap rows used for logreg (default: first N; "
        "use --seed for a random subset).",
    )
    g.add_argument(
        "--seed",
        type=int,
        default=None,
        help="2wiki/math: dataset shuffle seed (2wiki default 100, matches RL env). "
        "With --load_dataset: random row subset when --max_samples < .npz size.",
    )
    g.add_argument(
        "--math_samples_num",
        type=int,
        default=10_000,
        help="math only: problems drawn from train.jsonl (cf. configs/envs/math.yaml)",
    )
    g.add_argument(
        "--math_examples_num",
        type=int,
        default=1000,
        help="math only: few-shot pool size from train.jsonl",
    )
    g.add_argument(
        "--math_n_icl_chunks",
        type=int,
        default=2,
        help="math only: few-shot examples sampled per problem (cf. env max_steps)",
    )
    g.add_argument(
        "--math_seed",
        type=int,
        default=42,
        help="math only: RNG seed for sampling ICL examples per problem",
    )

    g = p.add_argument_group("models")
    g.add_argument(
        "--device",
        default="cuda:0",
        help="Device for the causal LM (e.g. cuda:0). Use a GPU not occupied by vLLM; "
        "or CUDA_VISIBLE_DEVICES=0. Avoid cuda:1 if vLLM runs there.",
    )
    g.add_argument(
        "--model_name", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace causal LM for generation + hidden states",
    )
    g.add_argument(
        "--nli_model", default="microsoft/deberta-large-mnli",
        help="NLI model for semantic clustering",
    )
    g.add_argument(
        "--nli_device", default="cpu",
        help="Device for NLI model (default: cpu to save GPU memory)",
    )
    g.add_argument(
        "--max_gpu_memory",
        default=None,
        metavar="STR",
        help="Cap VRAM for the causal LM (accelerate max_memory), e.g. 6GiB or 8GB. "
        "Pushes overflow to CPU; slower but uses less GPU. Not a substitute for "
        "freeing VRAM held by other processes.",
    )
    g.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=None,
        help="torch.cuda.set_per_process_memory_fraction (0.0–1.0) before loading "
        "the LM — caps this process's share of total GPU memory. "
        "Use only if the model still fits within that fraction.",
    )
    g.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the causal LM in 4-bit (bitsandbytes); much lower VRAM than bf16.",
    )

    g = p.add_argument_group("SEP hyper-parameters")
    g.add_argument("--n_samples", type=int, default=10)
    g.add_argument(
        "--sample_batch_size",
        type=int,
        default=1,
        help="Answers per generate() call when estimating semantic entropy. "
        "Use 1 on ~10 GiB GPUs; increase (e.g. 10) if you have headroom.",
    )
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument(
        "--position", default="tbg", choices=["tbg", "slt"],
        help="tbg = token-before-generation, slt = second-last-token",
    )
    g.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Transformer layer indices (default: last 5)",
    )
    g.add_argument("--max_new_tokens", type=int, default=64)

    g = p.add_argument_group("output")
    g.add_argument("--output", default="sep_probe.pkl")
    g.add_argument(
        "--save_dataset", default=None,
        help="Save (X, se_values) as .npz for re-use",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.load_dataset and args.save_dataset:
        raise SystemExit("Cannot use --load_dataset and --save_dataset together")

    if args.load_dataset:
        if args.dataset is not None or args.dataset_path is not None:
            print(
                "  note: --dataset / --dataset_path ignored when --load_dataset is set"
            )
        print(f"[1/2] Loading saved SEP dataset from {args.load_dataset} …")
        with np.load(args.load_dataset, allow_pickle=False) as data:
            if "X" not in data or "se_values" not in data:
                raise SystemExit(
                    f"{args.load_dataset} must contain 'X' and 'se_values' arrays"
                )
            X = np.asarray(data["X"])
            se_values = np.asarray(data["se_values"])
            position = args.position
            if "position" in data:
                position = _npz_scalar(data, "position") or position
            model_name = args.model_name
            if "model_name" in data:
                model_name = _npz_scalar(data, "model_name") or model_name
            actual_layers = _resolve_layers(args.layers, data)
        print(f"  loaded X.shape = {X.shape}")
        X, se_values = _subset_sep_arrays(
            X, se_values, args.max_samples, seed=args.seed,
        )
        print(f"  training X.shape = {X.shape}")
        print(f"  H_SE  range = [{se_values.min():.3f}, {se_values.max():.3f}]")

        print("[2/2] Training logistic regression probe …")
        probe, gamma, metrics = SemanticEntropyProbe.train_probe(X, se_values)

        sep = SemanticEntropyProbe(
            probe=probe,
            threshold=gamma,
            layers=actual_layers,
            position=position,
            model_name=model_name,
        )
        sep.save(args.output)

        print("\n✓ Done")
        print(f"  probe   : {args.output}")
        print(f"  model   : {model_name}")
        print(f"  layers  : {actual_layers}")
        print(f"  position: {position}")
        print(f"  γ*      : {gamma:.4f}")
        print(f"  metrics : {metrics}")
        return

    if args.dataset is None or args.dataset_path is None:
        raise SystemExit(
            "--dataset and --dataset_path are required unless --load_dataset is set"
        )

    # 1 — load questions
    print(f"[1/4] Loading {args.dataset} …")
    if args.dataset == "hotpotqa":
        samples = load_hotpotqa(args.dataset_path, args.split, args.max_samples)
    elif args.dataset == "2wiki":
        wiki_seed = args.seed if args.seed is not None else 100
        samples = load_2wikimultihopqa(
            args.dataset_path,
            args.split,
            args.max_samples,
            skip_bridge_comparison=args.skip_bridge_comparison,
            seed=wiki_seed,
            n_context_chunks=args.wiki_n_context_chunks,
            partial_context=args.wiki_partial_context,
        )
    elif args.dataset == "musique":
        samples = load_musique(
            args.dataset_path,
            args.max_samples,
            n_refs_exactly=args.musique_n_refs_exactly,
        )
    elif args.dataset == "math":
        samples = load_math(
            args.dataset_path,
            split=args.split,
            max_samples=args.max_samples,
            samples_num=args.math_samples_num,
            examples_num=args.math_examples_num,
            n_icl_chunks=args.math_n_icl_chunks,
            seed=args.math_seed,
        )
    else:
        samples = load_babilong(args.dataset_path, args.max_samples)
    print(f"  {len(samples)} samples loaded")

    # 2 — initialise models
    if args.gpu_memory_fraction is not None and torch.cuda.is_available():
        if not 0.0 < args.gpu_memory_fraction <= 1.0:
            raise SystemExit("--gpu_memory_fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        print(
            f"  torch.cuda.set_per_process_memory_fraction({args.gpu_memory_fraction})"
        )

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        dev_ix = int(str(args.device).split(":")[-1])
        free_b, total_b = torch.cuda.mem_get_info(dev_ix)
        print(
            f"  Target GPU {dev_ix}: {free_b / 2**30:.2f} GiB free / "
            f"{total_b / 2**30:.2f} GiB total"
        )
        if free_b < 4 * 2**30:
            print(
                "  WARNING: <4 GiB free on target GPU — another process may be using it "
                "(e.g. vLLM). Use --device cuda:0 or a free GPU, or --load_in_4bit.",
                file=sys.stderr,
            )

    print(f"[2/4] Loading LLM ({args.model_name}) on {args.device} …")
    extractor = HiddenStateExtractor(
        args.model_name,
        device=args.device,
        max_gpu_memory=args.max_gpu_memory,
        load_in_4bit=args.load_in_4bit,
    )

    print(f"[2/4] Loading NLI model ({args.nli_model}) on {args.nli_device} …")
    clusterer = SemanticClusterer(args.nli_model, device=args.nli_device)

    # 3 — build (hidden_state, SE) pairs
    print("[3/4] Generating answers & computing semantic entropy …")
    X, se_values, answers = SemanticEntropyProbe.build_dataset(
        samples, extractor, clusterer,
        n_samples=args.n_samples,
        temperature=args.temperature,
        layers=args.layers,
        position=args.position,
        max_new_tokens=args.max_new_tokens,
        sample_batch_size=args.sample_batch_size,
    )
    print(f"  X.shape = {X.shape}")
    print(f"  H_SE  range = [{se_values.min():.3f}, {se_values.max():.3f}]")

    actual_layers = _resolve_layers(args.layers, n_model_layers=extractor.n_layers)

    if args.save_dataset:
        np.savez(
            args.save_dataset,
            X=X,
            se_values=se_values,
            layers=np.asarray(actual_layers, dtype=np.int32),
            position=np.array(args.position),
            model_name=np.array(args.model_name),
        )
        print(f"  dataset saved → {args.save_dataset}")

    # 4 — train probe
    print("[4/4] Training logistic regression probe …")
    probe, gamma, metrics = SemanticEntropyProbe.train_probe(X, se_values)

    sep = SemanticEntropyProbe(
        probe=probe,
        threshold=gamma,
        layers=actual_layers,
        position=args.position,
        model_name=args.model_name,
    )
    sep.save(args.output)

    print("\n✓ Done")
    print(f"  probe   : {args.output}")
    print(f"  model   : {args.model_name}")
    print(f"  layers  : {actual_layers}")
    print(f"  position: {args.position}")
    print(f"  γ*      : {gamma:.4f}")
    print(f"  metrics : {metrics}")


if __name__ == "__main__":
    main()
