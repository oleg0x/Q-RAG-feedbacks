"""
Synthetic Semantic Information Gain reward (InfoReasoner; Hu et al., 2026).

This module implements the stochastic sampling + NLI semantic clustering +
likelihood-weighted beliefs described in the paper, and exposes two IG variants
(Eq. 15 and Eq. 17) plus the terminal exact-match term (Eq. 18).

Per-step pipeline (maps to Algorithm 1 / Def. 4.1)
--------------------------------------------------
1. Query the policy LLM once per belief estimate (prior: question only; posterior:
   question + retrieved chunks).  With ``BasicVllmClient`` the completion is greedy
   and broadcast to ``M`` slots; prior statistics are cached when ``cache_prior``
   is true.
2. Sequence log-probabilities from vLLM (when ``use_sample_logprobs``) yield
   normalized importance weights over samples (Eq. 13); otherwise weights are
   uniform ``1/M``.
3. Semantic equivalence uses *bidirectional* NLI: pair (a, b) is linked only if
   both directions exceed threshold τ on the *entailment* class probability
   (Def. 4.1).  Clustering is *connected components* of that graph (Algorithm 1,
   union-find in ``NLIChecker.cluster_connected_components``).
4. ``golden_class``: IG = log p(c*|Z_post) − log p(c*|Z_prior) with p(c*|Z) the
   sum of weights of samples NLI-equivalent to the gold string (Eq. 17).
   ``entropy_reduction``: IG = H_sem(prior) − H_sem(posterior) with H_sem from
   weighted cluster masses (Eq. 14–15).
5. At the final env step, add ``em_weight * 1[majority vote agrees with gold]``
   where “majority” is over **normalized** exact match (same string prep as
   ``eval_llm_openqa.normalize_answer`` / HotpotQA-style EM): strictly more than
   half of posterior samples match the gold after lowercasing, punctuation and
   article stripping.  NLI is still used for IG / semantic belief (items 3–4);
   EM uses normalized text equality for alignment with offline EM metrics.

Prompting vs. paper Figure 7
----------------------------
``InfoGainPromptFormatter`` follows the paper’s short QA style (document block +
question, or knowledge-only wording for empty retrieval).

Training entrypoint
--------------------
Hydra: ``feedback.type=info_gain`` with a running vLLM OpenAI-compatible server;
see ``configs/feedback/defaults.yaml``.  Uses ``BasicVllmClient`` with optional
token logprobs for likelihood-weighted beliefs; greedy ``temperature=0`` path
broadcasts one completion ``M`` times (equal weights when logprobs disabled).

Requires a running vLLM server, e.g.:
  CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-4B \\
      --host 127.0.0.1 --port 10001 --api-key keykey \\
      --served-model-name feedback --gpu-memory-utilization 0.5 \\
      --max-model-len 2048 --tensor-parallel-size 1
"""

from __future__ import annotations

import logging
import re
import string
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

from prompts_and_metrics import prompts
from rl.feedback.feedback import AFeedbackModel
from rl.feedback.llm_answer import get_final_answer, prepare_examples
from utils.process_latex import process_latex_for_cmp, simplify_latex, simplify_latex

logger = logging.getLogger(__name__)

# Open-domain multi-hop QA: retrieved passages in ``pred_chunks`` (not ICL examples).
_MULTI_HOP_QA_TASKS = frozenset({"HotPotQA", "Musique", "2WikiMultihopQA"})


# ── Diagnostics (logging) ───────────────────────────────────────────────
#
# Train vs post-hoc eval gap — useful checks when train loss/eval_r_sum look
# good but offline metrics are poor:
#
#  1. Policy mismatch: rollout may use stochastic action selection while your
#     standalone eval script uses argmax — compare eval protocols.
#  2. Reward horizon: SSIG mixes marginal IG each step + terminal EM proxy;
#     sum of discounted rewards may not correlate with EM/F1 on answers.
#  3. Feedback-model mismatch: terminal EM uses normalized EM (eval_llm_openqa);
#     IG still uses NLI — compare logs to offline EM/F1.
#  4. Distribution shift: vLLM temperature/top_p vs greedy at eval time changes
#     belief estimates and thus IG noise.
#  5. Chunk quality: log n_pred_chunks and truncated chunk previews when
#     IG is large but posterior_frac stays low (retrieval hurting NLI belief).
#
# Enable DEBUG on ``rl.feedback.info_gain_feedback`` for full answer strings;
# ``log_reward_every_step=true`` + INFO floods logs in parallel training.


def _truncate_for_log(text: Optional[str], max_chars: int) -> str:
    if not text:
        return ""
    s = str(text).replace("\r", "").replace("\n", "\\n")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _answers_preview(
    answers: List[str],
    max_chars_total: int,
    max_unique: int = 8,
) -> str:
    """Compact, human-readable preview of sampled answers for logs."""
    if not answers:
        return "(none)"
    seen = []
    for a in answers:
        t = (a or "").strip()
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= max_unique:
            break
    blob = " | ".join(_truncate_for_log(u, min(120, max_chars_total // max(1, len(seen)))) for u in seen)
    nu = len({(a or "").strip() for a in answers})
    suffix = "" if nu <= len(seen) else f" …(+{nu - len(seen)} more unique)"
    return f"[{blob}{suffix}] u={nu}/{len(answers)}"


def _normalize_logprobs(raw: List[float]) -> np.ndarray:
    """Normalize raw log-scores to a probability vector (Eq. 13)."""
    lp = np.asarray(raw, dtype=np.float64)
    lp -= np.max(lp)
    w = np.exp(lp)
    total = float(w.sum())
    if total < 1e-300:
        return np.ones(len(raw), dtype=np.float64) / max(len(raw), 1)
    return w / total


def normalize_answer(s: str) -> str:
    """Same normalization as ``eval_llm_openqa.normalize_answer`` (HotpotQA-style EM)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def normalized_em_fraction(answers: List[str], golden: str) -> float:
    """
    Fraction of samples whose normalized text equals normalized gold (Eq. 18 EM proxy).

    Matches ``eval_llm_openqa.compute_exact_match`` logic aggregated over samples.
    """
    if not answers:
        return 0.0
    gt = normalize_answer(str(golden))
    hits = sum(
        1 for a in answers if normalize_answer(str(a or "")) == gt
    )
    return hits / len(answers)


# ── Prompt formatter (paper Figure 7) ───────────────────────────────────


class InfoGainPromptFormatter:
    """Prompt formatter matching Figure 7 of the InfoReasoner paper."""

    def __init__(self, **_kwargs):
        # Hydra may pass keys from other configs (e.g. ``babi_task`` when swapping
        # ``BabilongPromptFormatter`` for this class under ``feedback.prompt_formatter``).
        pass

    def __call__(self, question: str, pred_chunks: List[str]):
        if pred_chunks:
            documents = "\n".join(pred_chunks)
            user_msg = (
                "Answer the question based on the given document. "
                "Only give me the answer and do not output any other words.\n"
                f"The following are given documents.\n{documents}\n"
                f"Question: {question}"
            )
        else:
            user_msg = (
                "Answer the question based on your own knowledge. "
                "Only give me the answer and do not output any other words.\n"
                f"Question: {question}"
            )
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer concisely.",
            },
            {"role": "user", "content": user_msg},
        ]


# ── Union-Find for connected components (Algorithm 1, lines 8-16) ──────


class _UnionFind:
    """Minimal union-find (disjoint set) for small M."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> List[List[int]]:
        groups: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            groups.setdefault(self.find(i), []).append(i)
        return list(groups.values())


# ── NLI-based semantic checker (Def 4.1) ────────────────────────────────


class NLIChecker:
    """
    Bidirectional NLI checker for semantic equivalence.

    Uses DeBERTa-Large fine-tuned on MNLI.  Two texts are considered
    *semantically equivalent* iff both entailment probabilities exceed
    the threshold τ (paper Def 4.1, Eq. 12).
    """

    ENTAILMENT_IDX = 2  # 0=contradiction, 1=neutral, 2=entailment

    def __init__(
        self,
        model_name: str = "microsoft/deberta-large-mnli",
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_safetensors=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            use_safetensors=False,
        ).to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def _batch_entails(self, pairs: List[tuple]) -> List[bool]:
        """Return per-pair entailment decisions using softmax + threshold τ."""
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
        probs = torch.softmax(logits, dim=-1)
        return [
            row[self.ENTAILMENT_IDX].item() > self.threshold for row in probs
        ]

    def match_mask(self, answers: List[str], golden: str) -> List[bool]:
        """
        Boolean mask: answers[i] is semantically equivalent to *golden*
        via bidirectional NLI (A⊨B ∧ B⊨A) with threshold τ.
        """
        if not answers:
            return []
        fwd_pairs = [(a, golden) for a in answers]
        bwd_pairs = [(golden, a) for a in answers]
        fwd = self._batch_entails(fwd_pairs)
        bwd = self._batch_entails(bwd_pairs)
        return [f and b for f, b in zip(fwd, bwd)]

    def cluster_connected_components(
        self, answers: List[str]
    ) -> List[List[int]]:
        """
        Semantic clustering via connected components (Algorithm 1, lines 8-16).

        Builds an undirected graph where nodes are answer indices and edges
        connect pairs with bidirectional entailment, then extracts connected
        components.  Returns list of clusters (each a list of indices).
        """
        n = len(answers)
        if n == 0:
            return []
        if n == 1:
            return [[0]]

        fwd_pairs = []
        bwd_pairs = []
        pair_indices = []
        for i in range(n):
            for j in range(i + 1, n):
                fwd_pairs.append((answers[i], answers[j]))
                bwd_pairs.append((answers[j], answers[i]))
                pair_indices.append((i, j))

        fwd_results = self._batch_entails(fwd_pairs)
        bwd_results = self._batch_entails(bwd_pairs)

        uf = _UnionFind(n)
        for (i, j), fwd_ok, bwd_ok in zip(
            pair_indices, fwd_results, bwd_results
        ):
            if fwd_ok and bwd_ok:
                uf.union(i, j)

        return uf.components()


# ── Likelihood-weighted semantic entropy (Eq. 13-14) ────────────────────


def _semantic_entropy_weighted(
    clusters: List[List[int]], weights: np.ndarray
) -> float:
    """
    H_SE = −Σ_k  p(C_k) · log p(C_k)   (Eq. 14)

    *weights* are normalized sequence probabilities (Eq. 13):
        p(c | Z) = Σ_{s ∈ c} p_θ(s | Z)
    """
    h = 0.0
    for cluster_indices in clusters:
        p = float(weights[cluster_indices].sum())
        if p > 1e-15:
            h -= p * np.log(p)
    return h


# ── InfoGainFeedback ────────────────────────────────────────────────────


class InfoGainFeedback(AFeedbackModel):
    """
    InfoReasoner-style feedback: IG from semantic belief dynamics (Eq. 15 / 17)
    plus optional terminal EM bonus (Eq. 18).

    LLM calls go through ``vllm_client``
    (:class:`vLLM_clients.vllm_client.BasicVllmClient`) using only
    ``chat_completion(query, examples, passages)`` — no extensions to the client.

    ``use_sample_logprobs`` toggles Eq. 13 likelihood weights via vLLM token logprobs.

    ``never_terminate`` is normally overridden from Hydra (defaults to true in
    ``configs/feedback/defaults.yaml``), which disables early termination flags
    below even when ``completed`` is set.
    """

    FEEDBACK_MODEL_NAME = "info_gain"

    def __init__(
        self,
        vllm_client,
        task: str = "HotPotQA",
        n_samples: int = 12,
        sampling_temperature: float = 1.0,
        sampling_top_p: float = 0.9,
        prompt_formatter: Optional[Callable[[str, List[str]], List[dict]]] = None,
        reward_scaling: float = 1.0,
        reward_mode: str = "golden_class",
        nli_model_name: str = "microsoft/deberta-large-mnli",
        nli_device: str = "cuda",
        nli_threshold: float = 0.5,
        cache_prior: bool = True,
        em_weight: float = 1.0,
        never_terminate: bool = True,
        log_reward_every_step: bool = False,
        log_max_answer_chars: int = 400,
        use_sample_logprobs: bool = True,
    ):
        super().__init__(never_terminate=never_terminate)
        self.task = task
        self.vllm = vllm_client
        self.vllm._system_message = {"role": "system", "content": prompts.sys_ssig}
        self.n_samples = n_samples
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.use_sample_logprobs = use_sample_logprobs
        self._prompt_formatter = prompt_formatter or InfoGainPromptFormatter()
        self.reward_scaling = reward_scaling
        self.reward_mode = reward_mode
        self.nli_model_name = nli_model_name
        self.nli_device = nli_device
        self.nli_threshold = nli_threshold
        self.cache_prior = cache_prior
        self.em_weight = em_weight
        self.log_reward_every_step = log_reward_every_step
        self.log_max_answer_chars = max(32, int(log_max_answer_chars))

        self._nli: Optional[NLIChecker] = None
        self._cached_prior_p: Optional[float] = None
        self._cached_prior_entropy: Optional[float] = None
        # Previous-step posteriors for marginal IG computation (Bug 1 fix).
        self._cached_prev_p: Optional[float] = None
        self._cached_prev_entropy: Optional[float] = None
        self._reward_step_idx = 0

    # ── lazy NLI loading (shared across copies) ─────────────────────────

    def _ensure_nli_loaded(self):
        if self._nli is not None:
            return
        self._nli = NLIChecker(
            self.nli_model_name,
            device=self.nli_device,
            threshold=self.nli_threshold,
        )
        logger.info(
            "InfoGainFeedback: NLI model loaded (%s on %s, τ=%.2f)",
            self.nli_model_name,
            self.nli_device,
            self.nli_threshold,
        )

    # ── sampling (Algorithm 1, lines 4-6) ───────────────────────────────

    @staticmethod
    def _passages_for_client(chunks: List[str]) -> List[str]:
        return chunks if chunks else [""]

    def _sample_stochastic(
        self, question: str, chunks: List[str], n: int, examples=None
    ) -> Tuple[List[str], List[Optional[float]]]:
        """
        Request ``n`` independent completions via ``BasicVllmClient.chat_completion_n``.

        Returns answers and optional cumulative sequence log-probs per completion
        when ``use_sample_logprobs`` is True (Eq. 13).

        Falls back to M copies of a single greedy answer on any error so that
        training is never blocked by a transient API failure.
        """
        lp_flag = self.use_sample_logprobs
        try:
            if examples is not None:
                results = self.vllm.chat_completion_n(
                    question,
                    n,
                    examples=examples,
                    temperature=self.sampling_temperature,
                    top_p=self.sampling_top_p,
                    logprobs=lp_flag,
                )
            else:
                results = self.vllm.chat_completion_n(
                    question,
                    n,
                    passages=self._passages_for_client(chunks),
                    temperature=self.sampling_temperature,
                    top_p=self.sampling_top_p,
                    logprobs=lp_flag,
                )
            answers = [(text or "").strip() for text, _ in results]
            raw_lps = [cum for _, cum in results]
            return answers, raw_lps
        except Exception as e:
            logger.warning(
                "InfoGainFeedback: stochastic sampling (n=%d, T=%.2f) failed: %s;"
                " falling back to single greedy call",
                n,
                self.sampling_temperature,
                e,
            )
            if examples is not None:
                text, cumulative_lp = self.vllm.chat_completion(
                    question, examples=examples, passages=None
                )
            else:
                text, cumulative_lp = self.vllm.chat_completion(
                    question,
                    examples=None,
                    passages=self._passages_for_client(chunks),
                )
            a = (text or "").strip()
            return [a] * n, [cumulative_lp] * n

    def _weights_from_raw_logprobs(
        self, raw_lps: List[Optional[float]]
    ) -> np.ndarray:
        """Eq. 13: softmax-normalized weights; uniform if disabled or all missing."""
        n = len(raw_lps)
        if n == 0:
            return np.array([], dtype=np.float64)
        if not self.use_sample_logprobs:
            return np.ones(n, dtype=np.float64) / n
        dense: List[float] = []
        for v in raw_lps:
            dense.append(float(v) if v is not None and np.isfinite(v) else -1.0e30)
        if all(v <= -1.0e29 for v in dense):
            logger.warning(
                "InfoGainFeedback: no valid logprobs from vLLM; using uniform weights"
            )
            return np.ones(n, dtype=np.float64) / n
        return _normalize_logprobs(dense)

    def _sample_with_logprobs(
        self, question: str, chunks: List[str], examples=None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Return ``(answers, weights)`` for NLI-based belief estimation (Eq. 13).

        When ``use_sample_logprobs`` and vLLM returns token logprobs, weights follow
        the normalized cumulative sequence log-probability per sample. Otherwise
        weights are uniform ``1/M``.

        When ``sampling_temperature > 0`` we request ``n_samples`` independent
        completions (needed for ``entropy_reduction`` diversity).

        When ``sampling_temperature == 0`` one greedy completion is broadcast
        to all M slots (equal weights under logprobs, or uniform).

        When ``examples`` is provided (ICL tasks like MATH), the raw question is
        passed as ``query`` and examples are forwarded to the vLLM client; the
        prompt formatter is bypassed.
        """
        n = max(int(self.n_samples), 1)
        lp_flag = self.use_sample_logprobs

        if self.sampling_temperature > 0.0:
            answers, raw_lps = self._sample_stochastic(
                question, chunks, n, examples=examples
            )
        else:
            if examples is not None:
                text, cumulative_lp = self.vllm.chat_completion(
                    question, examples=examples, passages=None, get_cumulative=True
                )
            else:
                text, cumulative_lp = self.vllm.chat_completion(
                    question,
                    examples=None,
                    passages=self._passages_for_client(chunks),
                    get_cumulative=True
                )
            answers = [(text or "").strip()] * n
            raw_lps = [cumulative_lp] * n

        weights = self._weights_from_raw_logprobs(raw_lps)

        if logger.isEnabledFor(logging.DEBUG):
            mode = (
                "stochastic_n"
                if self.sampling_temperature > 0.0
                else "greedy_broadcast"
            )
            slots_trunc = [
                _truncate_for_log(a, self.log_max_answer_chars) for a in answers
            ]
            logger.debug(
                "info_gain.vllm mode=%s M=%d T=%.3f top_p=%.3f use_logprobs=%s "
                "question_head=%r n_chunks=%d preview=%s",
                mode,
                n,
                self.sampling_temperature,
                self.sampling_top_p,
                self.use_sample_logprobs,
                _truncate_for_log(question, 200),
                len(chunks),
                _answers_preview(answers, self.log_max_answer_chars),
            )
            logger.debug(
                "info_gain.vllm weights(first8)=%r raw_lps(first8)=%r",
                weights[: min(8, len(weights))].tolist(),
                raw_lps[: min(8, len(raw_lps))],
            )
            # Exact slot strings as processed for NLI (truncated per log_max_answer_chars).
            logger.debug("info_gain.vllm answers=%r", slots_trunc)

        return answers, weights

    # ── belief estimation: golden class (Eq. 13 + Eq. 17) ───────────────

    def _estimate_p_correct(
        self, question: str, chunks: List[str], golden: str, examples=None
    ) -> Tuple[float, float, List[str]]:
        """
        Estimate p(c* | Z) using likelihood-weighted belief (Eq. 13).

        Returns (p_correct_weighted, em_majority_frac, vllm_sampled_answers).
        """
        self._ensure_nli_loaded()
        answers, weights = self._sample_with_logprobs(question, chunks, examples=examples)
        mask = self._nli.match_mask(answers, golden)
        mask_arr = np.array(mask, dtype=np.float64)

        p_weighted = float((weights * mask_arr).sum())
        em_majority_frac = normalized_em_fraction(answers, golden)
        p_floor = 1.0 / (self.n_samples + 1)
        return max(p_weighted, p_floor), em_majority_frac, answers

    # ── belief estimation: entropy (Eq. 13-15) ──────────────────────────

    def _estimate_entropy(
        self, question: str, chunks: List[str], golden: str, examples=None
    ) -> Tuple[float, float, List[str]]:
        """
        Estimate H_SE(Z) via connected-component clustering (Algorithm 1)
        with likelihood-weighted cluster probabilities (Eq. 13-14).

        Returns (entropy, em_majority_frac, vllm_sampled_answers).
        """
        self._ensure_nli_loaded()
        answers, weights = self._sample_with_logprobs(question, chunks, examples=examples)
        clusters = self._nli.cluster_connected_components(answers)
        entropy = _semantic_entropy_weighted(clusters, weights)

        em_majority_frac = normalized_em_fraction(answers, golden)
        return entropy, em_majority_frac, answers

    # ── AFeedbackModel interface ────────────────────────────────────────

    def reset(self, obs, info) -> None:
        super().reset(obs, info)
        self._cached_prior_p = None
        self._cached_prior_entropy = None
        self._cached_prev_p = None
        self._cached_prev_entropy = None
        self._reward_step_idx = 0
        if logger.isEnabledFor(logging.DEBUG):
            q = obs.get("question", "") if isinstance(obs, dict) else ""
            logger.debug(
                "info_gain.episode_reset q_head=%r gold_head=%r",
                _truncate_for_log(q, 160),
                _truncate_for_log(str(info.get("answer", "")), 120),
            )

    def reward(self, obs, info, is_final=False) -> float:
        self._reward_step_idx += 1
        question = obs["question"]
        pred_chunks_raw = obs.get("pred_chunks", [])
        golden = info.get("answer", "")

        # Route context: passages for multi-hop QA, examples for ICL tasks (cf. llm_answer.py lines 135-140)
        if self.task in _MULTI_HOP_QA_TASKS:
            chunks = pred_chunks_raw
            examples = None
        else:  # ICL tasks (MATH, GSM8K, …): pred_chunks are few-shot examples
            chunks = []
            examples = prepare_examples(pred_chunks_raw)

        # Task-specific gold answer normalization (cf. llm_answer.py lines 161-171)
        if self.task in _MULTI_HOP_QA_TASKS:
            golden_norm = get_final_answer(str(golden))
        elif self.task == "MATH":
            golden_norm = simplify_latex(str(golden))
        else:
            golden_norm = str(golden)

        # Fraction of vLLM samples matching gold under HotpotQA-style normalized EM (terminal bonus).
        posterior_frac = 0.0
        posterior_p = float("nan")
        prev_p = float("nan")
        posterior_entropy = float("nan")
        prev_entropy = float("nan")
        vllm_posterior_answers: List[str] = []

        if self.reward_mode == "golden_class":
            # Ensure prior is estimated once per episode (empty context).
            if self._cached_prior_p is None or not self.cache_prior:
                (
                    self._cached_prior_p,
                    _,
                    _,
                ) = self._estimate_p_correct(question, [], golden_norm, examples=examples)
                # On the very first step the "previous" posterior is the prior
                # itself so the first marginal IG equals the absolute IG from
                # the prior — exactly the same value as the old cumulative
                # formula, but subsequent steps now credit only their own
                # marginal contribution (Bug 1 fix).
                self._cached_prev_p = self._cached_prior_p

            (
                posterior_p,
                posterior_frac,
                vllm_posterior_answers,
            ) = self._estimate_p_correct(question, chunks, golden_norm, examples=examples)

            # Marginal IG: log p(c*|Z_t) − log p(c*|Z_{t-1})  (Bug 1 fix).
            # Both values are already floored by _estimate_p_correct, so no
            # extra epsilon is needed here (Bug 2 fix).
            prev_p = float(self._cached_prev_p or 1.0 / (self.n_samples + 1))
            ig = np.log(posterior_p) - np.log(prev_p)
            self._cached_prev_p = posterior_p

            if posterior_p > 0.8:
                self.completed = True

        elif self.reward_mode == "entropy_reduction":
            # Ensure prior entropy is estimated once per episode.
            if self._cached_prior_entropy is None or not self.cache_prior:
                (
                    self._cached_prior_entropy,
                    _,
                    _,
                ) = self._estimate_entropy(question, [], golden_norm, examples=examples)
                self._cached_prev_entropy = self._cached_prior_entropy

            (
                posterior_entropy,
                posterior_frac,
                vllm_posterior_answers,
            ) = self._estimate_entropy(question, chunks, golden_norm, examples=examples)

            # Marginal entropy reduction: H_{t-1} − H_t
            prev_entropy = float(self._cached_prev_entropy or 0.0)
            ig = prev_entropy - posterior_entropy
            self._cached_prev_entropy = posterior_entropy

            if posterior_entropy < 0.1:
                self.completed = True

        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        # For MATH: recompute EM fraction using LaTeX-normalized comparison (cf. llm_answer.py line 167-168)
        if self.task == "MATH" and vllm_posterior_answers:
            posterior_frac = sum(
                1 for a in vllm_posterior_answers
                if process_latex_for_cmp(str(a or "")) == golden_norm
            ) / len(vllm_posterior_answers)

        # SSIG components: marginal information gain (scaled) vs terminal EM proxy (Eq. 18).
        ig_raw = float(ig)
        ig_scaled = float(self.reward_scaling * ig_raw)
        em_component = 0.0
        if is_final and self.em_weight > 0:
            em_correct = float(posterior_frac > 0.5)
            em_component = float(self.em_weight * em_correct)
        total_reward = ig_scaled + em_component

        want_info = self.log_reward_every_step or is_final
        if want_info and logger.isEnabledFor(logging.INFO):
            n_chunks = len(chunks)
            chunks_head = ""
            if chunks:
                chunks_head = _truncate_for_log(
                    "\n---\n".join(chunks[:2]), min(300, self.log_max_answer_chars)
                )
            extra_gc = ""
            if self.reward_mode == "golden_class":
                extra_gc = (
                    f"prior_p={self._cached_prior_p:.6f} "
                    f"posterior_p={posterior_p:.6f} prev_p={prev_p:.6f} "
                )
            else:
                extra_gc = (
                    f"prior_H={self._cached_prior_entropy:.6f} "
                    f"posterior_H={posterior_entropy:.6f} prev_H={prev_entropy:.6f} "
                )
            logger.info(
                "info_gain.SSIG step=%d final=%s mode=%s "
                "IG_raw=%.6f IG_scaled=%.6f EM_ssig=%.6f total=%.6f "
                "em_norm_sample_frac=%.4f %s"
                "n_chunks=%d chunks_retrieved=%r vllm_posterior_samples=%s "
                "question=%r gold_answer=%r",
                self._reward_step_idx,
                is_final,
                self.reward_mode,
                ig_raw,
                ig_scaled,
                em_component,
                total_reward,
                posterior_frac,
                extra_gc,
                n_chunks,
                chunks_head,
                _answers_preview(vllm_posterior_answers, self.log_max_answer_chars),
                _truncate_for_log(question, 160),
                _truncate_for_log(golden_norm, 120),
            )

        return total_reward

    def copy(self):
        self._ensure_nli_loaded()
        clone = InfoGainFeedback(
            vllm_client=self.vllm,
            task=self.task,
            n_samples=self.n_samples,
            sampling_temperature=self.sampling_temperature,
            sampling_top_p=self.sampling_top_p,
            prompt_formatter=self._prompt_formatter,
            reward_scaling=self.reward_scaling,
            reward_mode=self.reward_mode,
            nli_model_name=self.nli_model_name,
            nli_device=self.nli_device,
            nli_threshold=self.nli_threshold,
            cache_prior=self.cache_prior,
            em_weight=self.em_weight,
            never_terminate=self.never_terminate,
            log_reward_every_step=self.log_reward_every_step,
            log_max_answer_chars=self.log_max_answer_chars,
            use_sample_logprobs=self.use_sample_logprobs,
        )
        # Share the already-loaded NLI model to avoid reloading it.
        clone._nli = self._nli
        # Episode-level cached state starts fresh in each clone, which is
        # correct — reset() will populate it before the first reward() call.
        return clone
