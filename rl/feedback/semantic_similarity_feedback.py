"""
Semantic-Similarity Feedback for Q-RAG training.

Reward = cosine_sim(embed(generated_answer), embed(gold_answer))  ∈ [0.0, 1.0]

Uses:
  - A vLLM-backed LLMGenerator to produce a short answer from the retrieved
    context.
  - A local sentence-transformers embedding model (default:
    Qwen/Qwen3-Embedding-0.6B) to compute cosine similarity between the
    predicted and gold answer embeddings.

Custom hyperparameters live in configs/feedback/base_rewards.yaml:
  feedback.sem_embed_model   — HuggingFace embedding model id
  feedback.sem_embed_device  — device for the embedding model ("cpu", "cuda:1", …)
  feedback.sem_embed_fp16    — load embedding weights in float16 on CUDA

General hyperparameters (api_base_url, model, reward_scaling, …) are taken
from the same file, mirroring the feedback.* namespace of defaults.yaml.
"""

from __future__ import annotations

import re
import logging
import time

import numpy as np

from rl.feedback.feedback import AFeedbackModel
from rl.feedback.llm_feedback import LLMGenerator

logger = logging.getLogger(__name__)


# ── Text helpers ──────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3-style outputs."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    return result.strip()


# ── Feedback class ────────────────────────────────────────────────────────────

class SemanticSimilarityFeedback(AFeedbackModel):
    """
    Reward = cosine_sim(embed(generated_answer), embed(gold_answer)).

    The embedding model is loaded once at construction and shared across all
    ``copy()`` instances (no duplicate model weights for parallel envs).

    Cosine similarity is clamped to [0, 1]: for short factoid answers the
    vectors are well-separated so negative values are rare, but clamping
    avoids spurious negative rewards.
    """

    FEEDBACK_MODEL_NAME = "semantic_similarity"

    def __init__(
        self,
        llm_generator: LLMGenerator,
        embed_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        embed_device: str = "cpu",
        embed_use_fp16: bool = False,
        only_at_final: bool = True,
        reward_scaling: float = 1.0,
        never_terminate: bool = True,
    ):
        super().__init__(never_terminate=never_terminate)
        self.llm = llm_generator
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.embed_use_fp16 = embed_use_fp16
        self.only_at_final = only_at_final
        self.reward_scaling = reward_scaling

        self._embed_model = self._load_embed_model()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_embed_model(self):
        from sentence_transformers import SentenceTransformer
        import torch

        st_kw: dict = {"device": self.embed_device}
        if self.embed_use_fp16 and str(self.embed_device).startswith("cuda"):
            st_kw["model_kwargs"] = {"torch_dtype": torch.float16}

        logger.info(
            "SemanticSimilarityFeedback: loading %s on %s",
            self.embed_model_name,
            self.embed_device,
        )
        return SentenceTransformer(self.embed_model_name, **st_kw)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _cosine_sim(self, text_a: str, text_b: str) -> float:
        embs = self._embed_model.encode(
            [text_a, text_b],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return float(max(0.0, np.dot(embs[0], embs[1])))

    # ── AFeedbackModel interface ──────────────────────────────────────────────

    def reset(self, obs, info) -> None:
        super().reset(obs, info)

    def reward(self, obs, info, is_final: bool = False) -> float:
        if self.only_at_final and not is_final:
            return 0.0

        question = obs["question"]
        pred_chunks = obs.get("pred_chunks", [])
        gold = info.get("answer", "")

        raw_pred = self.llm.generate_answer(question, pred_chunks)
        pred = _strip_thinking(raw_pred)
        score = self._cosine_sim(pred, gold)
        logger.debug(
            "SemanticSimilarityFeedback: pred=%r  gold=%r  sim=%.4f",
            pred[:60],
            gold[:60],
            score,
        )
        return score * self.reward_scaling

    def copy(self) -> "SemanticSimilarityFeedback":
        # Bypass __init__ to reuse the already-loaded embedding model.
        obj = SemanticSimilarityFeedback.__new__(SemanticSimilarityFeedback)
        AFeedbackModel.__init__(obj, never_terminate=self.never_terminate)
        obj.llm = self.llm
        obj.embed_model_name = self.embed_model_name
        obj.embed_device = self.embed_device
        obj.embed_use_fp16 = self.embed_use_fp16
        obj.only_at_final = self.only_at_final
        obj.reward_scaling = self.reward_scaling
        obj._embed_model = self._embed_model  # shared reference, no extra VRAM
        return obj
