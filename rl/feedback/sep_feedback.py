from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

from rl.feedback.feedback import AFeedbackModel
from rl.feedback.llm_answer import get_final_answer, prepare_examples
from utils.process_latex import process_latex_for_cmp, simplify_latex

logger = logging.getLogger(__name__)

_MULTI_HOP_QA_TASKS = frozenset({"HotPotQA", "Musique", "2WikiMultihopQA"})


def _truncate_for_log(text: Optional[str], max_chars: int) -> str:
    """Same truncation helper as ``info_gain_feedback._truncate_for_log``."""
    if not text:
        return ""
    s = str(text).replace("\r", "").replace("\n", "\\n")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _chunks_preview(chunks: List[str], max_chars: int, max_chunks: int = 2) -> str:
    """Compact preview of retrieved passages for logs (cf. info_gain ``chunks_head``)."""
    if not chunks:
        return ""
    head = "\n---\n".join(chunks[:max_chunks])
    return _truncate_for_log(head, min(300, max_chars))


class SEPFeedback(AFeedbackModel):
    """
    Reward model that runs the SEP *inference* pathway (Kossen et al.) on
    (question + retrieved chunks), then converts P(high semantic entropy) into
    a reward (see module docstring — conversion modes are Q-RAG-specific).
    """

    FEEDBACK_MODEL_NAME = "sep"

    def __init__(
        self,
        probe_path: str,
        model_name: str,
        task: str = "HotPotQA",
        reward_mode: str = "uncertainty_penalty",
        reward_scaling: float = 1.0,
        threshold: float = 0.5,
        only_at_final: bool = True,
        never_terminate: bool = False,
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        log_reward_every_step: bool = False,
        log_max_chars: int = 512,
    ):
        super().__init__(never_terminate=never_terminate)
        self.probe_path = probe_path
        self.model_name = model_name
        self.task = task
        self.reward_mode = reward_mode
        self.reward_scaling = reward_scaling
        self.threshold = threshold
        self.only_at_final = only_at_final
        self._device = device
        self._torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.log_reward_every_step = log_reward_every_step
        self.log_max_chars = max(32, int(log_max_chars))

        self._extractor = None
        self._probe = None
        self._prev_p_high_se: Optional[float] = None
        self._reward_step_idx = 0

    # -- lazy loading (shared across copies) ----------------------------------

    def _ensure_loaded(self):
        if self._probe is not None:
            return
        from train_semantic_entropy_probe import (
            HiddenStateExtractor,
            SemanticEntropyProbe,
        )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self._extractor = HiddenStateExtractor(
            self.model_name,
            device=self._device,
            torch_dtype=dtype_map.get(self._torch_dtype, torch.bfloat16),
        )
        self._probe = SemanticEntropyProbe.load(self.probe_path)
        logger.info(
            "SEPFeedback: loaded probe from %s  (layers=%s, position=%s)",
            self.probe_path,
            self._probe.layers,
            self._probe.position,
        )

    # -- core inference -------------------------------------------------------

    def _run_sep_inference(
        self,
        question: str,
        chunks: List[str],
        examples=None,
    ) -> Tuple[float, str]:
        """Greedy forward + probe -> (P(high semantic entropy), greedy_answer_text)."""
        self._ensure_loaded()
        context = " ".join(chunks) if chunks else ""
        h, greedy_answer = self._extractor.extract_hidden_states_and_greedy_answer(
            question,
            context,
            max_new_tokens=self.max_new_tokens,
            layers=self._probe.layers,
            position=self._probe.position,
            examples=examples,
        )
        p = float(self._probe.predict_proba(h)[0])
        return p, (greedy_answer or "").strip()

    def _log_sep_io(
        self,
        *,
        question: str,
        pred_chunks: List[str],
        context_char_len: int,
        p_high_se: float,
        greedy_answer: str,
        reward: float,
        is_final: bool,
        extra: str,
    ) -> None:
        """DEBUG: inputs + probe output + reward (truncated)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        n_chunks = len(pred_chunks)
        ctx_preview = _chunks_preview(pred_chunks, self.log_max_chars)
        logger.debug(
            "sep.io step=%d final=%s mode=%s reward=%.6f p_high_se=%.6f "
            "n_chunks=%d context_char_len=%d max_new_tokens=%d probe_pos=%s "
            "greedy_answer=%r context_preview=%r question=%r %s",
            self._reward_step_idx,
            is_final,
            self.reward_mode,
            reward,
            p_high_se,
            n_chunks,
            context_char_len,
            self.max_new_tokens,
            getattr(self._probe, "position", "?"),
            _truncate_for_log(greedy_answer, self.log_max_chars),
            ctx_preview,
            _truncate_for_log(question, min(200, self.log_max_chars)),
            extra,
        )

    # -- AFeedbackModel interface ---------------------------------------------

    def reset(self, obs, info) -> None:
        super().reset(obs, info)
        self._prev_p_high_se = None
        self._reward_step_idx = 0
        if logger.isEnabledFor(logging.DEBUG):
            q = obs.get("question", "") if isinstance(obs, dict) else ""
            logger.debug(
                "sep.episode_reset q_head=%r gold_head=%r",
                _truncate_for_log(q, 160),
                _truncate_for_log(str(info.get("answer", "")), 120),
            )

    def reward(self, obs, info, is_final=False) -> float:
        self._reward_step_idx += 1
        question = obs["question"]
        pred_chunks_raw = obs.get("pred_chunks", [])

        # Route context: passages for multi-hop QA, examples for ICL tasks (cf. info_gain_feedback)
        if self.task in _MULTI_HOP_QA_TASKS:
            pred_chunks = pred_chunks_raw
            examples = None
        else:  # ICL tasks (MATH, GSM8K, …): few-shot examples for the probe
            pred_chunks = []
            examples = prepare_examples(pred_chunks_raw)

        # Task-specific gold answer normalization for logging (cf. llm_answer.py lines 161-171)
        gold_raw = str(info.get("answer", ""))
        if self.task == "MATH":
            gold_log = simplify_latex(gold_raw)
        else:
            gold_log = gold_raw

        context_joined = " ".join(pred_chunks) if pred_chunks else ""
        context_char_len = len(context_joined)

        if self.reward_mode == "uncertainty_drop":
            p, greedy_answer = self._run_sep_inference(
                question, pred_chunks, examples=examples
            )
            # Task-specific answer normalization for logging (cf. llm_answer.py lines 161-171)
            if self.task in _MULTI_HOP_QA_TASKS:
                greedy_answer = get_final_answer(greedy_answer)
            elif self.task == "MATH":
                greedy_answer = process_latex_for_cmp(greedy_answer)
                greedy_answer = simplify_latex(greedy_answer)
            if self._prev_p_high_se is None:
                r = 0.0
                extra = "drop_first_step"
            else:
                drop = self._prev_p_high_se - p
                r = self.reward_scaling * max(0.0, drop)
                extra = (
                    f"prev_p_high_se={self._prev_p_high_se:.6f} "
                    f"raw_drop={self._prev_p_high_se - p:.6f}"
                )
            self._prev_p_high_se = p

            want_info = self.log_reward_every_step or is_final
            if want_info and logger.isEnabledFor(logging.INFO):
                logger.info(
                    "sep.SSIG step=%d final=%s mode=%s reward=%.6f p_high_se=%.6f "
                    "reward_scaling=%.4f n_chunks=%d context_char_len=%d "
                    "n_chunks_preview=%r greedy_answer_head=%r question=%r gold_answer=%r %s",
                    self._reward_step_idx,
                    is_final,
                    self.reward_mode,
                    r,
                    p,
                    self.reward_scaling,
                    len(pred_chunks),
                    context_char_len,
                    _chunks_preview(pred_chunks, self.log_max_chars),
                    _truncate_for_log(greedy_answer, min(200, self.log_max_chars)),
                    _truncate_for_log(question, 160),
                    _truncate_for_log(gold_log, 120),
                    extra,
                )
            self._log_sep_io(
                question=question,
                pred_chunks=pred_chunks,
                context_char_len=context_char_len,
                p_high_se=p,
                greedy_answer=greedy_answer,
                reward=r,
                is_final=is_final,
                extra=extra,
            )
            return r

        if self.only_at_final and not is_final:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "sep.skip step=%d only_at_final=True non-final (reward=0)",
                    self._reward_step_idx,
                )
            return 0.0

        p, greedy_answer = self._run_sep_inference(
            question, pred_chunks, examples=examples
        )
        # Task-specific answer normalization for logging (cf. llm_answer.py lines 161-171)
        if self.task in _MULTI_HOP_QA_TASKS:
            greedy_answer = get_final_answer(greedy_answer)
        elif self.task == "MATH":
            greedy_answer = process_latex_for_cmp(greedy_answer)

        if self.reward_mode == "uncertainty_penalty":
            r = self.reward_scaling * (1.0 - p)
        elif self.reward_mode == "uncertainty_threshold":
            r = self.reward_scaling if p < self.threshold else 0.0
        else:
            raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

        if r >= self.reward_scaling - 1e-6:
            self.completed = True

        want_info = self.log_reward_every_step or is_final
        if want_info and logger.isEnabledFor(logging.INFO):
            thr_note = (
                f"threshold={self.threshold:.4f} pass={p < self.threshold}"
                if self.reward_mode == "uncertainty_threshold"
                else ""
            )
            logger.info(
                "sep.SSIG step=%d final=%s mode=%s reward=%.6f p_high_se=%.6f "
                "reward_scaling=%.4f n_chunks=%d context_char_len=%d "
                "n_chunks_preview=%r greedy_answer_head=%r question=%r gold_answer=%r %s",
                self._reward_step_idx,
                is_final,
                self.reward_mode,
                r,
                p,
                self.reward_scaling,
                len(pred_chunks),
                context_char_len,
                _chunks_preview(pred_chunks, self.log_max_chars),
                _truncate_for_log(greedy_answer, min(200, self.log_max_chars)),
                _truncate_for_log(question, 160),
                _truncate_for_log(gold_log, 120),
                thr_note,
            )
        extra_debug = (
            thr_note
            if self.reward_mode == "uncertainty_threshold"
            else "r=reward_scaling*(1-p_high_se)"
        )
        self._log_sep_io(
            question=question,
            pred_chunks=pred_chunks,
            context_char_len=context_char_len,
            p_high_se=p,
            greedy_answer=greedy_answer,
            reward=r,
            is_final=is_final,
            extra=extra_debug,
        )

        return r

    def copy(self):
        self._ensure_loaded()
        clone = SEPFeedback(
            probe_path=self.probe_path,
            model_name=self.model_name,
            task=self.task,
            reward_mode=self.reward_mode,
            reward_scaling=self.reward_scaling,
            threshold=self.threshold,
            only_at_final=self.only_at_final,
            never_terminate=self.never_terminate,
            device=self._device,
            torch_dtype=self._torch_dtype,
            max_new_tokens=self.max_new_tokens,
            log_reward_every_step=self.log_reward_every_step,
            log_max_chars=self.log_max_chars,
        )
        clone._extractor = self._extractor
        clone._probe = self._probe
        return clone
