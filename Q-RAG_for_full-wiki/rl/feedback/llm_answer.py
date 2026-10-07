"""Q-ICL/reader feedback backed by an OpenAI-compatible vLLM server."""

import logging
import time
from typing import List, Optional

from rouge import Rouge

from envs.dataloaders.base import separator
from prompts_and_metrics import prompts
from prompts_and_metrics.general_qa import normalize_answer
from rl.feedback.feedback import AFeedbackModel
from utils.process_latex import extract_boxed_expression
from vLLM_clients.vllm_client import BasicVllmClient
#from vLLM_clients.vllm_docs_qa import VllmDocsQA

logger = logging.getLogger(__name__)


class VllmUnavailableError(RuntimeError):
    """The server has been silent longer than the threshold; training must stop.

    Unlike an ordinary request error, there is nothing to ride it out with:
    while the server is down every episode is left without a reward, and
    continuing would only log hours of empty curves. Caught in the training
    loop, which saves a checkpoint and exits.
    """

OPEN_QA_TASKS = {
    "HotPotQA",
    "Musique",
    "2WikiMultihopQA",
    "HotPotQA+2WikiMultihopQA",
    "NQ+HotPotQA",
}


def prepare_examples(examples):
    prepared = []
    for example in examples:
        parts = example.split(separator, 1)
        if len(parts) == 2:
            prepared.append((parts[0].strip(), parts[1].strip()))
    return prepared


def discard_reasoning(response: str) -> str:
    if "<think>" not in response:
        return response
    if "</think>" not in response:
        return "[incomplete reasoning]"
    return response.rsplit("</think>", 1)[1].strip()


def get_final_answer(text: str) -> str:
    marker = "final answer:"
    position = text.lower().rfind(marker)
    if position != -1:
        text = text[position + len(marker) :]
    return text.strip().rstrip(".")


def answer_variants(info: dict) -> List[str]:
    """Accepted answers of an episode, always a list, even with a single item.

    NQ has up to 25 aliases, HotpotQA exactly one. Datasets without aliases
    do not provide the field, and the list is built from the single
    ``answer``, so their reward does not change at all.
    """
    variants = info.get("answer_variants")
    if variants is None:
        variants = info.get("answer")
    if isinstance(variants, (list, tuple)):
        # An empty list means no gold, not an empty answer; there is nothing to
        # compare with, and max() over an empty sequence would crash inside the
        # reward, i.e. in a pool worker thread.
        return [str(item).strip() for item in variants] or [""]
    return [str(variants).strip()]


class LlmAnswer(AFeedbackModel):
    FEEDBACK_MODEL_NAME = "LLM-Answer"

    def __init__(self,
        model: str,
        base_url: str,
        max_tokens: int,
        thinking: bool,
        task: str,
        api_key: Optional[str] = None,
        retries: int = 3,
        retry_backoff: float = 1.0,
        retry_backoff_max: float = 30.0,
        stall_timeout: Optional[float] = 900.0,
        judge_max_tokens: int = 100,
    ):
        super().__init__(never_terminate=True)
        if task not in prompts.sys_prompts:
            raise ValueError(f"Unsupported LlmAnswer task: {task}")
        if retries < 1:
            raise ValueError(f"retries must be positive: {retries}")
        if thinking:
            # The evaluation reader is hardwired to thinking=False, and the reward
            # must be the same quantity as the evaluation metric. A reasoning
            # reader gives different answers, i.e. a different reward for the same
            # retrieval; silently diverging is worse than failing.
            raise ValueError(
                "thinking=True would diverge from evaluation, where the reader "
                "is always non-reasoning"
            )

        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.task = task
        self.api_key = api_key
        self.retries = retries
        self.retry_backoff = retry_backoff
        self.retry_backoff_max = retry_backoff_max
        self.stall_timeout = stall_timeout
        self.judge_max_tokens = judge_max_tokens
        # Error counters: without them a server outage looks like a reward
        # collapse in the log and cannot be told apart from policy degradation.
        self.error_count = 0
        self.request_count = 0
        self.last_transition_valid = True
        self._first_failure_at: Optional[float] = None
        self.rouge = Rouge()

        self.vllm_client = BasicVllmClient(
            model=model,
            base_url=base_url,
            sys_prompt=prompts.sys_prompts[task],
            api_key=api_key,
            max_tokens=max_tokens,
            thinking=thinking,
        )

        # Reward contract v2 judge: the reference is a list of aliases, hence
        # the plural prompt. max_tokens 100 as in evaluation: the verdict is one
        # line, and 500 gave the judge room to drift into reasoning.
        self.vllm_client_judge = BasicVllmClient(
            model=model,
            base_url=base_url,
            sys_prompt=prompts.sys_judge_v2,
            api_key=api_key,
            max_tokens=judge_max_tokens,
            thinking=False,
        )

        self.last_metrics = {}


    def copy(self):
        return LlmAnswer(
            model=self.model,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            thinking=self.thinking,
            task=self.task,
            api_key=self.api_key,
            retries=self.retries,
            retry_backoff=self.retry_backoff,
            retry_backoff_max=self.retry_backoff_max,
            stall_timeout=self.stall_timeout,
            judge_max_tokens=self.judge_max_tokens,
        )


    def reset(self, obs: dict | List[dict], info: dict | List[dict]) -> None:
        super().reset(obs, info)
        self.last_metrics = {}
        self.last_transition_valid = True


    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)


    def _call_with_retries(self, client, **kwargs):
        """vLLM request with retries and exponential backoff.

        A network error and "context is useless" are different events and must
        not both reach training as zero. The request is retried first, and if
        the server keeps failing for longer than `stall_timeout`,
        `VllmUnavailableError` is raised: there is nothing to continue with.
        """
        delay = self.retry_backoff
        last_error = None
        for attempt in range(1, self.retries + 1):
            self.request_count += 1
            try:
                result = client.chat_completion(**kwargs)
            except Exception as error:  # network, timeout, 5xx all land here
                last_error = error
                self.error_count += 1
                if self._first_failure_at is None:
                    self._first_failure_at = time.monotonic()
                stalled = time.monotonic() - self._first_failure_at
                if self.stall_timeout is not None and stalled >= self.stall_timeout:
                    raise VllmUnavailableError(
                        f"vLLM at {self.base_url} has been failing for "
                        f"{stalled:.0f}s (>{self.stall_timeout:.0f}s): {error}"
                    ) from error
                logger.warning(
                    "vLLM request failed (attempt %d/%d, %.0fs since first "
                    "failure): %s",
                    attempt,
                    self.retries,
                    stalled,
                    error,
                )
                if attempt < self.retries and delay > 0:
                    self._sleep(delay)
                    delay = min(delay * 2, self.retry_backoff_max)
            else:
                self._first_failure_at = None
                return result
        raise RuntimeError(
            f"vLLM request failed {self.retries} times: {last_error}"
        ) from last_error


    def reward(self, obs, info, is_final):
        self.last_transition_valid = True
        if not is_final:
            return 0.0

        question = obs["question"]
        variants = answer_variants(info)
        answer = variants[0]
        chunks = obs.get("pred_chunks", [])
        examples = None
        passages = chunks

        if self.task not in OPEN_QA_TASKS:
            examples = prepare_examples(chunks)
            passages = None

        prediction = ""
        judgment = ""
        reward = 0.0
        em = 0
        em_alias = 0
        f1 = 0.0
        judged = False

        try:
            response, mean_logprob = self._call_with_retries(
                self.vllm_client,
                query=question,
                examples=examples,
                passages=passages,
            )
            prediction = discard_reasoning(response).strip()

            if self.task in OPEN_QA_TASKS or self.task == "GSM8K":
                prediction = get_final_answer(prediction)
            elif self.task == "MATH":
                prediction = extract_boxed_expression(prediction)

            # normalize_answer instead of strip().lower(): the reward semantics
            # are unchanged (judge strictness was checked, no disagreements), but
            # it saves judge calls on answers differing only in punctuation and
            # articles.
            normalized_prediction = normalize_answer(prediction)
            normalized_answer = normalize_answer(answer)

            # EM against the primary answer is logged alongside: the final metric
            # is plain EM, and its gap to the reward must be visible at every
            # eval point, not only in a post-hoc analysis.
            em = int(normalized_prediction == normalized_answer)
            em_alias = max(
                int(normalized_prediction == normalize_answer(variant))
                for variant in variants
            )
            if self.task == "XL-Sum" and normalized_prediction:
                scores = self.rouge.get_scores(normalized_prediction, normalized_answer)
                f1 = float(scores[0]["rouge-l"]["f"])
                reward = f1
            elif em_alias:
                reward = 1.0
            else:
                # The judge gets raw strings, with all aliases joined by " | " as
                # the reference: normalization strips the punctuation and articles
                # that let the judge tell "1,000" from "one thousand".
                judged = True
                judgment, _ = self._call_with_retries(
                    self.vllm_client_judge,
                    query=question,
                    two_answers=(prediction, " | ".join(variants)),
                )
                judgment = get_final_answer(judgment).upper()
                reward = float(judgment == "CORRECT")

        except VllmUnavailableError:
            raise
        except Exception as exc:
            mean_logprob = None
            # Retries exhausted: there is no reward, and zero would be made up.
            # The transition is marked invalid and excluded from the loss.
            self.last_transition_valid = False
            logger.error("LlmAnswer reward failed: %s", exc)

        # The reward is the max of alias EM and the judge verdict, so both parts
        # are logged separately: reward == em_alias + judge_rescue, and without
        # the second term it is unclear what drives the curve.
        self.last_metrics = {
            "pred": prediction,
            "EM": em,
            "em_alias": em_alias,
            "F1": f1,
            "LLM-as-judge": reward,
            # Judge verdict only where the judge was called: an imputed 1 on an
            # EM match would mix imputation with verdicts.
            "judge": reward if judged else None,
            "answer_variants": len(variants),
            "mean_logprob": mean_logprob,
            "judgment": judgment,
            "valid": self.last_transition_valid,
        }
        return reward


    def get_metrics(self):
        return dict(self.last_metrics)


    def error_stats(self) -> dict:
        """Monitoring counters: vLLM errors go to the same log as the reward."""
        return {
            "vllm_requests": self.request_count,
            "vllm_errors": self.error_count,
            "vllm_failing": self._first_failure_at is not None,
        }
