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
    """Сервер молчит дольше порога — обучение обязано остановиться.

    Отличается от обычной ошибки запроса тем, что переживать её нечем:
    пока сервер лежит, каждый эпизод остаётся без награды, и продолжать
    значит писать в лог часы пустых кривых. Ловится в train-цикле, который
    сохраняет чекпоинт и выходит.
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
    """Допустимые ответы эпизода — всегда список, даже когда он из одного.

    У NQ вариантов до 25, у HotpotQA ровно один. Датасеты без вариантов поля
    не приносят, и тогда список собирается из единственного ``answer``:
    награда на них не меняется ни на бит.
    """
    variants = info.get("answer_variants")
    if variants is None:
        variants = info.get("answer")
    if isinstance(variants, (list, tuple)):
        # Пустой список — не «ответ пустой», а отсутствие голда; сравнивать
        # с ним нечего, и max() по пустой последовательности упал бы внутри
        # награды, то есть в потоке пула.
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
            # Ридер эвала создаётся с thinking=False жёстко, и награда обязана
            # быть той же величиной, что и колонка RESULTS.md. Рассуждающий
            # ридер даёт другие ответы, то есть другую награду при том же
            # ретривале — молча разойтись здесь дороже, чем упасть.
            raise ValueError(
                "thinking=True разошёлся бы с эвалом, где ридер всегда "
                "нерассуждающий"
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
        # Счётчики ошибок: без них падение сервера выглядит в логе просто как
        # обвал награды, и отличить его от распада политики невозможно.
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

        # Судья контракта v2: эталон — список вариантов, отсюда и промпт с
        # множественным числом. max_tokens 100, как у эвала: вердикт — одна
        # строка, а 500 давали судье место уехать в рассуждение.
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
        """Запрос к vLLM с ретраями и экспоненциальным бэкоффом.

        Сетевая ошибка и «контекст бесполезен» — разные события, и раньше они
        приходили к обучению одним и тем же нулём. Здесь запрос сначала
        повторяется, а если сервер молчит дольше `stall_timeout` подряд —
        поднимается `VllmUnavailableError`: продолжать нечем.
        """
        delay = self.retry_backoff
        last_error = None
        for attempt in range(1, self.retries + 1):
            self.request_count += 1
            try:
                result = client.chat_completion(**kwargs)
            except Exception as error:  # сеть, таймаут, 5xx — всё сюда
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

            # normalize_answer вместо strip().lower(): семантику награды это не
            # меняет — строгость судьи проверена, расхождений нет, — но снимает
            # вызовы судьи на ответах, отличающихся только пунктуацией и
            # артиклями.
            normalized_prediction = normalize_answer(prediction)
            normalized_answer = normalize_answer(answer)

            # EM по основному ответу логируется рядом: итоговая метрика ветки —
            # чистый EM, и его расхождение с наградой обязано быть видно на
            # каждой точке eval, а не в разборе потом.
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
                # Судье уходят сырые строки, а эталон — все варианты через
                # " | ": нормализация выкидывает пунктуацию и артикли, из-за
                # которых судья и отличает «1,000» от «one thousand».
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
            # Ретраи исчерпаны: награды нет, и ноль здесь был бы выдумкой.
            # Переход помечается невалидным и в лосс не попадает.
            self.last_transition_valid = False
            logger.error("LlmAnswer reward failed: %s", exc)

        # Награда — максимум EM по вариантам и вердикта судьи, поэтому обе её
        # половины пишутся раздельно: reward == em_alias + judge_rescue, и без
        # второй величины непонятно, чем именно растёт кривая.
        self.last_metrics = {
            "pred": prediction,
            "EM": em,
            "em_alias": em_alias,
            "F1": f1,
            "LLM-as-judge": reward,
            # Вердикт судьи — только там, где судью звали: вменённая единица
            # на совпавшем EM смешала бы вменение с вердиктом.
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
        """Счётчики для мониторинга: ошибки vLLM идут в тот же лог, что и награда."""
        return {
            "vllm_requests": self.request_count,
            "vllm_errors": self.error_count,
            "vllm_failing": self._first_failure_at is not None,
        }
