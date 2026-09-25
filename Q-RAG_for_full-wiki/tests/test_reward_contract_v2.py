"""Контракт награды v2: варианты ответа и совпадение запросов с эвалом.

Награда обучения и колонка RESULTS.md обязаны быть одной величиной. Эвал
(`Q-RAG-feedback/answer_judge_llms.py`) править нельзя, поэтому эталон — он, а
проверяется здесь ровно то, чем обучение от него отличалось: разделитель
чанков, что уходит судье, `max_tokens` судьи и рассуждающий ридер.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts_and_metrics import prompts
from rl.feedback.llm_answer import LlmAnswer, answer_variants
from vLLM_clients.vllm_client import BasicVllmClient


EVAL_SYS_QA = prompts.sys_qa


class FakeClient:
    """Клиент вместо vLLM: запоминает запросы и отдаёт заготовленные ответы."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("Лишний запрос к vLLM")
        return self.replies.pop(0), None


def make_feedback(reader_replies, judge_replies=()):
    feedback = LlmAnswer.__new__(LlmAnswer)
    feedback.task = "NQ+HotPotQA"
    feedback.rouge = None
    feedback.retries = 1
    feedback.retry_backoff = 0.0
    feedback.retry_backoff_max = 0.0
    feedback.stall_timeout = None
    feedback.error_count = 0
    feedback.request_count = 0
    feedback.last_transition_valid = True
    feedback._first_failure_at = None
    feedback.last_metrics = {}
    feedback.base_url = "http://fake/v1"
    feedback.vllm_client = FakeClient(reader_replies)
    feedback.vllm_client_judge = FakeClient(judge_replies)
    return feedback


def reward_of(feedback, variants, prediction_chunks=("chunk one", "chunk two")):
    obs = {
        "question": "who wrote it?",
        "sample_id": "x",
        "pred_idx": [0, 1],
        "pred_chunks": list(prediction_chunks),
    }
    info = {"answer": variants[0], "answer_variants": list(variants)}
    return feedback.reward(obs, info, is_final=True)


# --------------------------------------------------------------------------
# варианты ответа


def test_single_variant_reward_and_metrics_are_unchanged() -> None:
    """Пример с одним ответом: EM срабатывает, судью не зовут."""
    feedback = make_feedback(["Final Answer: The Beatles"])
    assert reward_of(feedback, ["the beatles"]) == 1.0
    metrics = feedback.get_metrics()
    assert metrics["EM"] == 1
    assert metrics["em_alias"] == 1
    assert metrics["judge"] is None
    assert feedback.vllm_client_judge.calls == []


def test_second_variant_wins_where_the_first_one_misses() -> None:
    """Многовариантный пример: EM берёт максимум, судью звать не за чем.

    Ровно это и ломала склейка через ', ': предсказание сравнивалось со
    строкой «Dai Xiuli, Dai Yongge, Yongge Dai», не совпадало ни с чем и
    уезжало к судье.
    """
    feedback = make_feedback(["Final Answer: Dai Yongge"])
    variants = ["Xiu Li Dai", "Dai Xiuli", "Dai Yongge", "Yongge Dai"]
    assert reward_of(feedback, variants) == 1.0
    metrics = feedback.get_metrics()
    # EM по основному ответу — ноль, и он логируется отдельно: итоговая
    # метрика ветки не должна незаметно подмениться alias-версией.
    assert metrics["EM"] == 0
    assert metrics["em_alias"] == 1
    assert metrics["answer_variants"] == 4
    assert feedback.vllm_client_judge.calls == []


def test_joined_variants_would_have_missed() -> None:
    """Прежнее поведение: склейка не совпадает ни с одним вариантом.

    Тот же ответ на том же примере при склейке через ', ' даёт EM = 0 и уходит
    к судье — цена, которую платил каждый второй пример NQ.
    """
    feedback = make_feedback(
        ["Final Answer: Dai Yongge"], ["Final Answer: INCORRECT"]
    )
    joined = ", ".join(["Xiu Li Dai", "Dai Xiuli", "Dai Yongge"])
    assert reward_of(feedback, [joined]) == 0.0
    assert len(feedback.vllm_client_judge.calls) == 1


def test_judge_is_called_with_all_variants_and_raw_strings() -> None:
    feedback = make_feedback(
        ["Final Answer: a thousand"], ["Final Answer: CORRECT"]
    )
    assert reward_of(feedback, ["1,000", "one thousand"]) == 1.0
    call = feedback.vllm_client_judge.calls[0]
    prediction, reference = call["two_answers"]
    # Сырые строки, а не normalize_answer: судья только по ним и отличает
    # «1,000» от «one thousand» — нормализация выкидывает пунктуацию.
    assert prediction == "a thousand"
    assert reference == "1,000 | one thousand"
    metrics = feedback.get_metrics()
    assert metrics["em_alias"] == 0
    assert metrics["judge"] == 1.0


def test_answer_variants_falls_back_to_the_single_answer() -> None:
    assert answer_variants({"answer": "yes"}) == ["yes"]
    assert answer_variants({"answer_variants": ["a", "b"]}) == ["a", "b"]
    assert answer_variants({"answer_variants": []}) == [""]


# --------------------------------------------------------------------------
# четыре расхождения с эвалом


def eval_reader_request(chunks, question):
    """Запрос ридера ровно так, как его собирает answer_judge_llms.py."""
    context = "\n\n".join(chunks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFinal Answer:"


def test_reader_request_is_byte_identical_to_the_eval_one() -> None:
    client = BasicVllmClient.__new__(BasicVllmClient)
    client._system_message = {"role": "system", "content": EVAL_SYS_QA}
    chunks = ['"Alpha"\nfirst chunk text', '"Beta"\nsecond chunk text']
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"

    messages = client._prepare_messages(query=question, passages=chunks)

    assert messages[0]["content"] == EVAL_SYS_QA
    assert messages[1]["content"] == eval_reader_request(chunks, question)


def test_judge_prompt_differs_from_v1_only_in_the_reference_description() -> None:
    v1 = prompts.sys_judge
    v2 = prompts.sys_judge_v2
    assert v1 != v2
    assert prompts.JUDGE_REFERENCE_V1 in v1
    assert prompts.JUDGE_REFERENCE_V2 in v2
    # Строгость судьи не трогаем: меняется только описание эталона.
    assert v2 == v1.replace(
        prompts.JUDGE_REFERENCE_V1, prompts.JUDGE_REFERENCE_V2
    )


def test_thinking_reader_is_refused() -> None:
    with pytest.raises(ValueError, match="thinking"):
        LlmAnswer(
            model="Qwen3-4B",
            base_url="http://fake/v1",
            max_tokens=1000,
            thinking=True,
            task="NQ+HotPotQA",
        )


def test_judge_max_tokens_defaults_to_the_eval_value() -> None:
    import inspect

    signature = inspect.signature(LlmAnswer.__init__)
    assert signature.parameters["judge_max_tokens"].default == 100
