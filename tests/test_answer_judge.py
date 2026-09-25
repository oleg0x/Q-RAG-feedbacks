"""Форк ридера и судьи: контракт v1 бит в бит, контракт v2 по вариантам.

Сеть здесь не нужна. Проверяется то, чем форк отличается от read-only
оригинала: собранные запросы, промпт судьи, выбор варианта и то, что
вменённая единица помечена отдельным полем.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

import answer_judge


# Тесты лежат в tests/, а реестр ранов и внешние репозитории — в корне.
LAB = Path(__file__).resolve().parent.parent
TRAIN_COPY_PROMPTS = LAB / "Q-RAG_for_full-wiki" / "prompts_and_metrics" / "prompts.py"

# Read-only оригинал форка. Форк самодостаточен (клиент и промпты вендорены в
# src/), поэтому без оригинала регрессия против него пропускается, а не падает.
FEEDBACK_REPO = LAB.parent / "Q-RAG-feedback"

needs_feedback = pytest.mark.skipif(
    not FEEDBACK_REPO.is_dir(),
    reason="../Q-RAG-feedback недоступен — регрессия против read-only оригинала "
    "идёт только на машине лаборатории",
)


def import_feedback(name: str):
    if str(FEEDBACK_REPO) not in sys.path:
        sys.path.append(str(FEEDBACK_REPO))
    return importlib.import_module(name)
JUDGE_FILE = LAB / "runs" / "runs_history" / "2026-07-28-gte-only-steps6" / "answer_judge.json"
NQ_JUDGE = LAB / "runs" / "runs_history" / "2026-08-07-sr1-noctx-nq" / "answer_judge.json"
RETRIEVAL_NQ = LAB / "runs" / "runs_history" / "2026-08-07-sr1-noctx-nq" / "retrieval.jsonl"
NQ_ALIASES = LAB / "runs" / "shared" / "searchr1" / "nq.jsonl"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# запросы и промпты


def test_reader_request_matches_the_read_only_original() -> None:
    """Оригинал склеивает чанки через "\\n\\n" — форк обязан делать так же."""
    sample = {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "pred_texts": ['"Alpha"\nfirst', '"Beta"\nsecond'],
    }
    expected = (
        'CONTEXT:\n"Alpha"\nfirst\n\n"Beta"\nsecond\n\n'
        "QUESTION:\nWere Scott Derrickson and Ed Wood of the same nationality?"
        "\n\nFinal Answer:"
    )
    assert answer_judge.build_reader_request(sample) == expected


def test_judge_request_matches_the_read_only_original() -> None:
    assert answer_judge.build_judge_request("q?", "pred", "gold") == (
        "QUESTION: q?\nPREDICTED ANSWER: pred\nGROUNDTRUTH ANSWER: gold"
    )


@needs_feedback
def test_v1_judge_prompt_is_the_original_one() -> None:
    prompts = import_feedback("prompts_and_metrics.prompts")

    assert answer_judge.judge_prompt("v1") == prompts.sys_judge


def test_v2_judge_prompt_matches_the_training_copy() -> None:
    """Награда обучения и эвал обязаны звать судью одним и тем же промптом."""
    training = load_module(TRAIN_COPY_PROMPTS, "training_prompts")
    assert answer_judge.judge_prompt("v2") == training.sys_judge_v2
    assert answer_judge.JUDGE_REFERENCE_V2 == training.JUDGE_REFERENCE_V2


# --------------------------------------------------------------------------
# метрики


@needs_feedback
def test_metrics_are_a_verbatim_copy_of_the_original() -> None:
    answer_judge_llms = import_feedback("answer_judge_llms")

    for prediction, target in (
        ("The Beatles", "beatles"),
        ("a 1000", "1,000"),
        ("", "yes"),
        ("New York City", "new york"),
    ):
        assert answer_judge.normalize_answer(prediction) == (
            answer_judge_llms.normalize_answer(prediction)
        )
        assert answer_judge.exact_match(prediction, target) == (
            answer_judge_llms.exact_match(prediction, target)
        )
        assert answer_judge.f1_score(prediction, target) == pytest.approx(
            answer_judge_llms.f1_score(prediction, target)
        )
        assert answer_judge.final_answer(prediction) == (
            answer_judge_llms.final_answer(prediction)
        )


def test_best_over_aliases_takes_the_maximum() -> None:
    variants = ["Xiu Li Dai", "Dai Xiuli", "Dai Yongge"]
    assert answer_judge.best_over_aliases("Dai Yongge", variants)[0] == 1
    assert answer_judge.best_over_aliases("Nobody", variants)[0] == 0
    # Единственный вариант — та же величина, что и основной EM.
    assert answer_judge.best_over_aliases("yes", ["yes"]) == (1, 1.0)


def test_variants_take_aliases_from_the_table_not_from_retrieval() -> None:
    sample = {"id": "nq_test_2", "answer": "Olivia"}
    aliases = {"nq_test_2": ["MFSK"]}
    assert answer_judge.variants_of(sample, aliases, 0) == ["Olivia", "MFSK"]
    # Ретривал алиасы роняет: без таблицы вариант остаётся один.
    assert answer_judge.variants_of(sample, {}, 0) == ["Olivia"]


# --------------------------------------------------------------------------
# сквозной прогон без сети


class FakeReader:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def chat_completion(self, request, examples=None):
        self.requests.append(request)
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def run_fork(
    monkeypatch, tmp_path, samples, reader_replies, judge_replies, argv, module=None
):
    module = module or answer_judge
    clients = []

    def factory(**kwargs):
        client = FakeReader(reader_replies if not clients else judge_replies)
        clients.append(client)
        return client

    monkeypatch.setattr(module, "SyncVllmClient", factory)
    monkeypatch.setattr(
        module,
        "extract_response_text",
        lambda payload: payload["choices"][0]["message"]["content"].strip(),
    )
    source = tmp_path / "retrieval.jsonl"
    source.write_text(
        "\n".join(json.dumps(item) for item in samples) + "\n", encoding="utf-8"
    )
    output = tmp_path / f"answer_judge-{module.__name__}.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "answer_judge.py",
            "--retriever-logfile", str(source),
            "--output-file", str(output),
            "--base-url", "http://fake/v1",
            "--answer-model", "Qwen3-4B",
            "--max-samples", "100",
            *argv,
        ],
    )
    module.main()
    return json.loads(output.read_text(encoding="utf-8")), clients


SAMPLES = [
    {"id": "a", "question": "q1?", "answer": "Olivia", "pred_texts": ["c1"]},
    {"id": "b", "question": "q2?", "answer": "Foreigner", "pred_texts": ["c2"]},
]


def test_v2_skips_the_judge_where_an_alias_matched(monkeypatch, tmp_path) -> None:
    aliases = tmp_path / "aliases.jsonl"
    aliases.write_text(
        json.dumps(
            {"_id": "a", "question": "q1?", "answer": "Olivia", "answer_aliases": ["MFSK"]}
        )
        + "\n"
        + json.dumps(
            {"_id": "b", "question": "q2?", "answer": "Foreigner", "answer_aliases": []}
        )
        + "\n",
        encoding="utf-8",
    )
    results, clients = run_fork(
        monkeypatch,
        tmp_path,
        SAMPLES,
        ["Final Answer: MFSK", "Final Answer: nobody"],
        ["Final Answer: INCORRECT"],
        ["--contract", "v2", "--aliases", str(aliases)],
    )

    first, second = results
    assert first["em_alias"] == 1 and first["EM"] == 0
    assert first["answer_variants"] == ["Olivia", "MFSK"]
    assert first["reward"] == 1.0
    # Вердикта не было — единица вменена, и это видно по полю.
    assert first["judge_imputed"] is True
    assert second["em_alias"] == 0 and second["judge_imputed"] is False
    assert second["reward"] == 0.0
    # Судью звали ровно один раз: только там, где EM не сработал.
    assert len(clients[1].requests) == 1
    assert "GROUNDTRUTH ANSWER: Foreigner" in clients[1].requests[0]


def test_judge_all_asks_on_every_sample_and_reward_stays_the_maximum(
    monkeypatch, tmp_path
) -> None:
    results, clients = run_fork(
        monkeypatch,
        tmp_path,
        SAMPLES,
        ["Final Answer: Olivia", "Final Answer: nobody"],
        ["Final Answer: INCORRECT", "Final Answer: INCORRECT"],
        ["--contract", "v2", "--judge-all"],
    )
    assert len(clients[1].requests) == 2
    assert all(item["judge_imputed"] is False for item in results)
    # Судья сказал INCORRECT там, где EM совпал: награда — максимум.
    assert results[0]["em_alias"] == 1
    assert results[0]["LLM_Judge_Score"] == 0.0
    assert results[0]["reward"] == 1.0


@needs_feedback
def test_requests_are_byte_identical_to_the_original_on_the_same_input(
    monkeypatch, tmp_path
) -> None:
    """Оба скрипта прогоняются на одном входе, запросы сверяются посимвольно.

    Сравнение с оригиналом, а не с переписанной от руки строкой: только так
    ловится расхождение, которое появится в оригинале завтра.
    """
    if not RETRIEVAL_NQ.is_file():
        pytest.skip(f"нет {RETRIEVAL_NQ}")
    answer_judge_llms = import_feedback("answer_judge_llms")

    samples = []
    with RETRIEVAL_NQ.open(encoding="utf-8") as source:
        for line in source:
            samples.append(json.loads(line))
            if len(samples) == 50:
                break
    replies = [f"Final Answer: guess {i}" for i in range(len(samples))]
    verdicts = ["Final Answer: CORRECT"] * len(samples)

    ours, our_clients = run_fork(
        monkeypatch, tmp_path, samples, list(replies), list(verdicts),
        ["--contract", "v1"],
    )
    theirs, their_clients = run_fork(
        monkeypatch,
        tmp_path,
        samples,
        list(replies),
        list(verdicts),
        [],
        module=answer_judge_llms,
    )

    assert our_clients[0].requests == their_clients[0].requests
    assert our_clients[1].requests == their_clients[1].requests
    assert ours == theirs


@pytest.mark.skipif(not JUDGE_FILE.is_file(), reason="нет сохранённого рана")
def test_em_and_f1_reproduce_the_saved_run_on_every_record() -> None:
    records = json.loads(JUDGE_FILE.read_text(encoding="utf-8"))
    assert len(records) == 7405
    for record in records:
        prediction = record["prediction"] or ""
        assert answer_judge.exact_match(prediction, record["answer"]) == record["EM"]
        assert answer_judge.f1_score(prediction, record["answer"]) == pytest.approx(
            record["F1"], abs=1e-12
        )


@pytest.mark.skipif(
    not (NQ_JUDGE.is_file() and NQ_ALIASES.is_file()),
    reason="нет сохранённого рана NQ или таблицы вариантов",
)
def test_alias_branch_reproduces_the_saved_nq_metric() -> None:
    """Ветка алиасов: em_alias 11.97% при среднем 1.80 варианта на пример."""
    records = json.loads(NQ_JUDGE.read_text(encoding="utf-8"))
    aliases = {}
    with NQ_ALIASES.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                aliases[str(item["_id"])] = [
                    str(value) for value in item.get("answer_aliases") or []
                ]
    em_alias = f1_alias = variants = 0
    for index, record in enumerate(records):
        allowed = answer_judge.variants_of(record, aliases, index)
        variants += len(allowed)
        em, f1 = answer_judge.best_over_aliases(record["prediction"] or "", allowed)
        em_alias += em
        f1_alias += f1
    total = len(records)
    assert em_alias / total == pytest.approx(0.11966759, abs=5e-8)
    assert f1_alias / total == pytest.approx(0.20312058, abs=5e-8)
    # Ноль алиасов дал бы ту же первую цифру случайно.
    assert variants / total == pytest.approx(1.7978, abs=1e-4)


def test_v1_keeps_the_old_fields_and_judges_everything(monkeypatch, tmp_path) -> None:
    results, clients = run_fork(
        monkeypatch,
        tmp_path,
        SAMPLES,
        ["Final Answer: Olivia", "Final Answer: nobody"],
        ["Final Answer: CORRECT", "Final Answer: INCORRECT"],
        ["--contract", "v1"],
    )
    assert len(clients[1].requests) == 2
    for item in results:
        assert set(item) >= {"prediction", "EM", "F1", "LLM_Judge_Score"}
        assert "em_alias" not in item
        assert "reward" not in item
        assert "judge_imputed" not in item
