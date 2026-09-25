"""Офлайн-регрессии награды: новый код обязан повторить сохранённые числа.

vLLM здесь не нужен — предсказания ридера уже лежат в ранах, а проверяется
именно то, что делает с ними награда. Два прогона:

* 7 405 предсказаний канонического ``2026-07-28-gte-only-steps6`` (у HotpotQA
  ровно один допустимый ответ): EM обязан совпасть с сохранённым на каждой
  записи — иначе контракт v2 сдвинул одноответный случай;
* 3 610 предсказаний ``2026-08-07-sr1-noctx-nq`` с таблицей вариантов
  ``runs/shared/searchr1/nq.jsonl``: ``em_alias`` обязан дать 11.97% при
  среднем 1.80 варианта на пример. Ноль алиасов значил бы, что таблица не
  подхватилась, а первая цифра совпала случайно.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_reward_contract_v2 import FakeClient, make_feedback


LAB = Path(__file__).resolve().parents[2]
JUDGE_FILE = LAB / "runs" / "2026-07-28-gte-only-steps6" / "answer_judge.json"
NQ_RUN = LAB / "runs" / "2026-08-07-sr1-noctx-nq" / "answer_judge.json"
NQ_ALIASES = LAB / "runs" / "shared" / "searchr1" / "nq.jsonl"


class EndlessJudge(FakeClient):
    """Судья, который всегда говорит INCORRECT.

    Регрессия меряет EM, а не вердикты: канонический ответ судьи здесь нужен
    только чтобы ветка `em_alias == 0` не ходила в сеть.
    """

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return "Final Answer: INCORRECT", None


def replay(record_prediction: str, variants: list[str]) -> dict:
    """Прогнать награду на сохранённом предсказании, не трогая сеть."""
    feedback = make_feedback([f"Final Answer: {record_prediction}"])
    feedback.vllm_client_judge = EndlessJudge([])
    feedback.reward(
        {"question": "q", "sample_id": "x", "pred_idx": [], "pred_chunks": ["c"]},
        {"answer": variants[0], "answer_variants": variants},
        is_final=True,
    )
    return feedback.get_metrics()


@pytest.mark.skipif(not JUDGE_FILE.is_file(), reason=f"нет {JUDGE_FILE}")
def test_single_answer_em_matches_the_saved_run_on_every_record() -> None:
    records = json.loads(JUDGE_FILE.read_text(encoding="utf-8"))
    assert len(records) == 7405

    mismatches = []
    for index, record in enumerate(records):
        metrics = replay(record["prediction"] or "", [str(record["answer"])])
        if metrics["EM"] != float(record["EM"]):
            mismatches.append((index, record.get("id")))
        # Одноответный случай: alias-версия обязана совпасть с основной.
        assert metrics["em_alias"] == metrics["EM"]
    assert not mismatches, mismatches[:5]


@pytest.mark.skipif(
    not (NQ_RUN.is_file() and NQ_ALIASES.is_file()),
    reason="нет сохранённого рана NQ или таблицы вариантов",
)
def test_alias_em_reproduces_the_saved_nq_metric() -> None:
    records = json.loads(NQ_RUN.read_text(encoding="utf-8"))
    aliases = {}
    with NQ_ALIASES.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                item = json.loads(line)
                aliases[str(item["_id"])] = [
                    str(value) for value in item.get("answer_aliases") or []
                ]
    assert len(records) == 3610

    em_alias = 0
    em_plain = 0
    variant_count = 0
    for record in records:
        variants = [str(record["answer"]), *aliases.get(str(record["id"]), [])]
        variant_count += len(variants)
        metrics = replay(record["prediction"] or "", variants)
        em_alias += metrics["em_alias"]
        em_plain += metrics["EM"]

    # Сохранённое metrics.json того же рана: em 9.25%, em_alias 11.97%.
    assert em_alias / len(records) == pytest.approx(0.11966759, abs=5e-8)
    assert em_plain / len(records) == pytest.approx(0.09252078, abs=5e-8)
    # Ноль алиасов дал бы ту же первую цифру случайно — проверяем, что
    # таблица действительно подхватилась.
    assert variant_count / len(records) == pytest.approx(1.7978, abs=1e-4)
