"""Offline reward regressions: the current code must reproduce saved numbers.

No vLLM is needed: the reader predictions are already stored in the runs, and
what is tested is what the reward does with them. Two runs:

* 7,405 predictions of the canonical ``2026-07-28-gte-only-steps6`` run
  (HotpotQA has exactly one accepted answer): EM must match the saved value on
  every record, otherwise reward contract v2 changed the single-answer case;
* 3,610 predictions of ``2026-08-07-sr1-noctx-nq`` with the alias table
  ``runs/shared/searchr1/nq.jsonl``: ``em_alias`` must give 11.97% with
  1.80 accepted answers per example on average. Zero aliases would mean the
  table was not picked up and the first number matched by chance.
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
    """Judge that always says INCORRECT.

    The regression measures EM, not verdicts: a canned judge reply is needed
    only so that the `em_alias == 0` branch does not hit the network.
    """

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return "Final Answer: INCORRECT", None


def replay(record_prediction: str, variants: list[str]) -> dict:
    """Run the reward on a saved prediction without touching the network."""
    feedback = make_feedback([f"Final Answer: {record_prediction}"])
    feedback.vllm_client_judge = EndlessJudge([])
    feedback.reward(
        {"question": "q", "sample_id": "x", "pred_idx": [], "pred_chunks": ["c"]},
        {"answer": variants[0], "answer_variants": variants},
        is_final=True,
    )
    return feedback.get_metrics()


@pytest.mark.skipif(not JUDGE_FILE.is_file(), reason=f"missing {JUDGE_FILE}")
def test_single_answer_em_matches_the_saved_run_on_every_record() -> None:
    records = json.loads(JUDGE_FILE.read_text(encoding="utf-8"))
    assert len(records) == 7405

    mismatches = []
    for index, record in enumerate(records):
        metrics = replay(record["prediction"] or "", [str(record["answer"])])
        if metrics["EM"] != float(record["EM"]):
            mismatches.append((index, record.get("id")))
        # Single-answer case: the alias version must equal the primary one.
        assert metrics["em_alias"] == metrics["EM"]
    assert not mismatches, mismatches[:5]


@pytest.mark.skipif(
    not (NQ_RUN.is_file() and NQ_ALIASES.is_file()),
    reason="saved NQ run or alias table is missing",
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

    # The saved metrics.json of the same run: em 9.25%, em_alias 11.97%.
    assert em_alias / len(records) == pytest.approx(0.11966759, abs=5e-8)
    assert em_plain / len(records) == pytest.approx(0.09252078, abs=5e-8)
    # With zero aliases the first number could match by chance, so check that
    # the table was actually picked up.
    assert variant_count / len(records) == pytest.approx(1.7978, abs=1e-4)
