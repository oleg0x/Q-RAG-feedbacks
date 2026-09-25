import json

from envs import (
    CandidateDatasetAdapter,
    CombinedCandidateDataset,
    Retrieval2WikiMultihopQA,
    RetrievalHotPotQA,
)


def write_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as destination:
        for sample in samples:
            destination.write(json.dumps(sample) + "\n")


def sample(source="hotpotqa"):
    return {
        "_id": f"{source}-1",
        "question": "Question stays verbatim?",
        "answer": "answer",
        "supporting_facts": [["Title", 0]],
        "context": [["Title", ["Sentence."]]],
        "candidates": ["answer", "wrong"],
        "judgements": [1, 0],
        "betas": [-1.0, -2.0],
        "source": source,
    }


def test_hotpotqa_train_clean_accepts_prepared_q0_s1_jsonl(tmp_path):
    path = tmp_path / "hotpot_candidate_train_q0_s1.jsonl"
    write_jsonl(path, [sample(), {**sample(), "_id": "hotpotqa-2"}])
    dataset = RetrievalHotPotQA(
        path=str(tmp_path),
        split="train-clean",
        length=1,
        seed=52,
    )
    assert dataset.name() == "hotpotqa"
    assert len(dataset) == 1
    assert dataset[0]["candidates"]


def test_twowiki_train_clean_accepts_prepared_q0_s1_jsonl(tmp_path):
    path = tmp_path / "2wiki_candidate_train_q0_s1.jsonl"
    write_jsonl(path, [sample("2WikiMultihopQA")])
    dataset = Retrieval2WikiMultihopQA(
        path=str(tmp_path),
        split="train-clean",
        skip_bridge_comparison=False,
        seed=100,
    )
    assert dataset.name() == "2WikiMultihopQA"
    assert len(dataset) == 1


def test_combined_candidate_preserves_source_and_question(tmp_path):
    path = tmp_path / "combined.jsonl"
    write_jsonl(path, [sample("2WikiMultihopQA")])
    dataset = CombinedCandidateDataset(path=str(path), seed=42)
    adapted = CandidateDatasetAdapter(dataset, min_chunks=1)[0]
    assert adapted["source"] == "2WikiMultihopQA"
    assert adapted["question"] == "Question stays verbatim?"
