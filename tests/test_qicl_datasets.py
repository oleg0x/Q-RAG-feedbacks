import json

import pytest

from envs import (
    QADatasetAdapter,
    RetrievalGSM8K,
    RetrievalHellaSwag,
    RetrievalMATH,
    RetrievalMMLUPro,
    RetrievalXLSum,
)


def write_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as destination:
        for sample in samples:
            destination.write(json.dumps(sample) + "\n")


def dataset_cases():
    return [
        (
            RetrievalGSM8K,
            "gsm8k",
            "train.jsonl",
            "test.jsonl",
            {
                "question": "What is 1 + 1?",
                "answer": "Add the numbers.\n#### 2",
            },
        ),
        (
            RetrievalMATH,
            "MATH",
            "train_ans.jsonl",
            "test_ans.jsonl",
            {
                "id": 1,
                "problem": "Compute $1+1$.",
                "solution": "We get \\boxed{2}.",
                "answer": "2",
            },
        ),
        (
            RetrievalHellaSwag,
            "HellaSwag",
            "hellaswag_train.jsonl",
            "hellaswag_val.jsonl",
            {
                "ind": 1,
                "ctx": "A runner reaches the finish line.",
                "endings": ["They stop.", "They start asleep.", "They vanish.", "They swim."],
                "label": 0,
            },
        ),
        (
            RetrievalMMLUPro,
            "MMLU-Pro",
            "train.jsonl",
            "test.jsonl",
            {
                "question_id": 1,
                "question": "Which letter is first?",
                "options": list("ABCDEFGHIJ"),
                "answer": "A",
            },
        ),
        (
            RetrievalXLSum,
            "XL-Sum",
            "english_train.jsonl",
            "english_val.jsonl",
            {
                "id": "story-1",
                "text": "A concise news article.",
                "summary": "A concise summary.",
            },
        ),
    ]


@pytest.mark.parametrize(
    "loader_cls,expected_name,train_file,eval_file,sample",
    dataset_cases(),
)
def test_qicl_loader_and_adapter(
    tmp_path,
    loader_cls,
    expected_name,
    train_file,
    eval_file,
    sample,
):
    second_sample = dict(sample)
    for key in ("id", "ind", "question_id"):
        if key in second_sample:
            second_sample[key] = (
                second_sample[key] + 1
                if isinstance(second_sample[key], int)
                else second_sample[key] + "-2"
            )
    write_jsonl(tmp_path / train_file, [sample, second_sample])
    write_jsonl(tmp_path / eval_file, [sample])

    train_dataset = loader_cls(
        path=str(tmp_path),
        split="train",
        samples_num=1,
        examples_num=1,
    )
    assert train_dataset.name() == expected_name
    assert len(train_dataset) == 1
    assert train_dataset[0]
    assert len(train_dataset.get_examples()) == 1

    eval_dataset = loader_cls(
        path=str(tmp_path),
        split="test",
        samples_num=1,
        examples_num=1,
    )
    adapted = QADatasetAdapter(eval_dataset)[0]
    assert adapted["id"] is not None
    assert adapted["question"]
    assert adapted["answer"] is not None
    assert adapted["chunks"] == eval_dataset.get_examples()
    assert adapted["sf_idx"] == [0]
    assert adapted["source"] == expected_name
