from __future__ import annotations

import json

import pytest

import exp


# Characters that ``str.splitlines()`` treats as line breaks but
# ``json.dumps`` leaves in the file unescaped. There are exactly three: the
# other five from the splitlines list (\x0b, \x0c, \x1c–\x1e) are below 0x20
# and therefore escaped, so they cannot split a record. Some TriviaQA
# questions in the Search-R1 eval set contain the first one.
NEL = "\x85"
LINE_SEPARATOR = " "
PARAGRAPH_SEPARATOR = " "


def write_jsonl(tmp_path, records):
    path = tmp_path / "retrieval.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


@pytest.mark.parametrize("character", [NEL, LINE_SEPARATOR, PARAGRAPH_SEPARATOR])
def test_records_are_not_split_on_characters_json_leaves_raw(tmp_path, character) -> None:
    """One record must stay one line, whatever it contains.

    Splitting on such a character breaks the record in two: the judge gets a
    fragment and fails, and the record count silently stops matching the input.
    """
    records = [
        {"id": "a", "question": "who?"},
        {"id": "b", "question": f"Vicky Christina {character}.?"},
        {"id": "c", "question": "what?"},
    ]
    path = write_jsonl(tmp_path, records)
    # Exactly what the old code did, and that was the bug.
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4
    lines = exp.jsonl_lines(path)
    assert len(lines) == 3
    assert [json.loads(line)["id"] for line in lines] == ["a", "b", "c"]
    assert json.loads(lines[1])["question"].endswith(f"{character}.?")


def test_trailing_and_blank_lines_are_dropped(tmp_path) -> None:
    path = tmp_path / "retrieval.jsonl"
    path.write_text('{"id": "a"}\n\n{"id": "b"}\n\n', encoding="utf-8")
    assert [json.loads(line)["id"] for line in exp.jsonl_lines(path)] == ["a", "b"]


def test_broken_lines_are_refused_before_thirty_two_processes_start(tmp_path) -> None:
    """A broken input must be reported at once, not minutes later.

    Without this check a split record shows up as a few failed shards out of
    32, after the judge has already run on the rest.
    """
    path = tmp_path / "retrieval.jsonl"
    good = ['{"id": "a"}', '{"id": "b"}']
    exp.check_jsonl_shape(good, path)
    with pytest.raises(exp.StageFailed, match="are not complete JSON objects"):
        exp.check_jsonl_shape(['{"id": "a"}', '{"id": "b"'], path)
    with pytest.raises(exp.StageFailed, match="lines \\[1\\]"):
        exp.check_jsonl_shape(['"question": "tail"}'], path)
