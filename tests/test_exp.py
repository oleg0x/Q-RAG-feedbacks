from __future__ import annotations

import json

import pytest

import exp


# Символы, которые ``str.splitlines()`` считает переводом строки, а
# ``json.dumps`` оставляет в файле как есть. Их ровно три: остальные пять
# из списка splitlines (\x0b, \x0c, \x1c–\x1e) меньше 0x20 и потому
# экранируются — через них запись не развалится. Три вопроса TriviaQA
# из eval-таблицы Search-R1 содержат первый из них.
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
    """Одна запись обязана остаться одной строкой, что бы в ней ни лежало.

    Разрез по такому символу разваливает запись надвое: судья получает
    обломок и падает, а число записей тихо перестаёт сходиться со входом.
    """
    records = [
        {"id": "a", "question": "who?"},
        {"id": "b", "question": f"Vicky Christina {character}.?"},
        {"id": "c", "question": "what?"},
    ]
    path = write_jsonl(tmp_path, records)
    # Ровно то, что делал прежний код и что было ошибкой.
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
    """Поломанный вход должен назваться сразу, а не через четыре минуты.

    Без этой проверки развалившаяся запись проявляется как «три шарда из
    тридцати двух упали» — уже после того, как судья отработал на остальных.
    """
    path = tmp_path / "retrieval.jsonl"
    good = ['{"id": "a"}', '{"id": "b"}']
    exp.check_jsonl_shape(good, path)
    with pytest.raises(exp.StageFailed, match="не являются целыми JSON-объектами"):
        exp.check_jsonl_shape(['{"id": "a"}', '{"id": "b"'], path)
    with pytest.raises(exp.StageFailed, match="строки \\[1\\]"):
        exp.check_jsonl_shape(['"question": "хвост"}'], path)
