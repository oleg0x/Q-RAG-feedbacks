from __future__ import annotations

import pytest

import build_musique_eval as musique


def musique_sample(**overrides) -> dict:
    sample = {
        "id": "2hop__1_2",
        "question": "who?",
        "answer": "someone",
        "answer_aliases": ["somebody"],
        "paragraphs": [
            {"idx": 0, "title": "Alpha", "paragraph_text": "alpha one", "is_supporting": True},
            {"idx": 1, "title": "Beta", "paragraph_text": "beta one", "is_supporting": False},
            {"idx": 2, "title": "Alpha", "paragraph_text": "alpha two", "is_supporting": True},
        ],
    }
    sample.update(overrides)
    return sample


def test_paragraphs_of_one_article_become_one_context_entry() -> None:
    result = musique.convert_sample(musique_sample(), 0)
    assert result["context"] == [
        ["Alpha", ["alpha one", "alpha two"]],
        ["Beta", ["beta one"]],
    ]
    # Оба опорных абзаца принадлежат одной статье, поэтому различный
    # gold-титул здесь ровно один — именно так его посчитает title_em.
    assert result["supporting_facts"] == [["Alpha", 0], ["Alpha", 1]]
    assert len({fact[0] for fact in result["supporting_facts"]}) == 1


def test_converted_sample_yields_the_original_supporting_texts() -> None:
    sample = musique_sample()
    converted = musique.convert_sample(sample, 0)
    # Та же проверка, что и в конвертере: читаем результат тем кодом,
    # которым его будет читать пайплайн.
    musique.verify_sample(sample, converted)


def test_verify_catches_a_broken_conversion() -> None:
    sample = musique_sample()
    converted = musique.convert_sample(sample, 0)
    converted["supporting_facts"] = [["Alpha", 0]]
    with pytest.raises(RuntimeError, match="gold-предложения"):
        musique.verify_sample(sample, converted)


def test_aliases_and_hop_type_are_carried_over() -> None:
    result = musique.convert_sample(musique_sample(id="4hop3__9_8"), 0)
    assert result["answer_aliases"] == ["somebody"]
    assert result["type"] == "4hop3"


def test_missing_answer_is_a_hard_error() -> None:
    with pytest.raises(ValueError, match="пустой ответ"):
        musique.convert_sample(musique_sample(answer=""), 0)


def test_sample_without_supporting_paragraphs_is_rejected() -> None:
    paragraphs = [
        {"idx": 0, "title": "Alpha", "paragraph_text": "alpha", "is_supporting": False}
    ]
    with pytest.raises(ValueError, match="нет опорных абзацев"):
        musique.convert_sample(musique_sample(paragraphs=paragraphs), 0)
