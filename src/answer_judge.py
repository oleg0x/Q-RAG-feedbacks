#!/usr/bin/env python3
"""Ридер и судья по retriever JSONL — форк с контрактом награды v2.

Функциональная копия read-only ``Q-RAG-feedback/answer_judge_llms.py``: те же
аргументы, тот же клиент, те же промпты, то же декодирование. Отличие одно и
задаётся флагом ``--contract``.

``v1`` — прежнее поведение, бит в бит: ``EM``/``F1`` считаются против
единственного ``answer``, судья зовётся на каждом примере. Этим контрактом
сняты все 65 старых ранов, и пересуживать их не нужно.

``v2`` (умолчание) — награда по вариантам ответа::

    em_alias = max(exact_match(prediction, v) для v в вариантах)
    если em_alias == 0:  судья, эталон = " | ".join(варианты)
    reward   = 1.0 если em_alias иначе float(вердикт == "CORRECT")

Зачем: у NQ 42.5% примеров имеют больше одного допустимого ответа (в среднем
1.80, максимум 23), и сравнение с единственным занижает EM с 11.97% до 9.25%
на одних и тех же предсказаниях. Судья же расходится с ``em_alias`` в обе
стороны, поэтому награда — максимум, а не вердикт судьи.

Варианты ответа берутся **не из ``retrieval.jsonl``**: стадия ретривала
алиасы роняет. Таблица грузится отдельно, ровно как в ``export_eval_json.py``
и ``report_phase0.py`` — по ``--dataset <ключ>`` через ``runlib.DATASETS`` или
по явному ``--aliases <путь>``. У датасета без алиасов список вариантов —
из одного ``answer``.

    python src/answer_judge.py --retriever-logfile runs/<id>/retrieval.jsonl \\
        --output-file runs/<id>/answer_judge.json --dataset sr1_nq
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
from collections import Counter
from pathlib import Path

# Клиент и промпты — вендорные байт-копии read-only оригинала
# (`src/vendored_prompts.py`, `src/vendored_vllm_client.py`), а не свой код:
# иначе регрессия против оригинала проверяла бы наш клиент, а не нашу логику.
# Копии внутри репозитория, чтобы форк работал без соседнего Q-RAG-feedback;
# их дрейф ловит tests/test_vendored.py там, где оригинал доступен.
import vendored_prompts as prompts
from vendored_vllm_client import SyncVllmClient, extract_response_text


CONTRACTS = ("v1", "v2")

# Одна правка промпта судьи: эталон стал списком. Строгость не трогаем —
# подстановка, а не свой текст, чтобы расхождение с оригиналом падало, а не
# копилось тихо. Копия определения из
# Q-RAG_for_full-wiki/prompts_and_metrics/prompts.py, равенство проверяет
# test_answer_judge.py.
JUDGE_REFERENCE_V1 = (
    "You are given a QUESTION, PREDICTED ANSWER and GROUNDTRUTH ANSWER."
)
JUDGE_REFERENCE_V2 = (
    "You are given a QUESTION, a PREDICTED ANSWER and a GROUNDTRUTH ANSWER "
    'field holding one or more accepted answers separated by " | "; the '
    "prediction is correct if it matches any one of them."
)


def judge_prompt(contract: str) -> str:
    if contract == "v1":
        return prompts.sys_judge
    if JUDGE_REFERENCE_V1 not in prompts.sys_judge:
        raise SystemExit(
            "Промпт судьи в vendored_prompts.py изменился: строку описания "
            "эталона не нашли, подстановка множественного числа больше не "
            "применима"
        )
    return prompts.sys_judge.replace(JUDGE_REFERENCE_V1, JUDGE_REFERENCE_V2)


# --------------------------------------------------------------------------
# метрики — копия оригинала, дословно


def normalize_answer(value):
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(char for char in value if char not in string.punctuation)
    return " ".join(value.split())


def exact_match(prediction, target):
    return int(normalize_answer(prediction) == normalize_answer(target))


def f1_score(prediction, target):
    prediction_tokens = normalize_answer(prediction).split()
    target_tokens = normalize_answer(target).split()
    if not prediction_tokens or not target_tokens:
        return float(prediction_tokens == target_tokens)
    overlap = sum(
        (Counter(prediction_tokens) & Counter(target_tokens)).values()
    )
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def final_answer(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    marker = "final answer:"
    if marker in text.lower():
        text = text[text.lower().rfind(marker) + len(marker) :]
    return text.strip()


def best_over_aliases(prediction, targets):
    """Максимум EM и F1 по вариантам — как в официальных эвалах."""
    em = max((exact_match(prediction, target) for target in targets), default=0)
    f1 = max((f1_score(prediction, target) for target in targets), default=0.0)
    return em, f1


# --------------------------------------------------------------------------
# варианты ответа


def load_variants(dataset: str | None, aliases_path: str | None) -> dict:
    """``id`` → допустимые ответы помимо основного.

    Пустой словарь означает «алиасов у датасета нет», и список вариантов
    каждого примера сведётся к его единственному ``answer``.
    """
    if dataset is None and aliases_path is None:
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import runlib
    from export_eval_json import collect_aliases

    if aliases_path is not None:
        return collect_aliases(Path(aliases_path).expanduser().resolve(), None)
    table = runlib.alias_table(dataset)
    return table or {}


def variants_of(sample, aliases: dict, index: int) -> list:
    answer = str(sample.get("answer") or "")
    extra = aliases.get(str(sample.get("id", index)), [])
    return [answer, *extra]


# --------------------------------------------------------------------------


def build_reader_request(sample) -> str:
    context = "\n\n".join(sample["pred_texts"])
    return (
        f"CONTEXT:\n{context}\n\nQUESTION:\n{sample['question']}\n\n"
        "Final Answer:"
    )


def build_judge_request(question, prediction, reference) -> str:
    return (
        f"QUESTION: {question}\n"
        f"PREDICTED ANSWER: {prediction}\n"
        f"GROUNDTRUTH ANSWER: {reference}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-logfile", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--answer-model", default=os.getenv("VLLM_MODEL"))
    parser.add_argument("--judge-model")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--judge-max-tokens", type=int, default=100)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--contract", choices=CONTRACTS, default="v2")
    parser.add_argument(
        "--judge-all",
        action="store_true",
        help="звать судью на каждом примере независимо от em_alias",
    )
    parser.add_argument("--dataset", help="ключ runlib.DATASETS для таблицы вариантов")
    parser.add_argument("--aliases", help="файл таблицы вариантов напрямую")
    args = parser.parse_args()
    if not args.base_url or not args.answer_model:
        parser.error("VLLM_BASE_URL/--base-url и VLLM_MODEL/--answer-model обязательны")
    if args.dataset and args.aliases:
        parser.error("--dataset и --aliases взаимоисключающие")

    aliases = load_variants(args.dataset, args.aliases) if args.contract == "v2" else {}

    with open(args.retriever_logfile, encoding="utf-8") as source:
        samples = [json.loads(line) for line in source if line.strip()]
    samples = samples[: args.max_samples]

    results = []
    with SyncVllmClient(
        llm=args.answer_model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=prompts.sys_qa,
        thinking=False,
        max_tokens=args.max_tokens,
        temperature=0.0,
    ) as reader:
        for index, sample in enumerate(samples):
            prediction = final_answer(
                extract_response_text(reader.chat_completion(build_reader_request(sample)))
            )
            record = {
                **sample,
                "prediction": prediction,
                "EM": exact_match(prediction, sample["answer"]),
                "F1": f1_score(prediction, sample["answer"]),
            }
            if args.contract == "v2":
                variants = variants_of(sample, aliases, index)
                em_alias, f1_alias = best_over_aliases(prediction, variants)
                record["answer_variants"] = variants
                record["em_alias"] = em_alias
                record["f1_alias"] = f1_alias
            results.append(record)

    if not args.skip_judge:
        with SyncVllmClient(
            llm=args.judge_model or args.answer_model,
            base_url=args.base_url,
            api_key=args.api_key,
            system_prompt=judge_prompt(args.contract),
            thinking=False,
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
        ) as judge:
            for result in results:
                if args.contract == "v1":
                    reference = result["answer"]
                    ask = True
                else:
                    reference = " | ".join(result["answer_variants"])
                    ask = args.judge_all or not result["em_alias"]
                if ask:
                    judgment = final_answer(
                        extract_response_text(
                            judge.chat_completion(
                                build_judge_request(
                                    result["question"],
                                    result["prediction"],
                                    reference,
                                )
                            )
                        )
                    ).upper()
                    result["LLM_Judge_Score"] = float(judgment == "CORRECT")
                    if args.contract == "v2":
                        result["judge_imputed"] = False
                else:
                    # Судью не звали: EM уже совпал. Вменённая единица помечается
                    # отдельным полем — вменение нельзя путать с вердиктом.
                    result["LLM_Judge_Score"] = 1.0
                    result["judge_imputed"] = True
                if args.contract == "v2":
                    result["reward"] = float(
                        result["em_alias"] or result["LLM_Judge_Score"]
                    )
                    result["contract"] = "v2"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
