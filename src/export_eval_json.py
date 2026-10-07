#!/usr/bin/env python3
"""Human-readable slice of a run: question, gold answer, model answer, chunks.

``answer_judge.json`` has everything needed but is awkward to read: it carries
``supporting_facts``, ``sf_idx`` and per-candidate dumps of every hop, the
article title is hidden as the first line of the chunk text, and there are no
answer aliases at all, since ``answer_judge_llms.py`` compares the prediction
with the single ``answer``.

Here each record is split into fields, and alias versions of EM and F1 appear
next to the primary metrics: 2Wiki keeps aliases in ``id_aliases.json``,
addressed by the ``answer_id`` field, MuSiQue in the example's own
``answer_aliases``. The official evals of both datasets take EM as the maximum
over aliases, so without this column our numbers are systematically below the
published ones.

The primary metrics are not recomputed but **verified**: ``EM`` and ``F1``
computed here against the primary answer must match those in
``answer_judge.json``. A mismatch means the normalization has diverged from
the judge's, and then the alias versions cannot be trusted, so the script
fails.

Example:

    python src/export_eval_json.py --run 2026-08-05-lineA-best-2wiki
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import runlib
from fullwiki_qrag import load_input_samples, sample_id, wiki_title


LOG = logging.getLogger("export-eval-json")

EXPORT_FILE = "export.json"
ALIAS_FILE = "id_aliases.json"


# --------------------------------------------------------------------------
# metrics
#
# A copy of the normalization from the original ``answer_judge_llms.py``. A
# copy rather than an import: that module pulls in the vLLM client and prompts
# on import, and it must stay unmodified. Agreement is guaranteed not by
# eyeballing but by a cross-check on every record in :func:`export_record`.


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(char for char in value if char not in string.punctuation)
    return " ".join(value.split())


def exact_match(prediction: str, target: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(target))


def f1_score(prediction: str, target: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    target_tokens = normalize_answer(target).split()
    if not prediction_tokens or not target_tokens:
        return float(prediction_tokens == target_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_aliases(
    prediction: str, targets: Sequence[str]
) -> tuple[int, float]:
    """Maximum EM and F1 over the acceptable answers, as in the official evals."""
    em = max((exact_match(prediction, target) for target in targets), default=0)
    f1 = max((f1_score(prediction, target) for target in targets), default=0.0)
    return em, f1


# --------------------------------------------------------------------------
# aliases


def load_id_aliases(path: Path) -> dict[str, list[str]]:
    """2WikiMultiHopQA ``id_aliases.json``: one line per Wikidata entity."""
    aliases: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            aliases[str(record["Q_id"])] = [
                str(value) for value in record.get("aliases", [])
            ]
    return aliases


def collect_aliases(dataset: Path, alias_source: Path | None) -> dict[str, list[str]]:
    """Map example id to acceptable answers besides the primary one.

    The format depends on the dataset: MuSiQue carries aliases in the example
    itself, 2Wiki references an external entity table via ``answer_id``, and
    HotpotQA has none. All three cases are handled here so that callers need
    not know which dataset they export.
    """
    samples = load_input_samples(dataset)
    inline = {
        sample_id(sample, index): [
            str(value) for value in sample.get("answer_aliases") or []
        ]
        for index, sample in enumerate(samples)
        if sample.get("answer_aliases")
    }
    if inline:
        LOG.info("Aliases from the dataset itself: %d examples", len(inline))
        return inline

    if alias_source is None:
        candidate = dataset.parent / ALIAS_FILE
        alias_source = candidate if candidate.is_file() else None
    if alias_source is None:
        LOG.info("The dataset has no aliases: alias metrics will equal the primary ones")
        return {}

    table = load_id_aliases(alias_source)
    LOG.info("Alias table: %d entities, %s", len(table), alias_source)
    by_sample = {}
    for index, sample in enumerate(samples):
        answer_id = sample.get("answer_id")
        if not answer_id:
            continue
        found = table.get(str(answer_id))
        if found:
            by_sample[sample_id(sample, index)] = found
    LOG.info("Aliases resolved for %d examples", len(by_sample))
    return by_sample


# --------------------------------------------------------------------------
# export


def split_chunk(text: str) -> tuple[str, str]:
    """A corpus chunk is stored as ``"Title"\\ntext``; split it back."""
    _, _, body = text.partition("\n")
    return wiki_title(text), body


def export_record(
    record: dict[str, Any],
    aliases: Sequence[str],
    index: int,
) -> dict[str, Any]:
    prediction = record.get("prediction") or ""
    answer = record.get("answer") or ""

    # Cross-check against the judge's numbers: this is what guarantees that the
    # alias versions use the same normalization as the EM column of the table.
    if exact_match(prediction, answer) != int(record["EM"]):
        raise RuntimeError(
            f"Record {record.get('id', index)!r}: EM disagrees with "
            f"answer_judge.json ({exact_match(prediction, answer)} vs "
            f"{record['EM']}); answer normalization differs from the judge's"
        )
    if abs(f1_score(prediction, answer) - float(record["F1"])) > 1e-9:
        raise RuntimeError(
            f"Record {record.get('id', index)!r}: F1 disagrees with answer_judge.json"
        )

    targets = [answer, *aliases]
    em_alias, f1_alias = best_over_aliases(prediction, targets)

    hops = record.get("retrieval_hops") or []
    chunks = []
    for position, text in enumerate(record.get("pred_texts", [])):
        title, body = split_chunk(text)
        hop = hops[position] if position < len(hops) else {}
        chunks.append(
            {
                "step": hop.get("step", position),
                "title": title,
                "text": body,
                "score": hop.get("selected_score"),
                "row": (record.get("pred_idx") or [None] * (position + 1))[position],
            }
        )

    return {
        "id": record.get("id", index),
        "question": record.get("question"),
        "gold_answer": answer,
        "gold_aliases": list(aliases),
        "prediction": prediction,
        "gold_titles": record.get("gold_titles", []),
        "chunks": chunks,
        "metrics": {
            "em": int(record["EM"]),
            "f1": float(record["F1"]),
            "judge": float(record["LLM_Judge_Score"]),
            "em_alias": em_alias,
            "f1_alias": f1_alias,
            "title_em": record.get("title_em"),
            "title_recall": record.get("title_recall"),
        },
    }


def summarize(exported: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [item["metrics"][key] for item in exported]
        return sum(values) / len(values) if values else 0.0

    return {
        "examples": len(exported),
        "em": mean("em"),
        "f1": mean("f1"),
        "judge": mean("judge"),
        "em_alias": mean("em_alias"),
        "f1_alias": mean("f1_alias"),
        "with_aliases": sum(1 for item in exported if item["gold_aliases"]),
    }


def build_export(
    run_id: str,
    manifest: dict[str, Any],
    records: Iterable[dict[str, Any]],
    aliases: dict[str, list[str]],
) -> dict[str, Any]:
    config = manifest.get("config", {})
    retrieve = config.get("retrieve", {})
    exported = [
        export_record(record, aliases.get(str(record.get("id")), []), index)
        for index, record in enumerate(records)
    ]
    return {
        "run_id": run_id,
        "dataset": runlib.dataset_key(config),
        "label": config.get("label"),
        "input": retrieve.get("input"),
        "checkpoint": retrieve.get("checkpoint"),
        "steps": retrieve.get("steps"),
        "max_chunks_per_title": retrieve.get("max_chunks_per_title"),
        "reader_model": (config.get("judge") or {}).get("answer_model"),
        "summary": summarize(exported),
        "examples": exported,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run_id from the registry")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="questions file; defaults to retrieve.input from the manifest",
    )
    parser.add_argument(
        "--alias-source",
        type=Path,
        default=None,
        help=f"alias table; defaults to {ALIAS_FILE} next to the dataset",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run = runlib.run_dir(args.run)
    manifest = runlib.read_manifest(run)
    judge_file = run / runlib.JUDGE_FILE
    if not judge_file.exists():
        raise SystemExit(f"Missing {judge_file}: nothing to export")

    dataset = args.dataset or Path(manifest["config"]["retrieve"]["input"])
    aliases = collect_aliases(dataset.expanduser().resolve(), args.alias_source)

    with judge_file.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    export = build_export(args.run, manifest, records, aliases)

    destination = args.output or run / EXPORT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(destination)
    summary = export["summary"]
    LOG.info(
        "%s: %d examples, EM=%.4f (alias %.4f), F1=%.4f, judge=%.4f → %s",
        export["dataset"],
        summary["examples"],
        summary["em"],
        summary["em_alias"],
        summary["f1"],
        summary["judge"],
        destination,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
