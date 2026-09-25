#!/usr/bin/env python3
"""Build the Phase-0 comparison table from reader/judge output files.

Every row is recomputed from the files rather than copied from a previous
report. Title metrics are recomputed here too, from the ``gold_titles`` and
``retrieved_titles`` that ``answer_judge_llms.py`` carries over from the
retrieval JSONL, so runs produced before title normalization existed are
scored by the same definition as new ones.

The ``share of ceiling`` column divides ``title_em`` by the measured corpus
coverage ceiling (see ``corpus_title_coverage.py``): HotpotQA gold titles do
not all exist in Wiki-18, so a raw ``title_em`` understates how much of the
reachable retrieval a run actually got.

Example:

    python src/report_phase0.py \
      --coverage runs/hotpotqa_dev_title_coverage.json \
      --run "no retrieval=runs/fullwiki_no_retrieval_answer_judge_mt1000.json" \
      --run "GTE top-2=runs/fullwiki_gte_only_steps2_answer_judge_mt1000.json"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from export_eval_json import best_over_aliases
from fullwiki_qrag import read_json, title_metrics


LOG = logging.getLogger("report-phase0")


def parse_run(spec: str) -> tuple[str, Path]:
    # Разрез по последнему «=», а не по первому: label — человеческий текст
    # и может содержать «=» («…, N=2»), а путь рана — никогда.
    label, separator, path = spec.rpartition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError(f"Expected 'label=path', got {spec!r}")
    return label.strip(), Path(path.strip()).expanduser().resolve()


def score_run(
    path: Path,
    ceiling: float | None,
    aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Метрики одного рана из его reader/judge JSON.

    Два поля необязательны, и оба — по одной причине: не у каждого датасета
    есть то, из чего они считаются.

    ``ceiling`` отсутствует, когда у датасета нет gold-титулов. Тогда
    title-метрики и доля потолка не занижаются и не обнуляются, а становятся
    ``None``: ``title_metrics`` на пустом множестве gold-титулов возвращает
    1.0, и молчаливая единица здесь была бы хуже отсутствующей метрики.

    ``aliases`` задан, когда официальный эвал датасета считает EM максимумом
    по списку допустимых ответов, а ``answer_judge_llms.py`` (read-only)
    сравнивает с единственным ``answer``. Тогда рядом с ``em``/``f1``
    появляются ``em_alias``/``f1_alias``, посчитанные тем же кодом, что и в
    ``export_eval_json.py``.
    """
    # read_json insists on a JSON object; reader/judge output is an array.
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array: {path}")

    # Различие «поля нет» и «поле пустое» здесь существенно: пустой список
    # gold-титулов — законное значение, а отсутствие ключа означает, что
    # титулов у датасета нет вовсе и метрику считать не по чему.
    has_titles = any("gold_titles" in record for record in records)

    rows = []
    for index, record in enumerate(records):
        row = {
            "em": float(record["EM"]),
            "f1": float(record["F1"]),
            "judge": float(record["LLM_Judge_Score"]),
            "chunks": len(record.get("pred_texts", [])),
            "empty": (record.get("prediction") or "") == "",
            # Контракт v2 зовёт судью не на каждом примере, а вменяет единицу
            # там, где сработал EM. Доля вменённых вердиктов отличает такой
            # ран от старого, где `judge` — вердикт на каждом вопросе.
            "judge_imputed": bool(record.get("judge_imputed")),
        }
        if has_titles:
            row.update(
                title_metrics(
                    record.get("gold_titles", []), record.get("retrieved_titles", [])
                )
            )
        if aliases is not None:
            prediction = record.get("prediction") or ""
            answer = str(record.get("answer") or "")
            # Сверка перед использованием, как в export_eval_json.py: ``em``
            # приходит из судейского файла, а ``em_alias`` считается здесь.
            # Если нормализации разойдутся, alias-версия молча окажется ниже
            # основной — то есть главное число таблицы станет неверным, и
            # заметить это будет нечем. Дешевле упасть.
            em_self, f1_self = best_over_aliases(prediction, [answer])
            if em_self != float(record["EM"]) or abs(f1_self - float(record["F1"])) > 1e-9:
                raise ValueError(
                    f"{path.name}, запись {index} ({record.get('id')!r}): "
                    f"пересчёт по основному ответу даёт EM={em_self} F1={f1_self:.6f}, "
                    f"а в файле судьи EM={record['EM']} F1={record['F1']}. "
                    "Нормализация разъехалась — alias-метрикам верить нельзя."
                )
            targets = [answer, *aliases.get(str(record.get("id", index)), [])]
            em_alias, f1_alias = best_over_aliases(prediction, targets)
            row["em_alias"] = float(em_alias)
            row["f1_alias"] = f1_alias
            row["has_aliases"] = bool(aliases.get(str(record.get("id", index))))
        rows.append(row)

    def mean(key: str, subset: Sequence[dict[str, Any]] | None = None) -> float:
        source = rows if subset is None else subset
        return float(np.mean([row[key] for row in source])) if source else float("nan")

    scored: dict[str, Any] = {
        "em_per_sample": [row["em"] for row in rows],
        # F1 и judge тоже покандидатно: EM бинарен, и на эффектах меньше
        # процентного пункта McNemar по нему упирается в число дискордантных
        # пар, тогда как парный t по непрерывной величине на тех же вопросах
        # различает их уверенно (ablation квоты 2026-08-08: EM z = +2.4 против
        # t(F1) = +3.44 на одних и тех же ранах).
        "f1_per_sample": [row["f1"] for row in rows],
        "judge_per_sample": [row["judge"] for row in rows],
        "samples": len(rows),
        "empty_predictions": sum(row["empty"] for row in rows),
        "chunks_min": min(row["chunks"] for row in rows),
        "chunks_max": max(row["chunks"] for row in rows),
        "em": mean("em"),
        "f1": mean("f1"),
        "judge": mean("judge"),
        "judge_imputed_share": mean("judge_imputed"),
    }

    if has_titles:
        hit = [row for row in rows if row["title_em"] == 1.0]
        miss = [row for row in rows if row["title_em"] != 1.0]
        title_em = mean("title_em")
        scored.update(
            {
                "title_recall": mean("title_recall"),
                "title_em": title_em,
                "title_em_exact": mean("title_em_exact"),
                "share_of_ceiling": title_em / ceiling if ceiling else float("nan"),
                "em_given_hit": mean("em", hit),
                "em_given_miss": mean("em", miss),
                "hit_share": len(hit) / len(rows),
            }
        )
    else:
        scored.update(
            {
                "title_recall": None,
                "title_em": None,
                "title_em_exact": None,
                "share_of_ceiling": None,
                "em_given_hit": None,
                "em_given_miss": None,
                "hit_share": None,
            }
        )

    if aliases is not None:
        scored["em_alias"] = mean("em_alias")
        scored["f1_alias"] = mean("f1_alias")
        scored["em_alias_per_sample"] = [row["em_alias"] for row in rows]
        scored["with_aliases"] = sum(row["has_aliases"] for row in rows)
    return scored


def percent(value: float | None) -> str:
    if value is None:
        return "—"
    return "n/a" if np.isnan(value) else f"{value * 100:.2f}"


def mcnemar(baseline: Sequence[float], variant: Sequence[float]) -> dict[str, Any]:
    """Paired exact-match comparison on the same questions.

    EM is binary per question and both runs answer the identical set, so the
    discordant pairs carry all the information. Under the null the number of
    wins is Binomial(discordant, 0.5); the normal approximation with a
    continuity correction is ample at these counts.
    """
    if len(baseline) != len(variant):
        raise ValueError("Paired comparison needs equal-length runs")
    wins = sum(1 for b, v in zip(baseline, variant) if v > b)
    losses = sum(1 for b, v in zip(baseline, variant) if v < b)
    discordant = wins + losses
    if discordant == 0:
        return {"wins": 0, "losses": 0, "z": float("nan")}
    z = (abs(wins - losses) - 1) / np.sqrt(discordant)
    return {
        "wins": wins,
        "losses": losses,
        "discordant": discordant,
        "z": float(z if wins >= losses else -z),
    }


def paired_t(baseline: Sequence[float], variant: Sequence[float]) -> float:
    """Парный t по непрерывной метрике на тех же вопросах.

    Нужен рядом с McNemar, а не вместо него: EM бинарен, поэтому весь его
    сигнал сидит в дискордантных парах, и эффект в половину пункта на 7 405
    вопросах он не разрешает. F1 и judge меняются на тех же вопросах
    непрерывно, и та же разница выходит значимой (см. docstring score_run).
    """
    if len(baseline) != len(variant):
        raise ValueError("Paired comparison needs equal-length runs")
    diff = np.asarray(variant, dtype=float) - np.asarray(baseline, dtype=float)
    if len(diff) < 2:
        return float("nan")
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    # Нулевой разброс разностей — два разных случая, и путать их нельзя.
    # Совпавшие построчно раны: разницы нет вовсе, t не определён. Постоянный
    # сдвиг на всех вопросах без исключения: разница есть и она идеально
    # согласована, то есть значимость бесконечна, а не отсутствует. На живых
    # данных не встречается, но молча выдать «нет эффекта» на «эффект везде»
    # — худшее, что может сделать эта функция.
    if sd == 0.0:
        return float("nan") if mean == 0.0 else float("inf") * (1 if mean > 0 else -1)
    return float(mean / (sd / np.sqrt(len(diff))))


def render(scored: Sequence[tuple[str, dict[str, Any]]], ceiling: float | None) -> str:
    """Таблица сравнения.

    Форма зависит от того, что вообще посчиталось. Колонки title-метрик
    остаются на месте с прочерками — так строки датасета без gold-титулов
    видно как таковые, а не как ран, у которого ретривал промахнулся. Колонка
    EM по алиасам появляется только там, где алиасы есть, и тогда жирным
    выделена она: это определение EM официальных эвалов, и именно его надо
    читать первым.
    """
    has_alias = any("em_alias" in row for _, row in scored)
    ceiling_note = "доля потолка" + (f" ({ceiling * 100:.1f}%)" if ceiling else "")
    columns = [
        "Конфигурация", "Чанков", "title recall", "title EM", ceiling_note,
    ]
    columns += ["EM по алиасам"] if has_alias else []
    columns += ["EM", "F1", "LLM Judge", "EM \\| попали", "EM \\| не попали"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for label, row in scored:
        chunks = (
            str(row["chunks_min"])
            if row["chunks_min"] == row["chunks_max"]
            else f"{row['chunks_min']}–{row['chunks_max']}"
        )
        # The oracle context comes from HotpotQA, not from Wiki-18, so the
        # corpus coverage ceiling simply does not apply to it.
        share = row["share_of_ceiling"]
        share = "—" if share is not None and share > 1.0 else percent(share)
        cells = [
            label, chunks, percent(row["title_recall"]),
            percent(row["title_em"]), share,
        ]
        if has_alias:
            cells.append(f"**{percent(row.get('em_alias'))}**")
        cells += [
            percent(row["em"]) if has_alias else f"**{percent(row['em'])}**",
            percent(row["f1"]),
            percent(row["judge"]),
            percent(row["em_given_hit"]),
            percent(row["em_given_miss"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="reader/judge JSON to score; repeatable, order is preserved",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=None,
        help="corpus_title_coverage.py output for the evaluated dataset; "
             "опускается у датасетов без gold-титулов",
    )
    parser.add_argument(
        "--aliases",
        type=Path,
        default=None,
        help="файл датасета с answer_aliases: добавляет alias-версии EM и F1, "
             "как их считают официальные эвалы",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--pair",
        action="append",
        nargs=2,
        default=[],
        metavar=("BASELINE", "VARIANT"),
        # Два отдельных аргумента, а не «BASELINE=VARIANT»: обе стороны здесь
        # человеческие метки, и обе могут содержать «=» («…, N=2»). У --run
        # выручает rpartition, потому что путь знака равенства не содержит, а
        # тут разрезать нечем и разбор молча даёт несуществующую метку.
        help="paired McNemar comparison on EM between two --run labels",
    )
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
    coverage = None
    ceiling = None
    if args.coverage is not None:
        coverage = read_json(args.coverage.expanduser().resolve())
        ceiling = float(coverage["title_em_ceiling_normalized"])
    aliases = None
    if args.aliases is not None:
        from export_eval_json import collect_aliases

        aliases = collect_aliases(args.aliases.expanduser().resolve(), None)
    scored = []
    for spec in args.run:
        label, path = parse_run(spec)
        LOG.info("Scoring %s", path)
        scored.append((label, score_run(path, ceiling, aliases)))

    for label, row in scored:
        if row["empty_predictions"]:
            LOG.warning(
                "%s has %d empty predictions", label, row["empty_predictions"]
            )
        if row["samples"] != scored[0][1]["samples"]:
            LOG.warning("%s has %d samples, expected %d", label, row["samples"],
                        scored[0][1]["samples"])

    table = render(scored, ceiling)
    print(table)

    by_label = dict(scored)
    pairs = {}
    # Парный тест идёт по той же величине, что вынесена в таблицу жирным: там,
    # где EM определён максимумом по алиасам, сравнивать одноответный EM
    # значило бы проверять не то число, которое потом читают.
    vector = "em_alias_per_sample" if aliases is not None else "em_per_sample"
    if args.pair:
        print()
        metric = "EM по алиасам" if aliases is not None else "EM"
        print(
            f"| Сравнение (парный McNemar по {metric}) | побед | поражений | z "
            "| t(F1) | t(judge) |"
        )
        print("|---|---:|---:|---:|---:|---:|")
    for baseline_label, variant_label in args.pair:
        baseline_label, variant_label = baseline_label.strip(), variant_label.strip()
        for label in (baseline_label, variant_label):
            if label not in by_label:
                raise ValueError(f"--pair references unknown run {label!r}")
        base_row, variant_row = by_label[baseline_label], by_label[variant_label]
        result = mcnemar(base_row[vector], variant_row[vector])
        for key, name in (("f1_per_sample", "t_f1"), ("judge_per_sample", "t_judge")):
            result[name] = paired_t(base_row[key], variant_row[key])
        pairs[f"{baseline_label} → {variant_label}"] = result
        print(
            f"| {variant_label} против {baseline_label} | {result['wins']} | "
            f"{result['losses']} | {result['z']:+.1f} "
            f"| {result['t_f1']:+.2f} | {result['t_judge']:+.2f} |"
        )

    summary = {
        label: {
            key: value for key, value in row.items() if not key.endswith("_per_sample")
        }
        for label, row in scored
    }
    print()
    print(json.dumps(summary, indent=2))
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "ceiling_normalized": ceiling,
                    "ceiling_exact": (
                        coverage["title_em_ceiling_exact"] if coverage else None
                    ),
                    "runs": {label: row for label, row in scored},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        LOG.info("Wrote %s", destination)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Failed: %s", error)
        raise
