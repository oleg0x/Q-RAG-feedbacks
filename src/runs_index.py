#!/usr/bin/env python3
"""Сборка runs/INDEX.md и RESULTS.md из манифестов ранов.

INDEX.md — реестр: одна строка на ран, чтобы ответ на «что уже прогонялось»
занимал один короткий файл, а не обход каталогов. RESULTS.md — витрина:
таблица сравнения строится тем же ``report_phase0.py``, что и раньше, но
список ранов берётся из манифестов, а не набирается флагами руками.

    python src/runs_index.py                  пересобрать оба файла
    python src/runs_index.py --only-index     без RESULTS.md (не нужен vLLM-раздел)

В сравнение попадают раны со ``"status": "ok"`` и непустыми метриками,
разложенные по датасетам: у каждого своя таблица со своим потолком покрытия
корпуса, потому что 7 405 вопросов HotpotQA и 12 576 вопросов 2Wiki — это
разные задачи, а не разные строки одной. Внутри датасета остаются раны с
модальным числом примеров: смешивать smoke-ран на 20 вопросах и полный
нельзя.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence

import runlib


LOG = logging.getLogger("runs-index")

RESULTS_FILE = runlib.REPO / "RESULTS.md"
TABLE_START = "<!-- BEGIN GENERATED TABLE -->"
TABLE_END = "<!-- END GENERATED TABLE -->"

# Пары для парного теста МакНемара: сравнивать Q-RAG нужно с first-stage на
# том же бюджете чанков, иначе таблица ничего не утверждает про вклад Q-RAG.
DEFAULT_PAIRS = (
    ("GTE top-2", "Q-RAG, 2 шага"),
    ("GTE top-4", "Q-RAG, 4 шага"),
    ("GTE top-6", "Q-RAG, 6 шагов"),
    ("no retrieval", "GTE top-6"),
    # Потолок реранкинга над тем же пулом: разрыв с GTE на том же бюджете
    # чанков и есть то, что может выиграть обучение реранкера.
    ("GTE top-2", "oracle по титулам, 2 чанка"),
    ("GTE top-4", "oracle по титулам, 4 чанка"),
    ("GTE top-6", "oracle по титулам, 6 чанков"),
    # Цена выбора чанка по рангу: тот же oracle над тем же пулом, но чанк
    # gold-титула выбирается по содержимому. Разница — то, что оракул по
    # титулам не мог достать, а чанковый реранкер может.
    ("oracle по титулам, 2 чанка", "oracle по чанкам, 2 чанка"),
    ("oracle по титулам, 4 чанка", "oracle по чанкам, 4 чанка"),
    ("oracle по титулам, 6 чанков", "oracle по чанкам, 6 чанков"),
    # Что добавляет квота N=2 к чанк-осознанному выбору: та же верхняя
    # граница, но реранкеру разрешено то же, что и ослабленной дедупликации.
    ("oracle по чанкам, 2 чанка", "oracle по чанкам, 2 чанка, 2 на титул"),
    ("oracle по чанкам, 6 чанков", "oracle по чанкам, 6 чанков, 2 на титул"),
    # Ослабленная дедупликация против жёсткой на том же бюджете чанков.
    ("GTE top-4", "GTE top-4, 2 чанка на титул"),
    ("GTE top-6", "GTE top-6, 2 чанка на титул"),
    # Refresh против fixed на том же бюджете шагов: режимы разные, поэтому
    # пара нужна именно как парный тест, а не как соседние строки таблицы.
    ("GTE top-2", "GTE refresh 1024, 2 шага"),
    ("GTE top-4", "GTE refresh 1024, 4 шага"),
    ("GTE top-6", "GTE refresh 1024, 6 шагов"),
    # Цена обрезки запроса: те же шесть шагов refresh при 256 и 1024 токенах.
    ("GTE refresh 256, 6 шагов", "GTE refresh 1024, 6 шагов"),
    # Вклад обучения линии A: та же башня, тот же бюджет, отличается только
    # чекпоинт. Пара одна на все датасеты — метки внутри группы уникальны,
    # поэтому она разворачивается в свой тест на каждом из них.
    (
        "Прямой поиск, zero-shot GTE, 6 шагов, N=2",
        "Прямой поиск, обученная башня, 6 шагов, N=2",
    ),
    # Обученный прямой поиск против сильнейшего first-stage на том же бюджете
    # чанков: без этой строки цифра линии A сравнивается только сама с собой.
    (
        "GTE top-6, 2 чанка на титул",
        "Прямой поиск, обученная башня, 6 шагов, N=2",
    ),
    # Цена выбора чекпоинта. Пик eval-кривой отстоит от плато примерно на одну
    # сигму, поэтому «лучший» чекпоинт может оказаться argmax по шуму; парный
    # тест на полном dev отвечает, значил ли этот выбор хоть что-то.
    (
        "Прямой поиск, обученная башня, 6 шагов, N=2",
        "Прямой поиск, последний чекпоинт, 6 шагов, N=2",
    ),
    # Сколько ридер отвечает по памяти без всякого контекста. На мультихопных
    # датасетах вопрос почти не стоял, а на однохопных бенчмарках Search-R1 он
    # центральный: у них Direct Inference без ретривала даёт на TriviaQA .408
    # при итоговых .638, то есть большая часть числа — знания модели.
    ("no retrieval", "Прямой поиск, zero-shot GTE, 6 шагов, N=2"),
    # Ablation квоты на инференсе, обе руки. Базой везде стоит N=2 — не потому
    # что она лучше, а потому что она действующая: знак теста читается как
    # «что даёт отклонение от умолчания». Zero-shot нужен как контроль: у него
    # нет N при обучении, поэтому его кривая по N — чистый инференсный эффект,
    # а разница форм двух кривых и есть цена рассогласования train/test.
    (
        "Прямой поиск, обученная башня, 6 шагов, N=2",
        "Прямой поиск, обученная башня, 6 шагов, N=1",
    ),
    (
        "Прямой поиск, обученная башня, 6 шагов, N=2",
        "Прямой поиск, обученная башня, 6 шагов, без квоты",
    ),
    (
        "Прямой поиск, zero-shot GTE, 6 шагов, N=2",
        "Прямой поиск, zero-shot GTE, 6 шагов, N=1",
    ),
    (
        "Прямой поиск, zero-shot GTE, 6 шагов, N=2",
        "Прямой поиск, zero-shot GTE, 6 шагов, без квоты",
    ),
)


def percent(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value * 100:.2f}"


def collect() -> list[tuple[Path, dict[str, Any], dict[str, Any] | None]]:
    collected = []
    for run in runlib.iter_run_dirs():
        manifest = runlib.read_manifest(run)
        collected.append((run, manifest, runlib.read_metrics(run)))
    # Порядок строк задаётся полем order из конфига: baseline снизу вверх по
    # силе, oracle последним. Раны без order уходят в конец по имени.
    return sorted(
        collected,
        key=lambda row: (row[1].get("config", {}).get("order", 10_000), row[0].name),
    )


def render_index(rows: Sequence[tuple[Path, dict[str, Any], dict[str, Any] | None]]) -> str:
    lines = [
        "# Реестр ранов",
        "",
        "Генерируется `python exp.py index`. Руками не править.",
        "",
        "| Ран | Датасет | Метка | Реранкер | Шагов | Статус | EM | F1 | Judge | title EM |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for run, manifest, metrics in rows:
        config = manifest.get("config", {})
        retrieve = config.get("retrieve", {})
        metrics = metrics or {}
        reranker = retrieve.get("reranker") or retrieve.get("context") or "—"
        rel = run.relative_to(runlib.RUNS).as_posix()
        lines.append(
            f"| [`{run.name}`]({rel}/manifest.json) "
            f"| {runlib.dataset_key(config)} "
            f"| {config.get('label', '—')} "
            f"| {reranker} "
            f"| {retrieve.get('steps', '—')} "
            f"| {manifest.get('status', '—')}"
            f"{' (backfill)' if manifest.get('backfilled') else ''} "
            f"| {percent(metrics.get('em'))} "
            f"| {percent(metrics.get('f1'))} "
            f"| {percent(metrics.get('judge'))} "
            f"| {percent(metrics.get('title_em'))} |"
        )

    lines += ["", "## Прежние имена файлов", ""]
    lines += [
        "До 2026-07-28 раны лежали в плоском `runs/` с параметрами в именах.",
        "Миграция (`src/migrate_runs.py`) перенесла файлы в каталоги ранов;",
        "симлинки со старыми именами, какое-то время сохранявшиеся ради команд",
        "из `docs/pipeline.md`, давно удалены. Точные команды каждого рана — в",
        "его `cmd.sh`.",
    ]

    lines += [
        "",
        "## Архив",
        "",
        "Раны, исключённые из сравнения, `exp.py archive` уносит в",
        "`runs/_archive/` и записывает причину в его README; каталог",
        "появляется вместе с первым архивированным раном.",
        "",
    ]
    return "\n".join(lines)


def comparable(
    rows: Sequence[tuple[Path, dict[str, Any], dict[str, Any] | None]]
) -> dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]]:
    """Раны по датасетам; внутри датасета — только сопоставимые между собой.

    Отбор двухступенчатый, и обе ступени нужны. Датасеты разделяются жёстко:
    у HotpotQA, 2Wiki и MuSiQue разные вопросы и разный потолок покрытия
    корпуса, поэтому их строки нельзя ни ставить в одну таблицу, ни
    сравнивать парным тестом. Внутри датасета отсеивается всё, что не
    совпадает с модальным числом примеров: так из витрины выпадают
    smoke-раны на 20 вопросах.
    """
    groups: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {}
    for run, manifest, metrics in rows:
        if manifest.get("status") != "ok" or not metrics or not metrics.get("samples"):
            continue
        dataset = runlib.dataset_key(manifest.get("config", {}))
        groups.setdefault(dataset, []).append((run, manifest, metrics))

    selected: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {}
    # Порядок таблиц задаётся реестром датасетов, а не порядком обхода
    # каталогов: витрина не должна переставлять разделы от рана к рану.
    for dataset in runlib.DATASETS:
        members = groups.pop(dataset, [])
        if not members:
            continue
        sizes = [metrics["samples"] for _, _, metrics in members]
        full = max(set(sizes), key=sizes.count)
        dropped = [run.name for run, _, metrics in members if metrics["samples"] != full]
        if dropped:
            LOG.warning(
                "Вне таблицы %s (другое число примеров, не %d): %s",
                dataset, full, ", ".join(dropped),
            )
        selected[dataset] = [
            (run, manifest, metrics)
            for run, manifest, metrics in members
            if metrics["samples"] == full
        ]
    for dataset in groups:
        LOG.warning("Датасет %s не зарегистрирован в runlib.DATASETS", dataset)
    return selected


def build_dataset_table(
    dataset: str,
    runs: Sequence[tuple[Path, dict[str, Any], dict[str, Any]]],
) -> str | None:
    meta = runlib.dataset_meta(dataset)
    labels = {manifest["config"]["label"] for _, manifest, _ in runs}
    # Путь берётся у runlib, а не пишется строкой: подпроцесс идёт с
    # cwd=REPO, а модуль лежит в src/, и голое имя здесь молча ломало
    # `make report`, оставляя таблицу в RESULTS.md прежней.
    argv: list[str] = [
        str(runlib.VENV_PYTHON),
        str(runlib.pipeline_script("report_phase0.py")),
    ]
    # Оба флага необязательны и по разным причинам: потолка нет там, где нет
    # gold-титулов, алиасы есть только там, где официальный эвал считает EM
    # максимумом по списку допустимых ответов.
    if meta.get("coverage") is not None:
        argv += ["--coverage", str(meta["coverage"])]
    if meta.get("aliases") is not None:
        argv += ["--aliases", str(meta["aliases"])]
    for run, manifest, _ in runs:
        label = manifest["config"]["label"]
        argv += ["--run", f"{label}={run / runlib.JUDGE_FILE}"]
    for baseline, variant in DEFAULT_PAIRS:
        if baseline in labels and variant in labels:
            argv += ["--pair", baseline, variant]
    LOG.info("Пересчитываю таблицу %s: %d ранов", dataset, len(runs))
    completed = subprocess.run(
        argv, cwd=runlib.REPO, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        LOG.error("report_phase0.py упал:\n%s", completed.stderr.strip()[-2000:])
        return None
    # Отбрасываем итоговый JSON-дамп: в файле нужны только таблицы.
    table, _, _ = completed.stdout.partition("\n{")
    return table.rstrip()


def build_results(rows) -> str | None:
    groups = comparable(rows)
    if not groups:
        LOG.warning("Нет сопоставимых ранов, RESULTS.md не трогаю")
        return None
    blocks = []
    for dataset, runs in groups.items():
        table = build_dataset_table(dataset, runs)
        if table is None:
            return None
        meta = runlib.dataset_meta(dataset)
        samples = runs[0][2]["samples"]
        blocks.append(
            f"### {meta['label']} — {samples} примеров\n\n{table}"
        )
    return "\n\n".join(blocks)


def splice_results(table: str) -> None:
    """Заменить сгенерированный блок в RESULTS.md, сохранив ручной разбор."""
    text = RESULTS_FILE.read_text(encoding="utf-8")
    block = f"{TABLE_START}\n\n{table}\n\n{TABLE_END}"
    if TABLE_START in text and TABLE_END in text:
        head, _, rest = text.partition(TABLE_START)
        _, _, tail = rest.partition(TABLE_END)
        RESULTS_FILE.write_text(head + block + tail, encoding="utf-8")
    else:
        raise SystemExit(
            f"В {RESULTS_FILE.name} нет маркеров {TABLE_START} / {TABLE_END}: "
            "непонятно, куда вставлять таблицу"
        )
    LOG.info("Обновлён %s", RESULTS_FILE.name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-index", action="store_true")
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
    rows = collect()
    runlib.INDEX_FILE.write_text(render_index(rows) + "\n", encoding="utf-8")
    LOG.info("Обновлён %s (%d ранов)", runlib.INDEX_FILE.name, len(rows))
    if args.only_index:
        return 0
    table = build_results(rows)
    if table:
        splice_results(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
