"""
Run question-only vs gold-supporting-facts answer success-rate experiments.

For each question, this script generates N answers in two modes:

1. question_only: empty context, only the question is informative.
2. with_support_facts: context contains all gold supporting fact sentences.

Each generated answer is judged against the gold answer and the script writes
per-question success rates plus histogram PNGs.

Example:
    python run_success_rate_experiment.py \
        --dataset all \
        --split dev \
        --reader-model Qwen/Qwen3-4B \
        --reader-base-url http://localhost:8000 \
        --judge-model Qwen/Qwen3-4B \
        --judge-base-url http://localhost:8000 \
        --output-dir output/success_rate_experiment
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import statistics
import string
from collections import Counter
from typing import Any

import gig_common as G


DATA_ROOT = "/home/a.anokhin/Judge/datasets/data_sources"

MODEL_NAME = "Qwen/Qwen3-4B"
N_GENERATIONS = 8
MAX_TOKENS = 64
TEMPERATURE = 1.0
MAX_WORKERS = 16
SAVE_EVERY = 100

HOTPOT_PATHS = {
    "train": ["hotpotqa/hotpot_train_v1.1.json"],
    "dev": ["hotpotqa/hotpot_dev_distractor_v1.json"],
    "fullwiki_dev": ["hotpotqa/hotpot_dev_fullwiki_v1.json"],
}
HOTPOT_PATHS["all"] = HOTPOT_PATHS["train"] + HOTPOT_PATHS["dev"]

TWOWIKI_PATHS = {
    "train": ["2WikiMultiHopQA/data_ids_april7/train.json"],
    "dev": ["2WikiMultiHopQA/data_ids_april7/dev.json"],
    "test": ["2WikiMultiHopQA/data_ids_april7/test.json"],
}
TWOWIKI_PATHS["all"] = TWOWIKI_PATHS["train"] + TWOWIKI_PATHS["dev"]

MODE_QUESTION_ONLY = "question_only"
MODE_WITH_SUPPORT = "with_support_facts"

JUDGE_SYSTEM_PROMPT = """You are an answer verification system for question-answering tasks.
Your task is to compare a PREDICTED ANSWER against the GROUNDTRUTH ANSWER and determine if they are semantically equivalent.
Strictly follow these rules:
Focus on factual equivalence, not exact string matching.
Ignore differences in wording, phrasing, or formatting (e.g., "USA" vs "United States", "2" vs "two").
For numeric answers, accept equivalent representations (e.g., "1,000" = "1000", "50%" = "0.5").
For dates, accept equivalent representations (e.g., "05/01/2026" vs "May 1, 2026" vs "01.05.2026").
If the predicted answer contains the correct information but adds extra details, reply YES.
If the predicted answer is partially correct but misses key information present in the groundtruth, reply NO.

Reply with ONLY one word: YES or NO."""

JUDGE_USER_TEMPLATE = """Question: {question}
Gold answer: {gold_answer}
Candidate answer: {candidate}

Is the candidate equivalent to the gold answer? Reply YES or NO:"""


def canonical_dataset_name(name: str) -> str:
    aliases = {
        "hotpot": "hotpotqa",
        "hotpotqa": "hotpotqa",
        "2wiki": "2wiki",
        "twowiki": "2wiki",
        "2wikimultihopqa": "2wiki",
        "all": "all",
    }
    key = name.lower()
    if key not in aliases:
        raise ValueError(f"Unknown dataset: {name}")
    return aliases[key]


def _load_json_array(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_sample(dataset: str, sample: dict, source_split: str) -> dict:
    return {
        "dataset": dataset,
        "split": source_split,
        "id": sample["_id"],
        "question": sample["question"],
        "answer": sample["answer"],
        "context": sample["context"],
        "supporting_facts": sample["supporting_facts"],
        "level": sample.get("level", ""),
        "type": sample.get("type", ""),
    }


def load_hotpotqa(split: str, data_root: str) -> list[dict]:
    if split not in HOTPOT_PATHS:
        raise ValueError(f"HotpotQA split must be one of {sorted(HOTPOT_PATHS)}; got {split!r}")

    samples = []
    for rel_path in HOTPOT_PATHS[split]:
        path = os.path.join(data_root, rel_path)
        source_split = "train" if "train" in rel_path else "dev"
        if "fullwiki" in rel_path:
            source_split = "fullwiki_dev"
        samples.extend(
            _normalize_sample("hotpotqa", sample, source_split)
            for sample in _load_json_array(path)
        )
    return samples


def load_2wiki(split: str, data_root: str, skip_bridge_comparison: bool) -> list[dict]:
    if split not in TWOWIKI_PATHS:
        raise ValueError(f"2Wiki split must be one of {sorted(TWOWIKI_PATHS)}; got {split!r}")

    samples = []
    for rel_path in TWOWIKI_PATHS[split]:
        path = os.path.join(data_root, rel_path)
        source_split = os.path.splitext(os.path.basename(path))[0]
        for sample in _load_json_array(path):
            if skip_bridge_comparison and sample.get("type") == "bridge_comparison":
                continue
            samples.append(_normalize_sample("2wiki", sample, source_split))
    return samples


def load_dataset(args: argparse.Namespace, dataset: str) -> list[dict]:
    if dataset == "hotpotqa":
        samples = load_hotpotqa(args.split, args.data_root)
        if not args.keep_nonstandard_hotpot:
            before = len(samples)
            samples = [sample for sample in samples if is_standard_hotpot_shape(sample)]
            removed = before - len(samples)
            if removed:
                print(f"Filtered {removed} HotpotQA samples without exactly 10 chunks and 2 sf chunks")
    elif dataset == "2wiki":
        samples = load_2wiki(args.split, args.data_root, args.skip_bridge_comparison)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if args.limit > 0:
        samples = samples[:args.limit]
    return samples


def is_standard_hotpot_shape(sample: dict) -> bool:
    qrag_context = build_qrag_chunks(sample)
    return qrag_context["chunk_count"] == 10 and qrag_context["sf_count"] == 2


def build_qrag_chunks(sample: dict) -> dict:
    """Build chunks exactly like Q-RAG's QADatasetAdapter for Hotpot-style data."""
    supporting_titles = {
        fact[0]
        for fact in sample.get("supporting_facts", [])
        if isinstance(fact, list) and fact
    }

    chunks = []
    sf_idx = []
    for idx, (title, sentences) in enumerate(sample.get("context", [])):
        if title in supporting_titles:
            sf_idx.append(idx)
        chunks.append(f"{title} {' '.join(sentences)}")

    sf_idx_set = set(sf_idx)
    support_chunks = [chunks[idx] for idx in sf_idx]
    noise_idx = [idx for idx in range(len(chunks)) if idx not in sf_idx_set]
    return {
        "chunks": chunks,
        "sf_idx": sf_idx,
        "noise_idx": noise_idx,
        "supporting_facts_text": support_chunks,
        "chunk_count": len(chunks),
        "sf_count": len(sf_idx),
    }


def strip_answer(text: str) -> str:
    return G.strip_candidate_prefix(text).strip()


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower().strip())))


def exact_judgement(gold_answer: str, candidate: str) -> int:
    return int(normalize_answer(gold_answer) == normalize_answer(candidate))


def parse_yes_no(response_text: str) -> int | None:
    text = response_text.strip().upper()
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^A-Z]", "", first_word)
    if first_word == "YES":
        return 1
    if first_word == "NO":
        return 0
    if "YES" in text and "NO" not in text:
        return 1
    if "NO" in text and "YES" not in text:
        return 0
    return None


def judge_single_llm(
    question: str,
    gold_answer: str,
    candidate: str,
    model: str,
    base_url: str,
    enable_thinking: bool,
) -> int:
    user_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        candidate=strip_answer(candidate),
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(G.DEFAULT_MAX_RETRIES):
        try:
            response = G.chat_completion(
                messages,
                model=model,
                base_url=base_url,
                max_tokens=8,
                temperature=0.0,
                enable_thinking=enable_thinking,
                n=1,
            )[0]
            result = parse_yes_no(response)
            if result is not None:
                return result
            if attempt == G.DEFAULT_MAX_RETRIES - 1:
                print(f"[JUDGE PARSE ERROR] response={response[:120]!r}")
        except Exception as exc:
            if attempt == G.DEFAULT_MAX_RETRIES - 1:
                print(f"[JUDGE ERROR] {exc}")
    return -1


def judge_answers(
    question: str,
    gold_answer: str,
    candidates: list[str],
    args: argparse.Namespace,
) -> list[int]:
    if args.judge_mode == "exact":
        return [exact_judgement(gold_answer, candidate) for candidate in candidates]

    return [
        judge_single_llm(
            question,
            gold_answer,
            candidate,
            model=args.judge_model,
            base_url=args.judge_base_url,
            enable_thinking=args.enable_judge_thinking,
        )
        for candidate in candidates
    ]


def generate_answers(
    question: str,
    context: str,
    args: argparse.Namespace,
) -> list[str]:
    raw_answers = G.chat_completion(
        G.qa_messages(question, context=context),
        model=args.reader_model,
        base_url=args.reader_base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_reader_thinking,
        n=args.n_generations,
    )
    return [strip_answer(answer) for answer in raw_answers]


def success_rate(judgements: list[int]) -> float:
    if not judgements:
        return 0.0
    return sum(1 for value in judgements if value == 1) / len(judgements)


def build_mode_result(
    sample: dict,
    context: str,
    args: argparse.Namespace,
) -> dict:
    generations = generate_answers(sample["question"], context, args)
    judgements = judge_answers(sample["question"], sample["answer"], generations, args)
    return {
        "context": context,
        "generations": generations,
        "judgements": judgements,
        "success_rate": success_rate(judgements),
    }


def make_process_fn(args: argparse.Namespace):
    def process(sample: dict) -> dict:
        qrag_context = build_qrag_chunks(sample)
        supporting_texts = qrag_context["supporting_facts_text"]
        support_context = "\n\n---\n\n".join(supporting_texts)

        question_only = build_mode_result(sample, context="", args=args)
        with_support_facts = build_mode_result(sample, context=support_context, args=args)

        return {
            "dataset": sample["dataset"],
            "split": sample["split"],
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "level": sample.get("level", ""),
            "type": sample.get("type", ""),
            "supporting_facts": sample.get("supporting_facts", []),
            "chunks": qrag_context["chunks"],
            "sf_idx": qrag_context["sf_idx"],
            "noise_idx": qrag_context["noise_idx"],
            "chunk_count": qrag_context["chunk_count"],
            "sf_count": qrag_context["sf_count"],
            "supporting_facts_text": supporting_texts,
            MODE_QUESTION_ONLY: question_only,
            MODE_WITH_SUPPORT: with_support_facts,
            "question_only_success_rate": question_only["success_rate"],
            "with_support_facts_success_rate": with_support_facts["success_rate"],
            "n_generations": args.n_generations,
            "reader_model": args.reader_model,
            "judge_mode": args.judge_mode,
            "judge_model": args.judge_model if args.judge_mode == "llm" else "",
        }

    return process


def result_stem(dataset: str, split: str, limit: int) -> str:
    suffix = f"_n{limit}" if limit > 0 else ""
    return f"{dataset}_{split}{suffix}_qrag_chunks_success_rates"


def mode_counts(record: dict, mode: str) -> tuple[int, int, int]:
    judgements = record.get(mode, {}).get("judgements", [])
    correct = sum(1 for value in judgements if value == 1)
    failed = sum(1 for value in judgements if value == -1)
    return correct, len(judgements), failed


def write_csv_summary(path: str, records: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fields = [
        "dataset",
        "split",
        "id",
        "question",
        "answer",
        "type",
        "level",
        "chunk_count",
        "sf_count",
        "sf_idx",
        "noise_idx",
        "question_only_success_rate",
        "with_support_facts_success_rate",
        "delta_success_rate",
        "question_only_correct",
        "question_only_total",
        "question_only_failed_judge",
        "with_support_facts_correct",
        "with_support_facts_total",
        "with_support_facts_failed_judge",
        "supporting_facts_text",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            q_correct, q_total, q_failed = mode_counts(record, MODE_QUESTION_ONLY)
            s_correct, s_total, s_failed = mode_counts(record, MODE_WITH_SUPPORT)
            q_rate = record.get("question_only_success_rate", 0.0)
            s_rate = record.get("with_support_facts_success_rate", 0.0)
            writer.writerow({
                "dataset": record.get("dataset", ""),
                "split": record.get("split", ""),
                "id": record.get("id", ""),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "type": record.get("type", ""),
                "level": record.get("level", ""),
                "chunk_count": record.get("chunk_count", ""),
                "sf_count": record.get("sf_count", ""),
                "sf_idx": " ".join(map(str, record.get("sf_idx", []))),
                "noise_idx": " ".join(map(str, record.get("noise_idx", []))),
                "question_only_success_rate": q_rate,
                "with_support_facts_success_rate": s_rate,
                "delta_success_rate": s_rate - q_rate,
                "question_only_correct": q_correct,
                "question_only_total": q_total,
                "question_only_failed_judge": q_failed,
                "with_support_facts_correct": s_correct,
                "with_support_facts_total": s_total,
                "with_support_facts_failed_judge": s_failed,
                "supporting_facts_text": " || ".join(record.get("supporting_facts_text", [])),
            })


def plot_histograms(
    records: list[dict],
    dataset: str,
    split: str,
    plots_dir: str,
    n_generations: int,
) -> None:
    if not records:
        print(f"No records to plot for {dataset}/{split}")
        return

    use_matplotlib = True
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        use_matplotlib = False
        print(f"[WARN] Could not import matplotlib; writing SVG histograms instead: {exc}")

    os.makedirs(plots_dir, exist_ok=True)
    metrics = [
        ("question_only_success_rate", "Question only"),
        ("with_support_facts_success_rate", "Question + gold supporting facts"),
    ]
    xs = [i / n_generations for i in range(n_generations + 1)]
    bar_width = 0.75 / n_generations

    for metric, title in metrics:
        counts = Counter(round(float(record.get(metric, 0.0)) * n_generations) for record in records)
        ys = [counts.get(i, 0) for i in range(n_generations + 1)]

        if not use_matplotlib:
            path = os.path.join(plots_dir, f"{dataset}_{split}_{metric}_hist.svg")
            write_histogram_svg(path, xs, ys, f"{dataset} {split}: {title}")
            print(f"Plot: {path}")
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(xs, ys, width=bar_width, align="center", color="#4C78A8", edgecolor="#1F2937")
        ax.set_title(f"{dataset} {split}: {title}")
        ax.set_xlabel("success rate")
        ax.set_ylabel("questions")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{value:.3g}" for value in xs], rotation=45)
        ax.set_xlim(-bar_width, 1.0 + bar_width)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        path = os.path.join(plots_dir, f"{dataset}_{split}_{metric}_hist.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        print(f"Plot: {path}")


def write_histogram_svg(path: str, xs: list[float], ys: list[int], title: str) -> None:
    width = 900
    height = 520
    left = 74
    right = 28
    top = 58
    bottom = 82
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_y = max(ys) if ys else 0
    y_scale_max = max(1, max_y)
    bar_slot = plot_width / max(1, len(xs))
    bar_width = bar_slot * 0.68

    def x_pos(index: int) -> float:
        return left + index * bar_slot + (bar_slot - bar_width) / 2

    def y_pos(value: int) -> float:
        return top + plot_height - (value / y_scale_max) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="20" fill="#111827">'
        f'{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="#111827" stroke-width="1"/>',
    ]

    for tick in range(5):
        value = round(y_scale_max * tick / 4)
        y = y_pos(value)
        lines.append(
            f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="12" fill="#374151">'
            f'{value}</text>'
        )

    for index, (x_value, y_value) in enumerate(zip(xs, ys)):
        x = x_pos(index)
        y = y_pos(y_value)
        bar_height = top + plot_height - y
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="#4C78A8" stroke="#1F2937" stroke-width="1"/>'
        )
        if y_value:
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" '
                'font-family="Arial, sans-serif" font-size="12" fill="#111827">'
                f'{y_value}</text>'
            )
        label_x = x + bar_width / 2
        label_y = top + plot_height + 22
        lines.append(
            f'<text x="{label_x:.1f}" y="{label_y}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#374151" '
            f'transform="rotate(-45 {label_x:.1f},{label_y})">'
            f'{x_value:.3g}</text>'
        )

    lines.extend([
        f'<text x="{left + plot_width / 2}" y="{height - 18}" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="14" fill="#111827">success rate</text>',
        f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" '
        'font-family="Arial, sans-serif" font-size="14" fill="#111827" '
        f'transform="rotate(-90 18,{top + plot_height / 2})">questions</text>',
        '</svg>',
    ])

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def print_summary(records: list[dict], dataset: str, split: str) -> None:
    print(f"\nSummary for {dataset}/{split}")
    print(f"Records: {len(records)}")
    for metric in ("question_only_success_rate", "with_support_facts_success_rate"):
        values = [float(record.get(metric, 0.0)) for record in records]
        if not values:
            print(f"  {metric}: no values")
            continue
        print(
            f"  {metric}: mean={statistics.mean(values):.4f} "
            f"median={statistics.median(values):.4f} "
            f"min={min(values):.4f} max={max(values):.4f}"
        )

    q_failed = sum(mode_counts(record, MODE_QUESTION_ONLY)[2] for record in records)
    s_failed = sum(mode_counts(record, MODE_WITH_SUPPORT)[2] for record in records)
    if q_failed or s_failed:
        print(f"  failed judge calls: question_only={q_failed}, with_support_facts={s_failed}")


def run_one_dataset(args: argparse.Namespace, dataset: str) -> None:
    samples = load_dataset(args, dataset)
    print(f"\nLoaded {len(samples)} samples for {dataset}/{args.split}")

    stem = result_stem(dataset, args.split, args.limit)
    output_jsonl = os.path.join(args.output_dir, f"{stem}.jsonl")
    output_csv = os.path.join(args.output_dir, f"{stem}.csv")

    process_fn = make_process_fn(args)
    records = G.run_with_resume(
        samples,
        output_jsonl,
        process_fn,
        max_workers=args.workers,
        save_every=args.save_every,
        sanity_check=None,
    )

    write_csv_summary(output_csv, records)
    print(f"JSONL: {output_jsonl}")
    print(f"CSV:   {output_csv}")
    print_summary(records, dataset, args.split)

    if not args.no_plots:
        plot_histograms(
            records,
            dataset=dataset,
            split=args.split,
            plots_dir=args.plots_dir,
            n_generations=args.n_generations,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QA success-rate experiment")
    parser.add_argument("--dataset", choices=["hotpot", "hotpotqa", "2wiki", "twowiki", "2wikimultihopqa", "all"], required=True)
    parser.add_argument("--split", choices=["train", "dev", "all", "test", "fullwiki_dev"], default="dev")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--output-dir", default="output/success_rate_experiment")
    parser.add_argument("--plots-dir", default=None)
    parser.add_argument("--limit", type=int, default=-1, help="Optional first-N sample limit for smoke tests")
    parser.add_argument("--skip-bridge-comparison", dest="skip_bridge_comparison", action="store_true", default=True,
                        help="Skip 2Wiki bridge_comparison samples, matching Q-RAG configs (default)")
    parser.add_argument("--include-bridge-comparison", dest="skip_bridge_comparison", action="store_false",
                        help="Include 2Wiki bridge_comparison samples, which usually have 4 support chunks")
    parser.add_argument("--keep-nonstandard-hotpot", action="store_true", default=False,
                        help="Keep HotpotQA samples that do not have exactly 10 chunks and 2 sf chunks")

    parser.add_argument("--reader-model", default=MODEL_NAME)
    parser.add_argument("--reader-base-url", default=G.VLLM_BASE_URL)
    parser.add_argument("--enable-reader-thinking", action="store_true", default=False)
    parser.add_argument("--n-generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)

    parser.add_argument("--judge-mode", choices=["llm", "exact"], default="llm")
    parser.add_argument("--judge-model", default=MODEL_NAME)
    parser.add_argument("--judge-base-url", default=G.VLLM_BASE_URL)
    parser.add_argument("--enable-judge-thinking", action="store_true", default=False)

    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY)
    parser.add_argument("--no-plots", action="store_true", default=False)
    args = parser.parse_args()

    args.dataset = canonical_dataset_name(args.dataset)
    if args.dataset == "all" and args.split in {"test", "fullwiki_dev"}:
        raise SystemExit("--dataset all supports only --split train/dev/all")
    if args.n_generations <= 0:
        raise SystemExit("--n-generations must be positive")
    if args.plots_dir is None:
        args.plots_dir = os.path.join(args.output_dir, "figs")
    return args


def main() -> None:
    args = parse_args()
    datasets = ["hotpotqa", "2wiki"] if args.dataset == "all" else [args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in datasets:
        run_one_dataset(args, dataset)


if __name__ == "__main__":
    main()
