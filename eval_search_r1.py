import argparse
import json
import os
import random
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import transformers
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
JUDGE_DIR = os.path.dirname(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

from envs import (  # noqa: E402
    QADatasetAdapter,
    Retrieval2WikiMultihopQA,
    RetrievalHotPotQA,
    RetrievalMusique,
)


DEFAULT_MODEL_ID = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
DEFAULT_RETRIEVER_MODEL = "intfloat/e5-base-v2"
SEARCH_R1_PROMPT = """Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>.
Question: {question}
"""
SEARCH_STOP_SEQUENCES = [
    "</search>",
    " </search>",
    "</search>\n",
    " </search>\n",
    "</search>\n\n",
    " </search>\n\n",
]
QWEN25_EOS_TOKEN_IDS = {151643, 151645}


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calc_fact_f1_em(predicted_support_idxs: Sequence[int], gt_support_idxs: Sequence[int]) -> Tuple[float, float]:
    pred_sf = set(map(int, predicted_support_idxs))
    gt_sf = set(map(int, gt_support_idxs))

    tp = len(pred_sf.intersection(gt_sf))
    fp = len(pred_sf.difference(gt_sf))
    fn = len(gt_sf.difference(pred_sf))

    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if gt_sf.issubset(pred_sf) else 0.0

    if not pred_sf and not gt_sf:
        f1, em = 1.0, 1.0
    return f1, em


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences: Sequence[str], tokenizer: transformers.PreTrainedTokenizerBase):
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False)
            for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._target_tensors: Dict[Tuple[torch.device, int], torch.Tensor] = {}

    def _target_tensor(self, target_index: int, device: torch.device) -> torch.Tensor:
        key = (device, target_index)
        if key not in self._target_tensors:
            self._target_tensors[key] = torch.as_tensor(self.target_ids[target_index], device=device)
        return self._target_tensors[key]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for target_index, target_len in enumerate(self.target_lengths):
            if target_len == 0 or input_ids.shape[1] < target_len:
                continue
            target = self._target_tensor(target_index, input_ids.device)
            if torch.equal(input_ids[0, -target_len:], target):
                return True
        return False


@dataclass
class SearchStep:
    query: str
    chunk_idx: int
    chunk_text: str
    score: float


def parse_search_query(text: str) -> Optional[str]:
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def parse_answer(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()

    stripped_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return stripped_lines[-1] if stripped_lines else ""


def normalize_tokenizer_eos(tokenizer: transformers.PreTrainedTokenizerBase) -> List[int]:
    eos_ids = set(QWEN25_EOS_TOKEN_IDS)
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, Iterable):
        eos_ids.update(int(token_id) for token_id in eos_token_id)
    return sorted(eos_ids)


def model_input_device(model: transformers.PreTrainedModel) -> torch.device:
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def prepare_prompt(tokenizer: transformers.PreTrainedTokenizerBase, question: str) -> str:
    prompt = SEARCH_R1_PROMPT.format(question=question)
    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return prompt


def encode_prompt(
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def search_local_chunks(
    query: str,
    available_indices: Sequence[int],
    paragraph_embeddings: torch.Tensor,
    retriever_model: SentenceTransformer,
    top_k: int,
) -> List[Tuple[int, float]]:
    if not query or not available_indices:
        return []

    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    available_embeddings = paragraph_embeddings[list(available_indices)]
    similarities = util.cos_sim(query_embedding, available_embeddings)[0]
    k = min(top_k, len(available_indices))
    top = torch.topk(similarities, k=k)

    results = []
    for score, local_idx in zip(top.values.tolist(), top.indices.tolist()):
        results.append((int(available_indices[int(local_idx)]), float(score)))
    return results


@torch.no_grad()
def run_search_r1_on_sample(
    sample: Dict,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerBase,
    retriever: SentenceTransformer,
    stopping_criteria: transformers.StoppingCriteriaList,
    eos_token_ids: Sequence[int],
    max_searches: int,
    max_new_tokens: int,
    retrieval_top_k: int,
    force_answer: bool,
    verbose: bool,
) -> Dict:
    chunks = list(sample["chunks"])
    paragraph_embeddings = retriever.encode(chunks, convert_to_tensor=True)
    input_device = model_input_device(model)
    prompt = prepare_prompt(tokenizer, sample["question"])
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, list):
        eos_token_id = eos_token_id[0]

    generated_parts: List[str] = []
    search_steps: List[SearchStep] = []
    pred_idx: List[int] = []
    available_indices = list(range(len(chunks)))

    if verbose:
        print(prompt, end="")

    for _ in range(max_searches):
        inputs = encode_prompt(tokenizer, prompt, input_device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            pad_token_id=eos_token_id,
            do_sample=False,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        raw_output_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
        generated_parts.append(raw_output_text)

        if verbose:
            print(raw_output_text, end="")

        last_token_id = int(outputs[0][-1].item())
        if "<answer>" in output_text or last_token_id in eos_token_ids:
            break

        search_query = parse_search_query(output_text)
        if not search_query:
            break

        search_results = search_local_chunks(
            search_query,
            available_indices,
            paragraph_embeddings,
            retriever,
            retrieval_top_k,
        )
        if not search_results:
            break

        chunk_idx, score = search_results[0]
        chunk_text = chunks[chunk_idx]
        pred_idx.append(chunk_idx)
        search_steps.append(SearchStep(search_query, chunk_idx, chunk_text, score))
        available_indices.remove(chunk_idx)

        information = f"<information>{chunk_text}</information>\n"
        prompt += raw_output_text + information
        if verbose:
            print(f"\n[SEARCH] query={search_query!r} idx={chunk_idx} score={score:.4f}")
            print(information, end="")

    full_generation = "".join(generated_parts)
    if force_answer and "<answer>" not in full_generation:
        force_answer_prompt = (
            prompt
            + "\nYou have reached the maximum number of searches. "
            + "Please provide the final answer inside <answer> and </answer> now.\n"
        )
        inputs = encode_prompt(tokenizer, force_answer_prompt, input_device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=eos_token_id,
            do_sample=False,
        )
        forced_answer_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )
        full_generation += forced_answer_text
        if verbose:
            print(forced_answer_text)

    return {
        "pred_idx": [int(idx) for idx in pred_idx],
        "pred_texts": [chunks[idx] for idx in pred_idx],
        "pred_text": [chunks[idx] for idx in pred_idx],
        "search_queries": [step.query for step in search_steps],
        "search_steps": [
            {
                "query": step.query,
                "chunk_idx": int(step.chunk_idx),
                "chunk_text": step.chunk_text,
                "score": step.score,
            }
            for step in search_steps
        ],
        "generated_answer": parse_answer(full_generation),
        "raw_generation": full_generation,
        "num_steps": len(search_steps),
    }


def build_dataset(
    dataset_name: str,
    data_path: str,
    split: str,
    seed: int,
    min_chunks: int,
    include_bridge_comparison: bool,
) -> QADatasetAdapter:
    if dataset_name == "hotpotqa":
        raw_dataset = RetrievalHotPotQA(path=data_path, split=split, seed=seed)
    elif dataset_name == "musique":
        raw_dataset = RetrievalMusique(path=data_path, split=split, seed=seed)
    elif dataset_name == "2wiki":
        raw_dataset = Retrieval2WikiMultihopQA(
            path=data_path,
            split=split,
            skip_bridge_comparison=not include_bridge_comparison,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = QADatasetAdapter(raw_dataset)
    if min_chunks <= 0:
        return dataset
    return MinChunksDataset(dataset, min_chunks=min_chunks)


class MinChunksDataset:
    def __init__(self, dataset: QADatasetAdapter, min_chunks: int):
        self.dataset = dataset
        self.indices = [
            idx for idx in range(len(dataset))
            if len(dataset[idx]["chunks"]) >= min_chunks
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict:
        return self.dataset[self.indices[index]]


def default_data_path(dataset_name: str) -> str:
    if dataset_name == "hotpotqa":
        return os.path.join(JUDGE_DIR, "datasets", "data_sources", "hotpotqa")
    if dataset_name == "musique":
        return os.path.join(JUDGE_DIR, "datasets", "data_sources", "musique")
    if dataset_name == "2wiki":
        return os.path.join(
            JUDGE_DIR,
            "datasets",
            "data_sources",
            "2WikiMultiHopQA",
            "data_ids_april7",
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def default_output_path(dataset_name: str, seed: int) -> str:
    return os.path.join(REPO_DIR, "runs", f"search_r1_{dataset_name}", f"eval_seed{seed}.jsonl")


def resolve_dataset_split(dataset_name: str, split: str) -> str:
    if dataset_name == "2wiki":
        if split == "eval":
            return "dev"
        if split not in {"train", "dev", "test"}:
            raise ValueError("2wiki supports split=train/dev/test; split=eval is accepted as dev.")
        return split

    if split not in {"train", "eval", "all"}:
        raise ValueError(f"{dataset_name} supports split=train/eval/all.")
    return split


def read_existing(path: str) -> Dict[str, Dict]:
    existing = {}
    if not os.path.exists(path):
        return existing

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = item.get("id")
            if sample_id is not None:
                existing[str(sample_id)] = item
    return existing


def summarize_metrics(items: Sequence[Dict]) -> Tuple[float, float]:
    if not items:
        return 0.0, 0.0

    f1_values = []
    em_values = []
    for item in items:
        if "f1" in item and "em" in item:
            f1_values.append(float(item["f1"]))
            em_values.append(float(item["em"]))
        else:
            f1, em = calc_fact_f1_em(item.get("pred_idx", []), item.get("sf_idx", []))
            f1_values.append(f1)
            em_values.append(em)
    return float(np.mean(f1_values)), float(np.mean(em_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Search-R1 inference over HotPotQA, MuSiQue, or 2Wiki and save retrieved "
            "supporting facts in Q-RAG eval_retriever JSONL format."
        )
    )
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wiki"], default="hotpotqa")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--split", choices=["train", "eval", "all", "dev", "test"], default="eval")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--retriever-model", default=DEFAULT_RETRIEVER_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-searches", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retrieval-top-k", type=int, default=1)
    parser.add_argument("--min-chunks", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--force-answer", action="store_true")
    parser.add_argument(
        "--include-bridge-comparison",
        action="store_true",
        help="For 2Wiki only: include bridge_comparison samples. Default is to exclude them.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    split = resolve_dataset_split(args.dataset, args.split)
    data_path = args.data_path or default_data_path(args.dataset)
    output_path = args.output_path or default_output_path(args.dataset, args.seed)
    max_searches = args.max_searches
    if max_searches is None:
        max_searches = 4 if args.dataset == "musique" else 2

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.overwrite and os.path.exists(output_path):
        os.remove(output_path)

    print(f"Dataset: {args.dataset} split={split} path={data_path}")
    if args.dataset == "2wiki" and not args.include_bridge_comparison:
        print("2Wiki bridge_comparison samples are excluded.")
    dataset = build_dataset(
        args.dataset,
        data_path,
        split,
        args.seed,
        args.min_chunks,
        args.include_bridge_comparison,
    )
    total_available = max(0, len(dataset) - args.start_index)
    total_to_process = total_available if args.max_samples < 0 else min(args.max_samples, total_available)
    end_index = args.start_index + total_to_process
    print(f"Samples: start={args.start_index} end={end_index} total_to_process={total_to_process}")
    print(f"Search limit: max_searches={max_searches}")
    print(f"Writing JSONL to {output_path}")

    existing = read_existing(output_path)
    existing_items = list(existing.values())
    if existing_items:
        f1, em = summarize_metrics(existing_items)
        print(f"Found {len(existing_items)} existing samples | fact EM={em:.4f} F1={f1:.4f}")

    if total_to_process == 0:
        print("Nothing to process.")
        return

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Loading Search-R1 model: {args.model_id}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    print(f"Loading retriever: {args.retriever_model}")
    retriever = SentenceTransformer(args.retriever_model)

    stopping_criteria = transformers.StoppingCriteriaList(
        [StopOnSequence(SEARCH_STOP_SEQUENCES, tokenizer)]
    )
    eos_token_ids = normalize_tokenizer_eos(tokenizer)

    processed_items = list(existing_items)
    mode = "a"
    with open(output_path, mode, encoding="utf-8") as f:
        iterator = range(args.start_index, end_index)
        for sample_index in tqdm(iterator, desc="Search-R1 eval", ncols=100):
            sample = dataset[sample_index]
            sample_id = str(sample["id"])
            if sample_id in existing:
                continue

            result = run_search_r1_on_sample(
                sample=sample,
                model=model,
                tokenizer=tokenizer,
                retriever=retriever,
                stopping_criteria=stopping_criteria,
                eos_token_ids=eos_token_ids,
                max_searches=max_searches,
                max_new_tokens=args.max_new_tokens,
                retrieval_top_k=args.retrieval_top_k,
                force_answer=args.force_answer,
                verbose=args.verbose,
            )
            f1, em = calc_fact_f1_em(result["pred_idx"], sample["sf_idx"])
            entry = {
                "id": sample["id"],
                "dataset": args.dataset,
                "split": split,
                "question": sample["question"],
                "answer": sample["answer"],
                "sf_idx": [int(idx) for idx in sample["sf_idx"]],
                "pred_idx": result["pred_idx"],
                "sf_texts": [sample["chunks"][idx] for idx in sample["sf_idx"]],
                "pred_texts": result["pred_texts"],
                "pred_text": result["pred_text"],
                "search_queries": result["search_queries"],
                "search_steps": result["search_steps"],
                "generated_answer": result["generated_answer"],
                "raw_generation": result["raw_generation"],
                "num_steps": result["num_steps"],
                "max_searches": max_searches,
                "f1": f1,
                "em": em,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            existing[sample_id] = entry
            processed_items.append(entry)

    fact_f1, fact_em = summarize_metrics(processed_items)
    print(
        f"Done. Logged {len(processed_items)} samples to {output_path} | "
        f"fact EM={fact_em:.4f} F1={fact_f1:.4f}"
    )


if __name__ == "__main__":
    main()
