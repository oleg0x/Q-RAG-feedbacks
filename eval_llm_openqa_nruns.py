import json
import os
import argparse
import numpy as np
from collections import namedtuple, defaultdict
from typing import Tuple, Dict, List, Any, Union
#import torch.utils
from torch.utils.data import Dataset
import json
import re
import string
#import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
import sys
from prompts_and_metrics import prompts
from prompts_and_metrics.chunk_filtering import build_chunk_filter

#os.environ["VLLM_USE_TORCH_COMPILE"] = "0"
#os.environ["TORCH_COMPILE_DISABLE"] = "1"
#os.environ["VLLM_DISABLE_CUDA_GRAPHS"] = "1"

qa_instruction_prompt = """You are a factoid question answering system.
Return only the short answer itself.
The answer must be a single entity, number, date, yes/no, or short phrase.
No explanations, no reasoning, no extra text.
Do not write labels such as 'Final answer:' or 'Answer:'."""

qa_prompt = """
GIVEN PASSAGES:
{context}

QUESTION:
{question}
"""



def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))


def compute_exact_match(prediction, target):
    target, prediction = normalize_answer(target), normalize_answer(prediction)
    return int(target == prediction)


def recall(prediction, target):
    target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
    len_true = len(target)
    len_good = 0
    for word in prediction:
        if word in target:
            len_good += 1
            target.remove(word)
    return len_good / len_true if len_true > 0 else 1


def precision(prediction, target):
    target, prediction = normalize_answer(target).split(), normalize_answer(prediction).split()
    len_gen = len(prediction)
    len_good = 0
    for word in target:
        if word in prediction:
            len_good += 1
            prediction.remove(word)
    return len_good / len_gen if len_gen > 0 else 1


def compute_f1(prediction, target):
    prec = precision(prediction, target)
    rec = recall(prediction, target)
    if (prec + rec) == 0.:
        return 0.

    f1 = (2. * prec * rec) / (prec + rec)
    return f1



parser = argparse.ArgumentParser(description="LLM answering with vLLM")
parser.add_argument("--retriever_logfile", type=str, required=True,
                    help="Path to the input JSONL file with extracted chunks")
parser.add_argument("--llm_name", type=str, required=True,
                    help="Path to the model (e.g. ../Qwen/Qwen3-4B)")
parser.add_argument("--output_file_path", type=str, default=None,
                    help="Path to save the output JSON (default: <input_dir>/<input_stem>_eval_llm.json)")
parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens to generate")
parser.add_argument('--gpu_util', type=float, default=0.8, help="Max gpu memory utilization. Default: 0.8")
parser.add_argument('--think', action="store_true", default=False, help='Enable thinking for Qwen3 models.')
parser.add_argument("--max_samples", type=int, default=-1,
                    help="Limit samples for a smoke run; -1 keeps all")
parser.add_argument("--q_icl_prompt", action="store_true",
                    help="Use the shared Q-ICL reader system prompt")
parser.add_argument("--llm_judge", action="store_true",
                    help="Also run the loaded model as an answer judge")
parser.add_argument('--chunk_filter', choices=['none', 'early_stop', 'llm', 'gt', 'no_noise', 'qvalue', 'retrieval_budget'], default='none',
                    help=("Filtering mode for the retrieved chunks. "
                          "Used for debugging and ablation studies of the Answering LLM."))
parser.add_argument('--stopping_threshold', type=float, default=float('-inf'),
                    help="Remove all chunks selected after Q-value drops below this threshold. Works only with chunk_filter='qvalue'.")
parser.add_argument('--max_retrieval_budget', type=int, default=1,
                    help="Maximum number of chunks to use. Works only with chunk_filter='retrieval_budget'.")
args = parser.parse_args()

filter_kwargs = dict()
if args.chunk_filter == 'qvalue':
    filter_kwargs['stopping_threshold'] = args.stopping_threshold
# elif args.chunk_filter == 'retrieval_step':
#     filter_kwargs['max_steps'] = args.max_retrieval_steps
chunk_filter = build_chunk_filter(args.chunk_filter, **filter_kwargs)

file_path = args.retriever_logfile
model_name = args.llm_name
if args.output_file_path:
    output_file_path = args.output_file_path
else:
    base, _ = os.path.splitext(file_path)
    output_file_path = base + "_eval_llm.json"

print(f"Input: {file_path}")
print(f"Output: {output_file_path}")
print(f"Model: {model_name}")
print(f"Chunk filter: {args.chunk_filter}")

dataset = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

if args.max_samples >= 0:
    dataset = dataset[:args.max_samples]
print(f"Samples in dataset: {len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(model=model_name,
          trust_remote_code=True,
          gpu_memory_utilization=args.gpu_util,
          max_model_len=32000,)
print(f"Model {model_name} loaded successfully with vLLM.")


# Prepare prompts for all trajectories of each sample
results = []
all_em_scores = []
all_f1_scores = []
all_prompts = []
all_filtered = []
prompt_metadata = []  # List of tuples (sample_idx, run_idx)

for sample_idx, data in enumerate(tqdm(dataset, desc="Preparing prompts")):
    question = data['question']
    runs = data.get('runs', [])
    if not runs: continue

    for run_idx, run_data in enumerate(runs):
        filtered_chunks = chunk_filter(run_data)
        filter_texts = filtered_chunks["filtered_texts"]
        context = "\n\n---\n\n".join(filter_texts)

        messages = [
            {"role": "system", "content": prompts.sys_qa if args.q_icl_prompt else qa_instruction_prompt},
            {"role": "user", "content": qa_prompt.format(context=context, question=question)}
        ]

        chat_template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if "Qwen3" in model_name:
            chat_template_kwargs['enable_thinking'] = args.think

        text = tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        all_prompts.append(text)
        prompt_metadata.append((sample_idx, run_idx))

print(f"Prepared {len(all_prompts)} prompts for batch processing.")


sampling_params = SamplingParams(
    max_tokens=args.max_tokens,
    temperature=0.0,
)
outputs = llm.generate(all_prompts, sampling_params)


# Group the predictions by samples
predictions_by_sample = defaultdict(list)
for (sample_idx, run_idx), output in zip(prompt_metadata, outputs):
    decoded_output = output.outputs[0].text
    if "Final answer:" in decoded_output:
        llm_prediction = decoded_output.split("Final answer:")[-1].strip()
    else:
        llm_prediction = decoded_output.strip()
    predictions_by_sample[sample_idx].append({
        "run_idx": run_idx,
        "prediction": llm_prediction,
        "full_output": decoded_output
    })


# Select the best candidates using EM and then F1
results = []
all_em_scores = []
all_f1_scores = []

for sample_idx, data in enumerate(tqdm(dataset, desc="Processing results")):
    ground_truth = data['answer']
    candidates = predictions_by_sample.get(sample_idx, [])

    best_candidate = None
    for cand in candidates:
        if compute_exact_match(cand["prediction"], ground_truth) == 1:
            best_candidate = cand
            break

    if best_candidate is None:
        best_f1 = -1.0
        for cand in candidates:
            f1 = compute_f1(cand["prediction"], ground_truth)
            if f1 > best_f1:
                best_f1 = f1
                best_candidate = cand

    final_em = compute_exact_match(best_candidate["prediction"], ground_truth)
    final_f1 = compute_f1(best_candidate["prediction"], ground_truth)
    all_em_scores.append(final_em)
    all_f1_scores.append(final_f1)

    results.append({
        "id": data.get("id"),
        "question": data["question"],
        "ground_truth": ground_truth,
        "model_output": best_candidate["full_output"],
        "prediction": best_candidate["prediction"],
        "best_run_idx": best_candidate["run_idx"],
        "EM": final_em,
        "F1": final_f1,
    })


# Run LLM-as-a-Judge for best candidates
all_judge_scores = []
if args.llm_judge:
    judge_prompts = []

    for result in results:
        messages = [
            {"role": "system", "content": prompts.sys_judge},
            {
                "role": "user",
                "content": (
                    f"QUESTION: {result['question']}\n"
                    f"PREDICTED ANSWER: {result['prediction']}\n"
                    f"GROUNDTRUTH ANSWER: {result['ground_truth']}"
                ),
            },
        ]
        judge_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if "Qwen3" in model_name:
            judge_kwargs["enable_thinking"] = False
        judge_prompts.append(
            tokenizer.apply_chat_template(messages, **judge_kwargs)
        )
    judge_outputs = llm.generate(
        judge_prompts, SamplingParams(max_tokens=500, temperature=0.0)
    )

    for result, output in zip(results, judge_outputs):
        judgment = output.outputs[0].text.upper()
        score = int("CORRECT" in judgment and "INCORRECT" not in judgment)
        result["LLM_Judge_Score"] = score
        all_judge_scores.append(score)


avg_em = sum(all_em_scores) / len(all_em_scores) if all_em_scores else 0
avg_f1 = sum(all_f1_scores) / len(all_f1_scores) if all_f1_scores else 0
avg_judge = (sum(all_judge_scores) / len(all_judge_scores) if all_judge_scores else None)

print("\n" + "-" * 50)
print("             EVAL RESULTS")
print("-" * 50)
print(f"Num samples: {len(results)}")
print(f"Mean Exact Match: {avg_em:.3f}")
print(f"Mean F1-Score: {avg_f1:.3f}")
if avg_judge is not None:
    print(f"Mean LLM-Judge Score: {avg_judge:.3f}")
print("-" * 50)

with open(output_file_path, 'w', encoding='utf-8') as f_out:
    json.dump(results, f_out, indent=4, ensure_ascii=False)
print(f"Evaluation log saved to {output_file_path}")
