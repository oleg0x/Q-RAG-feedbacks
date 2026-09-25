"""Run remote reader generation and optional LLM-as-judge on retriever JSONL."""

import argparse
import json
import os
import re
import string
from collections import Counter
from tqdm import tqdm

from prompts_and_metrics import prompts
from vLLM_clients import SyncVllmClient, extract_response_text


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-logfile", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--answer-model", default=os.getenv("VLLM_MODEL"))
    parser.add_argument("--judge-model", default="")  # Empty string means no judging
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=1000)
    args = parser.parse_args()
    if not args.base_url or not args.answer_model:
        parser.error("VLLM_BASE_URL/--base-url and VLLM_MODEL/--answer-model are required")

    with open(args.retriever_logfile, encoding="utf-8") as source:
        samples = [json.loads(line) for line in source if line.strip()]
    if 0 < args.max_samples < len(samples):
        samples = samples[:args.max_samples]

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
        for sample in tqdm(samples, desc="Generating answers", unit="sample"):
            context = "\n\n".join(sample["pred_texts"])
            request = (
                f"CONTEXT:\n{context}\n\nQUESTION:\n{sample['question']}\n\n"
                "Final Answer:"
            )
            prediction = final_answer(
                extract_response_text(reader.chat_completion(request))
            )
            results.append(
                {
                    **sample,
                    "prediction": prediction,
                    "EM": exact_match(prediction, sample["answer"]),
                    "F1": f1_score(prediction, sample["answer"]),
                }
            )

    # Judge only if judge_model was explicitly provided
    if args.judge_model:
        with SyncVllmClient(
            llm=args.judge_model or args.answer_model,
            base_url=args.base_url,
            api_key=args.api_key,
            system_prompt=prompts.sys_judge,
            thinking=False,
            max_tokens=800,
            temperature=0.0,
        ) as judge:
            for result in tqdm(results, desc="Judging answers", unit="sample"):
                request = (
                    f"QUESTION: {result['question']}\n"
                    f"PREDICTED ANSWER: {result['prediction']}\n"
                    f"GROUNDTRUTH ANSWER: {result['answer']}"
                )
                judgment = final_answer(
                    extract_response_text(judge.chat_completion(request))
                ).upper()
                result["LLM_Judge_Score"] = float(judgment == "CORRECT")

    total_samples = len(results)
    mean_em = sum(r["EM"] for r in results) / total_samples
    mean_f1 = sum(r["F1"] for r in results) / total_samples

    print(f"{'-'*50}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Mean Exact Match (EM): {mean_em:.4f} ({mean_em*100:.2f}%)")
    print(f"Mean F1-Score: {mean_f1:.4f} ({mean_f1*100:.2f}%)")

    if args.judge_model:
        mean_judge = sum(r["LLM_Judge_Score"] for r in results) / total_samples
        print(f"Mean LLM-as-Judge Score: {mean_judge:.4f} ({mean_judge*100:.2f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
