"""Evaluate MATH retriever output through an existing OpenAI-compatible vLLM."""

import argparse
import json
import os

from prompts_and_metrics import prompts
from rl.feedback.llm_answer import prepare_examples
from utils.process_latex import extract_boxed_expression, simplify_latex
from vLLM_clients import SyncVllmClient, extract_response_text

MATH_JUDGE_PROMPT = """Determine whether the predicted and reference MATH answers
are mathematically equivalent. Ignore harmless LaTeX formatting differences.
Reply with exactly CORRECT or INCORRECT."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-logfile", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL"))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--skip-judge", action="store_true")
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error("VLLM_BASE_URL/--base-url and VLLM_MODEL/--model are required")

    with open(args.retriever_logfile, encoding="utf-8") as source:
        samples = [json.loads(line) for line in source if line.strip()]
    samples = samples[: args.max_samples]

    results = []
    with SyncVllmClient(
        llm=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=prompts.sys_math,
        thinking=False,
        max_tokens=args.max_tokens,
        temperature=0.0,
    ) as reader:
        for sample in samples:
            response = extract_response_text(
                reader.chat_completion(
                    sample["question"],
                    examples=prepare_examples(sample["pred_texts"]),
                )
            )
            prediction = extract_boxed_expression(response)
            reference = str(sample["answer"])
            results.append(
                {
                    **sample,
                    "prediction": prediction,
                    "full_model_output": response,
                    "EM": int(
                        simplify_latex(prediction) == simplify_latex(reference)
                    ),
                }
            )

    if not args.skip_judge:
        with SyncVllmClient(
            llm=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            system_prompt=MATH_JUDGE_PROMPT,
            thinking=False,
            max_tokens=16,
            temperature=0.0,
        ) as judge:
            for result in results:
                request = (
                    f"QUESTION: {result['question']}\n"
                    f"REFERENCE: {result['answer']}\n"
                    f"PREDICTION: {result['prediction']}"
                )
                judgment = extract_response_text(
                    judge.chat_completion(request)
                ).upper()
                result["LLM_Judge_Score"] = float(judgment.strip() == "CORRECT")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} samples to {args.output_file}")


if __name__ == "__main__":
    main()
