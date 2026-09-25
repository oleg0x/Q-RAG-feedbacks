import logging
import os
import re
import sys
import math
import numpy as np
from asyncio import Semaphore
from openai import OpenAI, AsyncOpenAI
from typing import Optional, List, Tuple
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import prompts_and_metrics.prompts

logger = logging.getLogger(__name__)

_SECTION_TOKENS = {
    "query_analysis":  (None,                        "<|query_analysis_end|>"),
    "source_analysis": ("<|source_analysis_start|>", "<|source_analysis_end|>"),
    "reasoning":       ("<|reasoning_start|>",       "<|reasoning_end|>"),
    "status":          ("<|status_start|>",          "<|status_end|>"),
    "answer":          ("<|answer_start|>",          "<|answer_end|>"),
}



def parse_response(response: str) -> dict[str, str | None]:
    out = {}
    for name, (start, end) in _SECTION_TOKENS.items():
        start_pat = re.escape(start) if start is not None else r"\A"
        pattern = start_pat + r"(.*?)(?:" + re.escape(end) + r"|\Z)"
        matches = re.findall(pattern, response, re.DOTALL)
        out[name] = matches[-1].strip() if matches else None
    return out



class VllmOcc:
    def __init__(self, llm_path: str, gpu_util: float, max_tokens: int):
        self.llm = LLM(
            model=llm_path,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_util,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

        self.params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            skip_special_tokens=False,
        )
        logger.info(f" VllmOcc has been initialized: model={llm_path}")


    def chat_completion(self, query: str, docs) -> Tuple[str, str]:
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": prompts_and_metrics.prompts.sys_qa_occ},
                {"role": "user", "content": query}
            ],
            documents=docs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        outputs = self.llm.generate([prompt], self.params, use_tqdm=False)
        if not outputs or not outputs[0].outputs:
            logger.error("Model generated no output!")
            return ("", "error")

        response_text = outputs[0].outputs[0].text
        #print(response_text, '\n')
        parsed = parse_response(response_text)
        answer = parsed.get("answer")
        status = parsed.get("status")

        if answer is None:
            logger.warning(" No 'answer' section found in response.")
            answer = ""
        if status is None:
            logger.warning(" No 'status' section found in response.")
            status = "unknown"

        return (answer, status)


    def batch_chat_completion(self, queries: List[str], docs_list: List) -> List[Tuple[str, str]]:
        if len(queries) != len(docs_list):
            raise ValueError(f"Number of queries ({len(queries)}) must match number of document sets ({len(docs_list)})!")

        prompts = []
        for query, docs in zip(queries, docs_list):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": prompts_and_metrics.prompts.sys_qa_occ},
                    {"role": "user", "content": query}
                ],
                documents=docs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)

        outputs = self.llm.generate(prompts, self.params)

        results = []
        for i, output in enumerate(outputs):
            if not output or not output.outputs:
                logger.error(f"Model generated no output for query {i}!")
                results.append(("", "error"))
                continue

            response_text = output.outputs[0].text
            #print(response_text, '\n')
            parsed = parse_response(response_text)
            answer = parsed.get("answer")
            status = parsed.get("status")

            if answer is None:
                logger.warning(f" No 'answer' section found in response for query {i}.")
                answer = ""
            if status is None:
                logger.warning(f" No 'status' section found in response for query {i}.")
                status = "unknown"

            results.append((answer, status))

        return results
