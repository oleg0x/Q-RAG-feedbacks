import asyncio
import json
import re
import time
from tqdm import tqdm
from sync_vllm_client import SyncVllmClient
from vllm_clients import AsyncVllmClient, AsyncBatchedVllmClient

ip_port = "http://10.1.2.27:9314"  # Check 'ip a' on vLLM server or use "http://localhost:9314"



def simple_test():
    client = AsyncVllmClient(
        model="Qwen3-1.7B",
        base_url=ip_port,
        system_prompt = "You are a helpful assistant. Keep your chain-of-thought very brief and concise.",
        max_tokens=400,
        temperature=0.8,
        top_p=0.6,
        top_k=20,
        presence_penalty=0.1,
        frequency_penalty=0.2,
    )

    requests = [
        "Explain quantum entanglement briefly.",
        "Write a haiku about Python.",
        "Explain photosynthesis briefly.",
    ]

    results = asyncio.run(client.generate_batch(requests))

    for i, res in enumerate(results):
        if "error" in res or "choices" not in res:
            error_msg = res.get("error", "Unknown error")
            print(f"Request {i+1} failed: {error_msg}")
            print("Check the IP, port, and whether the server is running.")
        else:
            print(res['choices'][0]['message']['content'] + '\n')



def get_final_answer(answer):
    # Find last occurrence of "</think>" and get everything after it
    idx = answer.rfind("</think>")
    fin_answer = answer[idx+len("</think>"):] if idx != -1 else answer

    #Find last occurrence of "Final Answer:" and get everything after it
    idx = fin_answer.rfind("Final Answer:")
    fin_answer = fin_answer[idx+len("Final Answer:"):] if idx != -1 else fin_answer

    fin_answer = fin_answer.translate(str.maketrans('', '', '.,'))
    return fin_answer.strip()



def get_final_answer_2(answer):
    # Find last occurrence of "</think>" and get everything after it
    idx1 = answer.rfind("</think>")
    fin_answer = answer[idx1+len("</think>"):] if idx1 != -1 else answer

    #Find last occurrence of "Final Answer:" and get everything after it
    idx2 = fin_answer.rfind("Final Answer:")
    fin_answer = fin_answer[idx2+len("Final Answer:"):] if idx2 != -1 else fin_answer

    flag = (idx1 != -1) and (idx2 != -1)
    if flag == True:
        fin_answer = fin_answer.translate(str.maketrans('', '', '.,'))
    return flag, fin_answer.strip()



ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_short_answer(answer) -> str:
    match = ANS_RE.search(answer)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return ""



def parse_boxed(solution: str, sample_id):
    s = r"\boxed{"
    answer = None
    start = solution.rfind(s)
    if start != -1:
        count = 1
        end = start + len(s)
        while end < len(solution) and count > 0:
            if   solution[end] == '{':  count += 1
            elif solution[end] == '}':  count -= 1
            end += 1
        if count == 0:
            answer = solution[start:end]
            answer = re.sub(r'\\[dt]frac', r'\\frac', answer)
        else:
            print(f"Warning: Unbalanced braces in \\boxed{{...}} at sample ID {sample_id}.")
    return answer



testn = 100

sys_prompt_gsm8k = """You are a precise solver of simple math problems.
Strictly follow these instructions:
Keep your reasoning very brief and concise.
Always end your response with "Final Answer: [final answer]".
Your final answer must be a single number (e.g., '100' or '12.5') and nothing more.
Do NOT include in your final answer any explanation, commentary, units, punctuation (like '.', '?', '!'), markdown, or extra text.
"""

sys_prompt_math = """You are a precise solver of math problems.
Strictly follow these instructions:
Keep your reasoning very brief and concise.
Always use LaTeX format for math operations, for exapmle x^2, \\frac{2}{3}, \\sqrt{5}.
Simplify your answer as much as possible.
Always end your response with "\boxed{final_answer}".
Give me just final answer in the format "\boxed{final_answer}" and nothing more.
"""

params_gsm8k = {
    "llm": "Qwen3-1.7B",
    "base_url": ip_port,
    "system_prompt": sys_prompt_gsm8k,
    "api_key": "Q-RAG_ICL_2026",
    "max_tokens": 8000,
    "temperature": 0.0,
}

params_math = {
    "model": "Qwen3-1.7B",
    "base_url": ip_port,
    "system_prompt": sys_prompt_math,
    "max_tokens": 8000,
    "temperature": 0.0,
}



sync_preds = []
async_preds = []



def test_GSM8K_sync():
    print("\n----- Sync test on GSM8K dataset -----")
    with open("/mnt/Datasets/GSM8K/test.jsonl", 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    client = SyncVllmClient(**params_gsm8k)

    start = time.time()
    right = 0
    for i, sample in enumerate(tqdm(samples[:testn], desc="Processing samples")):
        answer = extract_short_answer(sample['answer'])
        prediction = client.chat_completion(sample['question'])
        flag, pred = get_final_answer_2(prediction['choices'][0]['message']['content'])
        sync_preds.append({
            'question': sample['question'],
            'answer': answer,
            'pred': pred,
            'flag': flag,
        })
        if pred == answer:
            right += 1
    elapsed = time.time() - start

    print(f"Number of right answers: {right} out of {testn}")
    print(f"Accuracy: {right/testn*100}%")
    print(f"Elapsed time: {round(elapsed, 2)} sec, Throughput: {round(testn/elapsed, 2)} req/sec, Request time: {round(elapsed/testn, 2)} sec/req.")
# Elapsed time: 364.45 sec, Throughput: 0.55 req/sec, Request time: 1.82 sec.



def test_GSM8K_async():
    print("\n----- Async test on GSM8K dataset -----")
    with open("/mnt/Datasets/GSM8K/test.jsonl", 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    #client = AsyncVllmClient(**params_gsm8k)
    client = AsyncBatchedVllmClient(**params_gsm8k)

    start = time.time()
    print("Processing samples concurrently...")
    requests = [sample['question'] for sample in samples[:testn]]
    results = asyncio.run(client.generate_batch(requests))
    right = 0
    for i, sample in enumerate(samples[:testn]):
        answer = extract_short_answer(sample['answer'])
        res = results[i]
        flag, pred = get_final_answer_2(res['choices'][0]['message']['content'])
        async_preds.append({
            'question': sample['question'],
            'answer': answer,
            'pred': pred,
            'flag': flag,
        })
        if pred == answer:
            right += 1
    elapsed = time.time() - start

    print(f"Number of right answers: {right} out of {testn}")
    print(f"Accuracy: {right/testn*100}%")
    print(f"Elapsed time: {round(elapsed, 2)} sec, Throughput: {round(testn/elapsed, 2)} req/sec, Request time: {round(elapsed/testn, 2)} sec/req.")
# Elapsed time: 15.22 sec, Throughput: 13.14 req/sec, Request time: 0.08 sec.



def test_MATH_async():
    print("\n----- Async test on MATH dataset -----")
    with open("/mnt/Datasets/MATH/test.jsonl", 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f]

    #client = AsyncVllmClient(**params_math)
    client = AsyncBatchedVllmClient(**params_math)

    start = time.time()
    print("Processing samples concurrently...")
    requests = [sample['problem'] for sample in samples[:testn]]
    results = asyncio.run(client.generate_batch(requests))
    right = 0
    for i, sample in enumerate(samples[:testn]):
        answer = sample['answer']
        res = results[i]
        prediction = res['choices'][0]['message']['content']
        pred = parse_boxed(prediction, sample['id'])
        #print("\nID:", sample['id'])
        #print("Prediction:", prediction)
        #print("Pred:", pred)
        #print("Answer:", answer)
        if pred == answer:
            right += 1
    elapsed = time.time() - start

    print(f"Number of right answers: {right} out of {testn}")
    print(f"Accuracy: {right/testn*100}%")
    print(f"Elapsed time: {round(elapsed, 2)} sec, Throughput: {round(testn/elapsed, 2)} req/sec, Request time: {round(elapsed/testn, 2)} sec/req.")



if __name__ == "__main__":
    #simple_test()
    test_GSM8K_sync()
    #test_GSM8K_async()
    #test_MATH_async()

    # for i in range(testn):
    #     if sync_preds[i]['flag'] != async_preds[i]['flag']: #or sync_preds[i]['pred'] != async_preds[i]['pred']:
    #         print(f"----- Sample {i} differs!")
    #         print(f"----- Question: {sync_preds[i]['question']}")
    #         print(f"----- Sync pred: {sync_preds[i]['pred']}")
    #         print(f"----- Async pred: {async_preds[i]['pred']}")
    #         print(f"----- Correct answer: {sync_preds[i]['answer']}\n")

