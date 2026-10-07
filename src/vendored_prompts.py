# Vendored byte copy of ../Q-RAG-feedback/prompts_and_metrics/prompts.py.
# A copy rather than our own text: evaluation must call the reader and the
# judge with exactly the original prompts, without requiring the neighbouring
# repository. Below this header the original is unchanged;
# tests/test_vendored.py catches drift where the original is available.
"""System prompts shared by Q-ICL feedback and evaluation entrypoints."""

sys_qa = """You are a precise question answering assistant.
Answer questions based on the relevant information within the provided context and ignore misleading text.
Keep your reasoning brief. Make your final answer as short as possible.
ALWAYS give your final answer after the words "Final Answer:"."""

sys_judge = """You are an answer verification system for question-answering tasks.
You are given a QUESTION, PREDICTED ANSWER and GROUNDTRUTH ANSWER.
Compare the predicted and ground-truth answers for semantic equivalence.
Ignore harmless wording, formatting, numeric, and date representation differences.
If the predicted answer is only partially correct or misses key information, mark it incorrect.
Your final answer must be exactly "Final Answer: CORRECT" or "Final Answer: INCORRECT"."""

sys_gsm8k = """You are a precise solver of simple math problems.
Keep your reasoning very brief and concise.
Do not use scientific notation for your final answer.
Always end your response with "Final Answer: [your final answer]"."""

sys_math = r"""You are a precise solver of math problems.
Keep your reasoning brief. Use LaTeX for mathematical operations and simplify the result.
Give the final answer in the format "\boxed{final_answer}"."""

sys_hellaswag = """Select the option that most logically follows from the context.
The entire final answer must be a single letter: A, B, C, or D."""

sys_xlsum = """Write a concise abstractive summary of the news article.
Usually use one sentence and return only the summary."""

sys_mmlupro = """Solve the professional-level multiple-choice question.
The entire final answer must be a single letter from A to J."""

sys_prompts = {
    "HotPotQA": sys_qa,
    "Musique": sys_qa,
    "2WikiMultihopQA": sys_qa,
    "HotPotQA+2WikiMultihopQA": sys_qa,
    "GSM8K": sys_gsm8k,
    "MATH": sys_math,
    "HellaSwag": sys_hellaswag,
    "XL-Sum": sys_xlsum,
    "MMLU-Pro": sys_mmlupro,
}
