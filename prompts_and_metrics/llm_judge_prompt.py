"""
Shared user prompt for LLM-as-judge (vLLM / OpenAI-compatible chat).

Used by ``scripts/eval_answer_quality.py`` and ``scripts/extract_judge_attention_targets.py``.
Must contain the literal substring ``Proposed answer: `` so attention extraction can locate ``pred``.
"""

JUDGE_USER_PROMPT = (
    "You are evaluating answer correctness.\n\n"
    "Question: {question}\n"
    "Expected answer: {gold}\n"
    "Proposed answer: {pred}\n\n"
    "Does the proposed answer convey the same meaning as the expected answer "
    "in the context of the question? Respond with ONLY 'yes' or 'no'."
)
