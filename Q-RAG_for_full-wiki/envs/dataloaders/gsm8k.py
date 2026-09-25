"""GSM8K loader for Q-ICL retrieval."""

import re

from envs.dataloaders.base import RetrievalBase, separator

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_short_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    return match.group(1).replace(",", "").strip() if match else ""


class RetrievalGSM8K(RetrievalBase):
    dataset_name = "gsm8k"
    train_file = "train.jsonl"
    eval_file = "test.jsonl"

    def _format_examples(self):
        self.formatted_examples = []
        for example in self.examples:
            solution, answer = example["answer"].rsplit("####", 1)
            formatted = solution.strip() + "\nFinal Answer: " + answer.strip()
            self.formatted_examples.append(
                example["question"] + separator + formatted
            )
