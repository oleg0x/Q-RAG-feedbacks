"""MMLU-Pro loader and formatting helper for Q-ICL retrieval."""

from envs.dataloaders.base import RetrievalBase, separator


def format_query(sample):
    options = "".join(
        f"\n{chr(65 + index)}) {option}"
        for index, option in enumerate(sample["options"])
    )
    return "QUESTION: " + sample["question"] + "\nOPTIONS:" + options


class RetrievalMMLUPro(RetrievalBase):
    dataset_name = "MMLU-Pro"
    train_file = "train.jsonl"
    eval_file = "test.jsonl"

    def _format_examples(self):
        self.formatted_examples = [
            format_query(example) + separator + example["answer"]
            for example in self.examples
        ]
