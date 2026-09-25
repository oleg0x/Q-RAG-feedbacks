"""XL-Sum loader for Q-ICL retrieval."""

from envs.dataloaders.base import RetrievalBase, separator


class RetrievalXLSum(RetrievalBase):
    dataset_name = "XL-Sum"
    train_file = "english_train.jsonl"
    eval_file = "english_val.jsonl"

    def _format_examples(self):
        self.formatted_examples = [
            "TEXT:\n"
            + example["text"]
            + separator
            + "SUMMARY:\n"
            + example["summary"]
            for example in self.examples
        ]
