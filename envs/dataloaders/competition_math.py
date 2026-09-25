"""Competition MATH loader for Q-ICL retrieval."""

from envs.dataloaders.base import RetrievalBase, separator


class RetrievalMATH(RetrievalBase):
    dataset_name = "MATH"
    train_file = "train_ans.jsonl"
    eval_file = "test_ans.jsonl"

    def _format_examples(self):
        self.formatted_examples = [
            example["problem"] + separator + example["solution"]
            for example in self.examples
        ]
