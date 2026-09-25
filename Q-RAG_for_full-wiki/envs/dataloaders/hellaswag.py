"""HellaSwag loader and formatting helpers for Q-ICL retrieval."""

from envs.dataloaders.base import RetrievalBase, separator


def format_query(sample):
    endings = "".join(
        f"\n{chr(65 + index)}) {ending}"
        for index, ending in enumerate(sample["endings"])
    )
    return sample["ctx"] + endings


def format_answer(sample):
    return chr(65 + int(sample["label"]))


class RetrievalHellaSwag(RetrievalBase):
    dataset_name = "HellaSwag"
    train_file = "hellaswag_train.jsonl"
    eval_file = "hellaswag_val.jsonl"

    def _format_examples(self):
        self.formatted_examples = [
            format_query(example) + separator + format_answer(example)
            for example in self.examples
        ]
