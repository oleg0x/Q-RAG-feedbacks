"""Shared deterministic JSONL loader for Q-ICL datasets."""

import json
import os
import random

from torch.utils.data import Dataset

separator = "\n #|||# "


class RetrievalBase(Dataset):
    dataset_name = "___"
    train_file = "train.jsonl"
    eval_file = "eval.jsonl"
    supported_splits = {"train", "test", "eval"}

    def __init__(self, path, split, samples_num, examples_num, seed=1):
        if split not in self.supported_splits:
            raise ValueError(f"Unknown split for {self.dataset_name} dataset: {split}")
        if samples_num < 0 or examples_num < 0:
            raise ValueError("samples_num and examples_num must be non-negative")
        super().__init__()
        rng = random.Random(seed)
        train_samples = self._read_jsonl(os.path.join(path, self.train_file))
        if split == "train":
            sample_count = samples_num or (len(train_samples) - examples_num)
            if sample_count + examples_num > len(train_samples):
                raise ValueError(
                    f"Not enough {self.dataset_name} train samples: requested "
                    f"{sample_count} samples + {examples_num} examples, "
                    f"available {len(train_samples)}"
                )
            indices = rng.sample(
                range(len(train_samples)), sample_count + examples_num
            )
            self.samples = [train_samples[i] for i in indices[:sample_count]]
            self.examples = [train_samples[i] for i in indices[sample_count:]]
        else:
            self.samples = self._read_jsonl(os.path.join(path, self.eval_file))
            if samples_num:
                self.samples = self.samples[:samples_num]
            if examples_num > len(train_samples):
                raise ValueError(
                    f"Requested {examples_num} examples, available {len(train_samples)}"
                )
            indices = rng.sample(range(len(train_samples)), examples_num)
            self.examples = [train_samples[i] for i in indices]
        self._format_examples()

    @staticmethod
    def _read_jsonl(path):
        with open(path, "r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    def _format_examples(self):
        self.formatted_examples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @classmethod
    def name(cls):
        return cls.dataset_name

    def get_examples(self):
        return self.formatted_examples
