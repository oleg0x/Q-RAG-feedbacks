"""
Dataset adapter for candidate-beta training.
Loads samples from the preprocessed JSONL that includes
candidates, judgements, and betas alongside HotpotQA context.
"""

import json
from torch.utils.data import Dataset

from envs.utils import wiki_style_chunk


class CandidateDataset(Dataset):
    """Loads the preprocessed candidate JSONL directly (no separate HotpotQA loader)."""

    def __init__(self, path: str, split: str = "train", seed: int = 42, **kwargs):
        super().__init__()
        import numpy as np
        np.random.seed(seed)

        self.tasks = []
        with open(path) as f:
            for line in f:
                self.tasks.append(json.loads(line))

        self.tasks = list(np.random.permutation(self.tasks))
        print(f"CandidateDataset loaded {len(self.tasks)} samples from {path}")

    def name(self):
        return "hotpotqa"

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


class CombinedCandidateDataset(CandidateDataset):
    """Candidate JSONL containing compatible HotPotQA and 2Wiki samples."""

    VALID_SOURCES = {"hotpotqa", "2WikiMultihopQA"}

    def __init__(self, path: str, split: str = "train", seed: int = 42, **kwargs):
        super().__init__(path=path, split=split, seed=seed, **kwargs)
        invalid_sources = {
            sample.get("source")
            for sample in self.tasks
            if sample.get("source") not in self.VALID_SOURCES
        }
        if invalid_sources:
            raise ValueError(
                "CombinedCandidateDataset contains invalid or missing sources: "
                f"{sorted(map(str, invalid_sources))}"
            )

    def name(self):
        return "combined"


class CandidateDatasetAdapter(Dataset):
    """
    Adapts CandidateDataset for QAEnv.
    Returns chunks, sf_idx, plus candidates/judgements/betas.
    """

    def __init__(self, dataset: CandidateDataset, min_chunks: int = 6):
        super().__init__()
        self.dataset_name = dataset.name()

        # Filter by min_chunks
        filtered = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if len(sample.get("context", [])) >= min_chunks:
                filtered.append(sample)
        self.dataset = filtered
        print(f"CandidateDatasetAdapter: {len(self.dataset)}/{len(dataset)} samples after min_chunks={min_chunks} filter")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        # Keep the question verbatim (including trailing '?') so that
        # beta-time and reward-time prompts are byte-identical.
        question = sample["question"]

        # Build chunks and sf_idx (same as QADatasetAdapter for hotpotqa)
        sf_idx = []
        chunk_texts = []
        sp_title_set = set()
        for sup in sample["supporting_facts"]:
            sp_title_set.add(sup[0])
        for idx, (title, sentences) in enumerate(sample["context"]):
            if title in sp_title_set:
                sf_idx.append(idx)
            chunk = wiki_style_chunk(title, " ".join(sentences))
            chunk_texts.append(chunk)

        result = {
            "id": sample["_id"],
            "question": question,
            "answer": sample["answer"],
            "chunks": chunk_texts,
            "sf_idx": sf_idx,
            "candidates": sample["candidates"],
            "judgements": sample["judgements"],
            "betas": sample["betas"],
        }
        if "source" in sample:
            result["source"] = sample["source"]
        return result
