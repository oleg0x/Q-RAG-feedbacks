import random
import torch
#from hydra.utils import instantiate
from envs.dataloaders import RetrievalHotPotQA
from envs.dataloaders import RetrievalMusique
from envs.dataloaders import Retrieval2WikiMultihopQA
from envs.dataloaders.babilong import RetrievalBabiLong



class RetrievalCombinedTwo(torch.utils.data.Dataset):

    def __init__(self, shuffle: bool,
                 dataset1: RetrievalHotPotQA,
                 dataset2: RetrievalMusique):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self._name = "combined"
        self.rng = random.Random(42)
        self.shuffle = shuffle
        self.total_length = len(self.dataset1) + len(self.dataset2)
        self.indices = list(range(self.total_length))
        if self.shuffle:
            self.rng.shuffle(self.indices)


    def name(self):
        return self._name


    def __len__(self):
        return self.total_length


    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if actual_idx < len(self.dataset1):
            sample = self.dataset1[actual_idx]
            return {**sample, "source": "hotpotqa"}
        else:
            sample = self.dataset2[actual_idx - len(self.dataset1)]
            return {**sample, "source": "musique"}


    def reshuffle(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


#------------------------------------------------------------------------------


class MinChunksDataset(torch.utils.data.Dataset):
    """Deterministic dataset view filtered by the number of context chunks."""

    def __init__(self, dataset, min_chunks: int, length: int = -1):
        self.dataset = dataset
        self.indices = [
            index
            for index in range(len(dataset))
            if len(dataset[index].get("context", [])) >= min_chunks
        ]
        if length > len(self.indices):
            raise ValueError(
                f"Requested length={length}, but only {len(self.indices)} samples "
                f"have at least {min_chunks} chunks"
            )
        if length > 0:
            self.indices = self.indices[:length]

    def name(self):
        return self.dataset.name()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


class RetrievalCombinedHotpot2Wiki(torch.utils.data.Dataset):
    """Combined HotPotQA + 2Wiki dataset with stable source labels."""

    def __init__(self, shuffle, hotpotqa, twowiki, seed=42):
        self.hotpotqa = hotpotqa
        self.twowiki = twowiki
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.hotpotqa_length = len(hotpotqa)
        self.indices = list(range(self.hotpotqa_length + len(twowiki)))
        if shuffle:
            self.rng.shuffle(self.indices)

    def name(self):
        return "combined"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]
        if actual_index < self.hotpotqa_length:
            return {**self.hotpotqa[actual_index], "source": "hotpotqa"}
        return {
            **self.twowiki[actual_index - self.hotpotqa_length],
            "source": "2WikiMultihopQA",
        }

    def reshuffle(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


#------------------------------------------------------------------------------


class RetrievalCombinedThree(torch.utils.data.Dataset):

    def __init__(self, shuffle: bool,
                 dataset1: RetrievalHotPotQA,
                 dataset2: RetrievalMusique,
                 dataset3: Retrieval2WikiMultihopQA):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.dataset3 = dataset3
        self.rng = random.Random(42)
        self.shuffle = shuffle
        self.len1 = len(self.dataset1)
        self.len2 = len(self.dataset2)
        self.len3 = len(self.dataset3)
        self.total_length = self.len1 + self.len2 + self.len3
        self.indices = list(range(self.total_length))
        if self.shuffle:
            self.rng.shuffle(self.indices)


    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        if actual_idx < self.len1:
            sample = self.dataset1[actual_idx]
            return {**sample, "source": "hotpotqa"}
        elif actual_idx < self.len1 + self.len2:
            sample = self.dataset2[actual_idx - self.len1]
            return {**sample, "source": "musique"}
        else:
            sample = self.dataset3[actual_idx - self.len1 - self.len2]
            return {**sample, "source": "2WikiMultihopQA"}


    def name(self):
        return "combined"


    def __len__(self):
        return self.total_length


    def reshuffle(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
