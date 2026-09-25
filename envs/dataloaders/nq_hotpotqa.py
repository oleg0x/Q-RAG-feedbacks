import json
import random
import os
import pandas as pd
import numpy
from torch.utils.data import Dataset



class RetrievalNqHotpotqa(Dataset):

    def __init__(self, path: str, split: str, length: int = -1, seed: int = 5):
        if split not in ['train', 'test']:
            raise ValueError(f'Unknown split for NQ+HotPotQA dataset: {split}')

        super().__init__()
        self.samples = []

        file_path = os.path.join(path, f"{split}.parquet")
        df = pd.read_parquet(file_path, columns=["id", "question", "golden_answers"])
        self.samples = df.to_dict('records')

        for sample in self.samples:
            answers = sample['golden_answers']
            if isinstance(answers, numpy.ndarray):
                sample['golden_answers'] = ', '.join(str(ans) for ans in answers)

        rng = random.Random(seed)
        rng.shuffle(self.samples)

        if length > 0 and length < len(self.samples):
            self.samples = self.samples[:length]

        print(f"NQ+HotPotQA_{split} has been loaded. Samples: {len(self.samples)}")


    def name(self) -> str:
        return "NQ+HotPotQA"


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]



if __name__ == '__main__':

    dataset = RetrievalNqHotpotqa(path="/home/o.inozemcev/Datasets/NQ_Hotpotqa_train",
        split="test", length=10)

    print("Name:", dataset.name())
    print("Length:", dataset.__len__())

    for sample in dataset:
        print(sample)


