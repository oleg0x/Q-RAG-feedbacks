import json
import random
import os
from torch.utils.data import Dataset



class Retrieval2WikiMultihopQA(Dataset):

    def __init__(self, path: str, split: str, skip_bridge_comparison: bool,
                 length: int = -1, seed: int = 100, clean_filename: str = None):
        super().__init__()

        if split not in ['train', 'dev', 'test', 'train-clean']:
            raise ValueError(f'Unknown split for 2WikiMultihopQA dataset: {split}')

        self.split = split
        self.length = length

        if split == 'train-clean':
            if os.path.isfile(path):
                file_path = path
            else:
                filenames = [
                    clean_filename,
                    "2wiki_candidate_train_q0_s1.jsonl",
                    "train_candidates_q0_s1.jsonl",
                ]
                candidates = [
                    os.path.join(path, name) for name in filenames if name
                ]
                file_path = next(
                    (name for name in candidates if os.path.exists(name)),
                    candidates[0],
                )
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
        else:
            file_path = os.path.join(path, f"{self.split}.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        if skip_bridge_comparison:
            self.samples = [sample for sample in data
                if sample.get('type') != 'bridge_comparison']
        else:
            self.samples = data

        rng = random.Random(seed)
        rng.shuffle(self.samples)

        if self.length > 0 and self.length < len(self.samples):
            self.samples = self.samples[:self.length]

        print(f"2WikiMultihopQA-{self.split} has been loaded. Samples: {len(self.samples)}")


    def name(self) -> str:
        return "2WikiMultihopQA"


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
