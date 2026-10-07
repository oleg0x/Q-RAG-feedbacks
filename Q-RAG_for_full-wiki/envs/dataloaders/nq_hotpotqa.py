import json
import random
import os
import numpy
import pyarrow.parquet as pq
from torch.utils.data import Dataset

# The HotpotQA half carries gold titles in metadata.supporting_facts, the NQ
# half carries nothing (metadata is empty in all 79,168 rows). A nested
# projection is read instead of the whole column: next to the titles, metadata
# holds the full distractor context, ten paragraphs per example, which is most
# of the 355 MB file.
TITLE_COLUMN = "metadata.supporting_facts.title"
SENT_ID_COLUMN = "metadata.supporting_facts.sent_id"


def example_key(data_source, sample_id) -> str:
    """Stable example key: ``id`` alone is not unique in this dataset.

    Both halves of the mix are numbered from zero (``train_0``, ``train_1``, …),
    and the NQ ids are a subset of the HotpotQA ids: all 79,168 coincide. The
    holdout and episode weights are addressed by this key, so a bare ``id``
    would merge an NQ question with an unrelated HotpotQA question.
    """
    return f"{data_source}:{sample_id}"


def answer_list(value) -> list:
    """``golden_answers`` as a list of strings, whatever type parquet returns."""
    if isinstance(value, (list, tuple, numpy.ndarray)):
        return [str(item) for item in value]
    return [str(value)]


class RetrievalNqHotpotqa(Dataset):

    def __init__(self, path: str, split: str, length: int = -1, seed: int = 5,
                 ids_file: str = None):
        if split not in ['train', 'test']:
            raise ValueError(f'Unknown split for NQ+HotPotQA dataset: {split}')

        super().__init__()
        self.samples = []

        file_path = os.path.join(path, f"{split}.parquet")
        # data_source is needed for per-source eval curves and the weights
        # file: the mix is trained as one dataset but read per half.
        table = pq.read_table(
            file_path,
            columns=[
                "id", "question", "golden_answers", "data_source",
                TITLE_COLUMN, SENT_ID_COLUMN,
            ],
        )
        columns = {name: table.column(name).to_pylist() for name in table.column_names}
        titles = columns.get("title", [None] * table.num_rows)
        sent_ids = columns.get("sent_id", [None] * table.num_rows)

        self.samples = []
        for index in range(table.num_rows):
            sample = {
                "id": columns["id"][index],
                "question": columns["question"][index],
                # A list, not a ', '-joined string. Joining turned up to 25
                # accepted NQ answers into one unreachable string: EM never
                # matched any alias and the whole reward shifted to the judge,
                # a bias HotpotQA (exactly one answer) does not have.
                "golden_answers": answer_list(columns["golden_answers"][index]),
                "data_source": columns["data_source"][index],
            }
            sample["key"] = example_key(sample["data_source"], sample["id"])
            # supporting_facts are stored only where they exist: the NQ half
            # has none, and an empty list would mean "the example has zero
            # gold titles" rather than "they are unknown".
            if titles[index]:
                sample["supporting_facts"] = [
                    [title, sent_id]
                    for title, sent_id in zip(titles[index], sent_ids[index] or [])
                ]
            self.samples.append(sample)

        if ids_file is not None:
            self.samples = self._keep_listed(self.samples, ids_file)

        rng = random.Random(seed)
        rng.shuffle(self.samples)

        if length > 0 and length < len(self.samples):
            self.samples = self.samples[:length]

        print(f"NQ+HotPotQA_{split} has been loaded. Samples: {len(self.samples)}")


    @staticmethod
    def _keep_listed(samples: list, ids_file: str) -> list:
        """Keep only the listed examples (holdout used as an eval set).

        Failing on a missing key is intentional: a silently truncated holdout
        would give an eval curve over the wrong questions, and nothing in the
        numbers would reveal it.
        """
        with open(ids_file, encoding="utf-8") as source:
            payload = json.load(source)
        wanted = list(payload["ids"] if isinstance(payload, dict) else payload)
        index = {sample['key']: sample for sample in samples}
        missing = [key for key in wanted if key not in index]
        if missing:
            raise ValueError(
                f"{ids_file}: {len(missing)} keys are missing from the dataset, "
                f"e.g. {missing[:3]}"
            )
        return [index[key] for key in wanted]


    def name(self) -> str:
        return "NQ+HotPotQA"


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]



if __name__ == '__main__':

    dataset = RetrievalNqHotpotqa(path="/path/to/data/NQ_Hotpotqa_train",
        split="test", length=10)

    print("Name:", dataset.name())
    print("Length:", dataset.__len__())

    for sample in dataset:
        print(sample)

