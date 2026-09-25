import json
import random
import os
import numpy
import pyarrow.parquet as pq
from torch.utils.data import Dataset

# Половина HotpotQA несёт gold-титулы в metadata.supporting_facts, половина NQ
# не несёт ничего (metadata пуст у всех 79 168 строк). Читается вложенная
# проекция, а не колонка целиком: рядом с титулами в metadata лежит весь
# дистракторный контекст — десять абзацев текста на пример, то есть основной
# вес файла в 355 МБ.
TITLE_COLUMN = "metadata.supporting_facts.title"
SENT_ID_COLUMN = "metadata.supporting_facts.sent_id"


def example_key(data_source, sample_id) -> str:
    """Устойчивый ключ примера: ``id`` в этом датасете сам по себе не уникален.

    Обе половины смеси нумеруются с нуля (``train_0``, ``train_1``, …), и
    множество идентификаторов NQ целиком вложено в множество HotpotQA — все
    79 168 совпадают. По ключу адресуются holdout и веса эпизодов, поэтому
    один голый ``id`` склеил бы вопрос NQ с чужим вопросом HotpotQA.
    """
    return f"{data_source}:{sample_id}"


def answer_list(value) -> list:
    """``golden_answers`` в виде списка строк, чем бы parquet его ни отдал."""
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
        # data_source нужен раздельным eval-кривым и файлу весов: смесь
        # обучается одна, а читается по половинам.
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
                # Список, а не строка через ', '. Склейка превращала до 25
                # допустимых ответов NQ в одну недостижимую строку: EM не
                # срабатывал ни на одном варианте, и награда целиком уезжала
                # на судью — перекос, которого у HotpotQA (ровно один ответ)
                # нет.
                "golden_answers": answer_list(columns["golden_answers"][index]),
                "data_source": columns["data_source"][index],
            }
            sample["key"] = example_key(sample["data_source"], sample["id"])
            # supporting_facts кладутся только там, где они есть: у половины
            # NQ их нет вовсе, и пустой список означал бы «gold-титулов у
            # примера ноль», а не «мы их не знаем».
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
        """Оставить только перечисленные примеры (holdout как eval-набор).

        Падение на недостающем ключе намеренное: молча урезанный holdout
        означал бы eval-кривую не по тем вопросам, а заметить это по числам
        нечем.
        """
        with open(ids_file, encoding="utf-8") as source:
            payload = json.load(source)
        wanted = list(payload["ids"] if isinstance(payload, dict) else payload)
        index = {sample['key']: sample for sample in samples}
        missing = [key for key in wanted if key not in index]
        if missing:
            raise ValueError(
                f"{ids_file}: {len(missing)} ключей нет в датасете, "
                f"например {missing[:3]}"
            )
        return [index[key] for key in wanted]


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

