"""Среда прямого поиска по всем 21 млн чанков Wiki-18.

Отличие от ``QAEnv`` одно, но оно меняет всё: множество действий больше не
приходит из датасета. Кандидаты шага — результат поиска ``s @ M.T`` по всему
корпусу, поэтому из ``q0_s1``-файлов берутся только ``question``, ``answer`` и
``supporting_facts``; поля ``candidates``/``judgements``/``betas``/``context``
относятся к дистракторным пулам прошлой постановки и здесь игнорируются.

Состояние эпизода хранит три вещи, от которых зависит маскирование:
взятые row ID, счётчик чанков на титул (квота ``N``) и номер шага.
"""

from __future__ import annotations

import json
from collections import Counter, namedtuple
from pathlib import Path
from typing import Any, Sequence

from torch.utils.data import Dataset

from envs.title_table import TitleTable


StepResult = namedtuple("StepResult", ["reward", "done", "valid"])


def answer_variants(sample: dict[str, Any]) -> list[str]:
    """Допустимые ответы примера, всегда списком.

    ``golden_answers`` приносит смесь NQ + HotpotQA (до 25 вариантов у NQ),
    ``answer`` — старые ``q0_s1``-файлы. У примера с единственным ответом
    список из одного элемента, и награда на нём не меняется ни на бит.
    """
    variants = sample.get("golden_answers")
    if variants is None:
        variants = sample.get("answer_variants")
    if variants is None:
        return [str(sample.get("answer", "")).strip()]
    if isinstance(variants, str):
        return [variants.strip()]
    return [str(item).strip() for item in variants] or [""]


def gold_titles(sample: dict[str, Any]) -> list[str]:
    """Титулы gold-статей эпизода в порядке первого появления."""
    supporting = sample.get("supporting_facts")
    if not isinstance(supporting, list):
        return []
    titles = []
    for fact in supporting:
        if isinstance(fact, (list, tuple)) and fact:
            title = str(fact[0])
            if title not in titles:
                titles.append(title)
    return titles


class SearchDatasetAdapter(Dataset):
    """``q0_s1`` → примеры для прямого поиска, без кандидатных полей.

    Флаг покрытия и идентификаторы gold-титулов считаются один раз здесь, а не
    на каждом эпизоде: полного комплекта gold-титулов в Wiki-18 нет у 38.5%
    объединённой выборки, награда на этих примерах почти недостижима, и кривые
    обучения обязаны читаться раздельно. Словарь титулов корпуса после
    разметки отпускается — он весит сотни мегабайт и дальше не нужен.
    """

    def __init__(self, dataset, title_table: TitleTable | None = None) -> None:
        super().__init__()
        self.samples: list[dict[str, Any]] = []
        for index in range(len(dataset)):
            raw = dataset[index]
            titles = gold_titles(raw)
            variants = answer_variants(raw)
            sample_id = str(raw.get("_id", raw.get("id", index)))
            source = str(raw.get("data_source", raw.get("source", "unknown")))
            self.samples.append(
                {
                    "id": sample_id,
                    # Ключ, а не id: в nq_hotpotqa_train обе половины смеси
                    # нумеруются с нуля, и по одному id holdout адресовал бы
                    # два разных вопроса.
                    "key": str(raw.get("key", f"{source}:{sample_id}")),
                    "question": str(raw["question"]),
                    "answer": variants[0],
                    "answer_variants": variants,
                    "supporting_facts": raw.get("supporting_facts", []),
                    "gold_titles": titles,
                    "source": source,
                    # None, а не False, когда gold-титулов у датасета нет
                    # вовсе (сырая смесь NQ + HotpotQA их не носит): «не
                    # проверяли» и «не покрыто» — разные вещи, и вторая
                    # нарисовала бы кривую covered_share = 0 на ровном месте.
                    "gold_titles_covered": (
                        bool(title_table.all_titles_covered(titles))
                        if title_table is not None and titles
                        else None
                    ),
                    # Пустой список титулов не должен строить нормализованный
                    # индекс: у сырой смеси gold-титулов нет ни у одного
                    # примера, а словарь 5.2 млн строк весит сотни мегабайт.
                    "gold_title_ids": (
                        title_table.title_ids_of(titles)
                        if title_table is not None and titles
                        else []
                    ),
                }
            )
        if title_table is not None:
            title_table.release_normalized_index()
            # Доля считается от примеров с известными титулами, а не от всех:
            # у половины NQ сырой смеси их нет вовсе, и общий знаменатель
            # занизил бы покрытие вдвое на ровном месте.
            known = [
                sample
                for sample in self.samples
                if sample["gold_titles_covered"] is not None
            ]
            covered = sum(1 for sample in known if sample["gold_titles_covered"])
            unknown = len(self.samples) - len(known)
            print(
                f"SearchDatasetAdapter: {len(self.samples)} samples, "
                f"{covered} of {len(known)} ({covered / max(len(known), 1):.1%}) "
                f"with the full set of gold titles in the corpus, "
                f"{unknown} without gold titles at all"
            )
        else:
            print(f"SearchDatasetAdapter: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def load_weights(path: str | Path, dataset) -> "Any":
    """Веса эпизодов из JSONL ``{"key": …, "w": …}`` в порядке датасета.

    Отдельный файл, а не флаг в коде: руки A2/A3 будут отличаться от сырой
    только им, и при одном сиде последовательность примеров останется
    сопоставимой. Holdout исключается тем же механизмом — весом 0.

    Ключ обязан найтись у каждого примера: молча взятый по умолчанию вес
    превратил бы забытую строку в тихое расширение обучающей выборки, а
    заметить это по кривым нечем.
    """
    import numpy as np

    table: dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            table[str(record["key"])] = float(record["w"])

    weights = np.empty(len(dataset), dtype=np.float64)
    missing = []
    for index in range(len(dataset)):
        key = str(dataset[index]["key"])
        if key not in table:
            missing.append(key)
            if len(missing) > 3:
                break
            continue
        weights[index] = table[key]
    if missing:
        raise ValueError(
            f"{path}: у {len(dataset)} примеров датасета нет веса, "
            f"например {missing[:3]}"
        )
    if not (weights >= 0).all():
        raise ValueError(f"{path}: отрицательные веса запрещены")
    if weights.sum() <= 0:
        raise ValueError(f"{path}: суммарный вес нулевой, тянуть нечего")
    return weights


class DenseSearchEnv:
    """Один эпизод прямого поиска: состояние, маски и награда."""

    def __init__(
        self,
        dataset,
        max_steps: int,
        feedback_model,
        max_chunks_per_title: int = 2,
        separator: str = " [SEP] ",
        seed: int | None = None,
        weights=None,
    ) -> None:
        if max_chunks_per_title < 1:
            raise ValueError(
                f"max_chunks_per_title must be positive: {max_chunks_per_title}"
            )
        self.dataset = dataset
        self.max_steps = max_steps
        self.feedback_model = feedback_model
        self.max_chunks_per_title = max_chunks_per_title
        self.separator = separator
        self.weights = weights
        self._cumulative_weights = None
        self._rng = None
        if dataset is not None:
            import numpy as np

            self._rng = np.random.default_rng(seed)
            if weights is not None:
                if len(weights) != len(dataset):
                    raise ValueError(
                        f"весов {len(weights)}, примеров {len(dataset)}"
                    )
                # Кумулятивная сумма считается один раз: эпизод выбирается на
                # каждом reset, а np.cumsum по 170 тыс. примеров в этот момент
                # стоил бы дороже самого поиска.
                cumulative = np.cumsum(np.asarray(weights, dtype=np.float64))
                self._cumulative_weights = cumulative / cumulative[-1]

        self.sample: dict[str, Any] = {}
        self.selected_rows: list[int] = []
        self.selected_texts: list[str] = []
        self.title_counts: Counter = Counter()
        self.num_steps = 0
        self.episode_reward = 0.0

    def copy(self) -> "DenseSearchEnv":
        return DenseSearchEnv(
            dataset=self.dataset,
            max_steps=self.max_steps,
            feedback_model=self.feedback_model.copy(),
            max_chunks_per_title=self.max_chunks_per_title,
            separator=self.separator,
            weights=self.weights,
        )

    def sample_index(self) -> int:
        """Индекс следующего эпизода.

        Эпизоды тянет среда, а не датлоадер, поэтому любой сэмплер обязан
        жить здесь: равномерный ``integers`` шёл мимо весов.
        """
        if self._cumulative_weights is None:
            return int(self._rng.integers(len(self.dataset)))
        return int(
            self._cumulative_weights.searchsorted(
                self._rng.random(), side="right"
            )
        )

    def reset(self, new_sample: dict[str, Any] | None = None) -> str:
        if new_sample is None:
            if self.dataset is None:
                raise ValueError("Either a dataset or an explicit sample is required")
            new_sample = self.dataset[self.sample_index()]
        self.sample = new_sample
        self.selected_rows = []
        self.selected_texts = []
        self.title_counts = Counter()
        self.num_steps = 0
        self.episode_reward = 0.0
        obs, info = self._observation()
        self.feedback_model.reset(obs, info)
        return self.state_text

    @property
    def question(self) -> str:
        return str(self.sample["question"])

    @property
    def state_text(self) -> str:
        """``вопрос [SEP] чанк₁ [SEP] …`` — то, что кодирует state-башня."""
        return self.separator.join([self.question, *self.selected_texts])

    @property
    def gold_titles_covered(self) -> bool | None:
        return self.sample.get("gold_titles_covered")

    def blocked_rows(self) -> list[int]:
        """Уже взятые строки: их запрещает не квота, а сам эпизод."""
        return list(self.selected_rows)

    def blocked_titles(self) -> list[int]:
        """Титулы, исчерпавшие квоту ``N``.

        Титул закрывается не первым взятым чанком, а исчерпанием квоты:
        gold-предложение регулярно лежит во втором чанке той же статьи.
        """
        return [
            title_id
            for title_id, count in self.title_counts.items()
            if count >= self.max_chunks_per_title
        ]

    def _observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return (
            {
                "question": self.question,
                "sample_id": self.sample.get("id"),
                "pred_idx": list(self.selected_rows),
                "pred_chunks": list(self.selected_texts),
            },
            {
                "answer": self.sample.get("answer"),
                "answer_variants": self.sample.get("answer_variants"),
                "sf_idx": [],
                "sf_chunks": [],
                "gold_titles": self.sample.get("gold_titles", []),
            },
        )

    def step(self, row_id: int, text: str, title_id: int) -> StepResult:
        """Взять чанк, спросить награду. Блокирующий вызов ридера и судьи.

        Вызывается из пула потоков: среды независимы, а ридер отвечает
        десятки миллисекунд, и последовательный обход батча упёрся бы в них.
        """
        row_id = int(row_id)
        if row_id in self.selected_rows:
            raise RuntimeError(f"Row {row_id} was already selected in this episode")
        self.num_steps += 1
        self.selected_rows.append(row_id)
        self.selected_texts.append(text)
        self.title_counts[int(title_id)] += 1

        truncated = self.num_steps >= self.max_steps
        obs, info = self._observation()
        feedback = self.feedback_model.get_feedback(obs, info, truncated)
        reward = float(feedback["reward"])
        self.episode_reward += reward
        return StepResult(
            reward=reward,
            done=bool(feedback["terminated"]) or truncated,
            # Ошибка vLLM — это не «контекст бесполезен»: такой переход
            # помечается невалидным и в лосс не идёт.
            valid=bool(feedback.get("valid", True)),
        )

    def selected_titles(self) -> list[int]:
        return list(self.title_counts.elements())

    def second_chunk_count(self) -> int:
        """Сколько взятых чанков — не первые чанки своего титула."""
        return sum(max(count - 1, 0) for count in self.title_counts.values())


def make_search_envs(
    template: DenseSearchEnv,
    count: int,
    seed: int | None = None,
) -> list[DenseSearchEnv]:
    """``count`` независимых сред с общим датасетом и своими клиентами vLLM."""
    if count < 1:
        raise ValueError(f"envs_parallel must be positive: {count}")
    import numpy as np

    envs = [template] + [template.copy() for _ in range(count - 1)]
    for offset, env in enumerate(envs):
        if env.dataset is not None:
            env._rng = np.random.default_rng(
                None if seed is None else seed + offset
            )
    return envs


def state_texts(envs: Sequence[DenseSearchEnv]) -> list[str]:
    return [env.state_text for env in envs]
