"""Direct-search environment over all 21M Wiki-18 chunks.

The one difference from ``QAEnv`` changes everything: the action set no longer
comes from the dataset. Step candidates are the result of searching
``s @ M.T`` over the whole corpus, so only ``question``, ``answer`` and
``supporting_facts`` are taken from the ``q0_s1`` files; the
``candidates``/``judgements``/``betas``/``context`` fields belong to the
distractor pools of the earlier setup and are ignored here.

The episode state holds the three things masking depends on: the selected row
IDs, the per-title chunk counter (quota ``N``) and the step number.
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
    """Accepted answers of an example, always as a list.

    ``golden_answers`` comes from the NQ + HotpotQA mix (up to 25 aliases for
    NQ), ``answer`` from the older ``q0_s1`` files. An example with a single
    answer gets a one-element list, so its reward does not change at all.
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
    """Titles of the episode's gold articles in order of first appearance."""
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
    """``q0_s1`` → examples for direct search, without the candidate fields.

    The coverage flag and gold-title ids are computed once here rather than per
    episode: 38.5% of the combined set lacks the full set of gold titles in
    Wiki-18, reward on those examples is nearly unreachable, and the training
    curves must be read separately. The corpus title dictionary is released
    after labelling: it takes hundreds of megabytes and is not needed later.
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
                    # Key, not id: in nq_hotpotqa_train both halves of the mix
                    # are numbered from zero, so a holdout addressed by id
                    # alone would hit two different questions.
                    "key": str(raw.get("key", f"{source}:{sample_id}")),
                    "question": str(raw["question"]),
                    "answer": variants[0],
                    "answer_variants": variants,
                    "supporting_facts": raw.get("supporting_facts", []),
                    "gold_titles": titles,
                    "source": source,
                    # None rather than False when the dataset has no gold
                    # titles at all (the NQ + HotpotQA raw mix carries none):
                    # "not checked" and "not covered" differ, and the latter
                    # would draw a spurious covered_share = 0 curve.
                    "gold_titles_covered": (
                        bool(title_table.all_titles_covered(titles))
                        if title_table is not None and titles
                        else None
                    ),
                    # An empty title list must not build the normalized
                    # index: no raw-mix example has gold titles, and the
                    # 5.2M-entry dict takes hundreds of megabytes.
                    "gold_title_ids": (
                        title_table.title_ids_of(titles)
                        if title_table is not None and titles
                        else []
                    ),
                }
            )
        if title_table is not None:
            title_table.release_normalized_index()
            # The share is over examples with known titles, not all of them:
            # the NQ half of the raw mix has none, and a common denominator
            # would halve the coverage for no reason.
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
    """Episode weights from a JSONL ``{"key": …, "w": …}``, in dataset order.

    A separate file rather than a code flag: weighted variants differ from the
    raw mix only by this file, and with the same seed the example sequence
    stays comparable. The holdout is excluded by the same mechanism, weight 0.

    Every example must have a key: a silently applied default weight would turn
    a forgotten line into a quiet extension of the training set, and nothing in
    the curves would reveal it.
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
            f"{path}: no weight for some of the {len(dataset)} dataset examples, "
            f"e.g. {missing[:3]}"
        )
    if not (weights >= 0).all():
        raise ValueError(f"{path}: negative weights are not allowed")
    if weights.sum() <= 0:
        raise ValueError(f"{path}: total weight is zero, nothing to sample")
    return weights


class DenseSearchEnv:
    """One direct-search episode: state, masks and reward."""

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
                        f"{len(weights)} weights for {len(dataset)} examples"
                    )
                # The cumulative sum is computed once: an episode is drawn on
                # every reset, and np.cumsum over 170k examples at that point
                # would cost more than the search itself.
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
        """Index of the next episode.

        Episodes are drawn by the environment, not by a dataloader, so any
        sampler must live here: a uniform ``integers`` draw ignored the weights.
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
        """``question [SEP] chunk₁ [SEP] …``, the text the state tower encodes."""
        return self.separator.join([self.question, *self.selected_texts])

    @property
    def gold_titles_covered(self) -> bool | None:
        return self.sample.get("gold_titles_covered")

    def blocked_rows(self) -> list[int]:
        """Rows already selected: blocked by the episode itself, not the quota."""
        return list(self.selected_rows)

    def blocked_titles(self) -> list[int]:
        """Titles that have used up the quota ``N``.

        A title is closed when its quota is exhausted, not after its first
        chunk: the gold sentence is often in the second chunk of the article.
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
        """Take a chunk and query the reward. Blocking reader and judge call.

        Called from a thread pool: environments are independent, and the reader
        takes tens of milliseconds per answer, so a sequential pass over the
        batch would be bound by it.
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
            # A vLLM error does not mean "the context is useless": such a
            # transition is marked invalid and excluded from the loss.
            valid=bool(feedback.get("valid", True)),
        )

    def selected_titles(self) -> list[int]:
        return list(self.title_counts.elements())

    def second_chunk_count(self) -> int:
        """Number of selected chunks that are not the first of their title."""
        return sum(max(count - 1, 0) for count in self.title_counts.values())


def make_search_envs(
    template: DenseSearchEnv,
    count: int,
    seed: int | None = None,
) -> list[DenseSearchEnv]:
    """``count`` independent envs with a shared dataset and their own vLLM clients."""
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
