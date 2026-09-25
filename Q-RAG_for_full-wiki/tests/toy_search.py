"""Игрушечная обвязка для юнитов линии A: без GPU, без BERT, без vLLM.

Всё, что здесь есть, повторяет интерфейсы боевых компонентов ровно настолько,
насколько их трогает проверяемый код: токенайзер отдаёт списки id, башня
считает mean pooling по маске (а значит не зависит от размера батча и
паддинга), обратная связь возвращает заранее заданную награду.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from rl.feedback.feedback import AFeedbackModel


class ToyTokenizer:
    """Пословный токенайзер с фиксированным словарём.

    ``stack_memory`` требует от токенайзера ровно этих полей; настоящий GTE
    сюда тащить незачем, а его загрузка сделала бы юниты офлайн-зависимыми.
    """

    def __init__(self, vocab_size: int = 128) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.eos_token_id = 2
        self.padding_side = "right"

    def _token_id(self, word: str) -> int:
        # Стабильный хеш: встроенный hash рандомизирован между процессами.
        digest = sum((index + 1) * ord(char) for index, char in enumerate(word))
        return 3 + digest % (self.vocab_size - 3)

    def __call__(self, texts, truncation=True, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        for text in texts:
            ids = [self.cls_token_id]
            ids.extend(self._token_id(word) for word in text.split())
            ids.append(self.sep_token_id)
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }


class ToyTower(nn.Module):
    """Кодировщик состояния: усреднение эмбеддингов по маске внимания.

    Инвариантность к паддингу здесь принципиальна — на ней стоит проверка
    «батчевый rollout совпадает с пошаговым».
    """

    def __init__(self, vocab_size: int = 128, dim: int = 8, seed: int = 0, scale: float = 1.0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.table = nn.Parameter(
            torch.randn(vocab_size, dim, generator=generator) * scale
        )

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        mask = attention_mask.to(self.table.dtype).unsqueeze(-1)
        embeds = self.table[input_ids.long()] * mask
        return embeds.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class ToyActionTower(nn.Module):
    """Замороженная action-башня: тот же интерфейс, что у ``EmbedderNone``."""

    def __init__(self, tower: ToyTower) -> None:
        super().__init__()
        self.tower = tower

    def forward(self, input_ids, attention_mask, positions=None, *args, **kwargs):
        return {"rope": self.tower(input_ids, attention_mask)}


class ConstantFeedback(AFeedbackModel):
    """Награда по расписанию: ``reward_at_final`` на терминальном шаге."""

    FEEDBACK_MODEL_NAME = "toy-constant"

    def __init__(self, reward_at_final: float = 1.0, fail: bool = False) -> None:
        super().__init__(never_terminate=True)
        self.reward_at_final = reward_at_final
        self.fail = fail
        self.last_transition_valid = True

    def reset(self, obs, info) -> None:
        super().reset(obs, info)
        self.last_transition_valid = True

    def reward(self, obs, info, is_final=False) -> float:
        self.last_transition_valid = not (self.fail and is_final)
        if not is_final or self.fail:
            return 0.0
        return self.reward_at_final

    def copy(self) -> "ConstantFeedback":
        return ConstantFeedback(self.reward_at_final, self.fail)


class ToyAgent:
    """Минимальный агент: три башни, α и размер пула политики."""

    class _Section:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    def __init__(self, state_tower, target_tower, action_tower, alpha=1.0, top_k_actions=4) -> None:
        self.critic = self._Section(state_embed=state_tower, action_embed=action_tower)
        self.v_net_target = self._Section(state_embed=target_tower)
        self.alpha = alpha
        self.top_k_actions = top_k_actions

    class _NullContext:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    def online_models_mode(self, training: bool):
        return self._NullContext()


def write_corpus(path: Path, rows: Sequence[tuple[str, str]]) -> None:
    """Корпус вида ``("Title", "passage")`` в формате Wiki-18."""
    with path.open("w", encoding="utf-8") as destination:
        for row_id, (title, passage) in enumerate(rows):
            contents = f'"{title}"\n{passage}'
            destination.write(
                json.dumps({"id": str(row_id), "contents": contents}, ensure_ascii=False)
                + "\n"
            )


def write_offsets(corpus: Path, path: Path) -> None:
    offsets = [0]
    with corpus.open("rb") as source:
        for line in source:
            offsets.append(offsets[-1] + len(line))
    np.save(path, np.asarray(offsets, dtype="<u8"), allow_pickle=False)
