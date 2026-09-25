"""Вендорные копии обязаны совпадать с read-only оригиналом, пока он рядом.

Копии в ``src/`` делают форк ридера и судьи самодостаточным, но их содержимое
не наше: промпты и клиент принадлежат ``Q-RAG-feedback``. Пока оригинал лежит
рядом — то есть на машине лаборатории, — любой его дрейф обязан валить эти
тесты, а не накапливаться молча. На чужой машине оригинала нет, и тесты
пропускаются: там контракт держит сама копия.

Сравнивается хвост файла: над оригиналом разрешена только шапка-комментарий
с происхождением копии, ниже неё — байт в байт.
"""

from __future__ import annotations

from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parent.parent
FEEDBACK_REPO = LAB.parent / "Q-RAG-feedback"

pytestmark = pytest.mark.skipif(
    not FEEDBACK_REPO.is_dir(),
    reason="../Q-RAG-feedback недоступен — синхронность вендорных копий "
    "проверяется на машине лаборатории",
)

VENDORED = {
    "vendored_prompts.py": FEEDBACK_REPO / "prompts_and_metrics" / "prompts.py",
    "vendored_vllm_client.py": FEEDBACK_REPO / "vLLM_clients" / "sync_vllm_client.py",
}


@pytest.mark.parametrize("copy_name", sorted(VENDORED))
def test_vendored_copy_ends_with_the_original(copy_name: str) -> None:
    vendored = (LAB / "src" / copy_name).read_text(encoding="utf-8")
    original = VENDORED[copy_name].read_text(encoding="utf-8")
    assert vendored.endswith(original), (
        f"{copy_name} разошёлся с оригиналом {VENDORED[copy_name]}: "
        "обновите копию вслед за ним"
    )
