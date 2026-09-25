from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import build_index_wiki_qrag as builder


def test_transform_contents_qrag_matches_training_shape() -> None:
    contents = '"All for One Theater"\nFirst sentence. Second sentence.'
    assert (
        builder.transform_contents(contents, "qrag")
        == "All for One Theater First sentence. Second sentence."
    )


def test_transform_contents_qrag_decodes_escaped_title() -> None:
    contents = '"A \\"quoted\\" title"\nPassage'
    assert (
        builder.transform_contents(contents, "qrag")
        == 'A "quoted" title Passage'
    )


def test_transform_contents_raw_is_byte_preserving() -> None:
    contents = '"Title"\n Passage with leading space'
    assert builder.transform_contents(contents, "raw") == contents


def test_extract_action_state_dict_strips_only_expected_prefix() -> None:
    critic = {
        "state_embed.model.weight": torch.tensor([1.0]),
        "action_embed.model.weight": torch.tensor([2.0]),
        "action_embed.model.bias": torch.tensor([3.0]),
    }
    extracted = builder.extract_action_state_dict(critic)
    assert set(extracted) == {"model.weight", "model.bias"}
    assert extracted["model.weight"].item() == 2.0


def test_extract_action_state_dict_rejects_missing_action_tower() -> None:
    with pytest.raises(ValueError, match="No keys"):
        builder.extract_action_state_dict(
            {"state_embed.model.weight": torch.tensor([1.0])}
        )


def test_raw_faiss_index_preserves_shard_vectors_and_norms(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    vectors = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 1.0, 1.0],
            [-4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    shard = builder.save_shard(tmp_path, 0, vectors, "float32")
    index = faiss.IndexFlatIP(3)
    index.add(vectors)

    verification = builder.verify_index(
        index=index,
        shards=[shard],
        expected_rows=4,
        expected_dimension=3,
        sample_size=4,
        seed=42,
    )

    assert verification["max_shard_error"] == 0.0
    assert verification["normalization_expected"] is False
    assert verification["sample_norms"]["max"] == pytest.approx(4.0)


def test_faiss_topk_matches_direct_raw_dot_product() -> None:
    faiss = pytest.importorskip("faiss")
    vectors = np.array(
        [
            [2.0, 0.0],
            [1.0, 1.0],
            [0.0, 3.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    query = np.array([[1.0, 0.25]], dtype=np.float32)
    expected = np.argsort(-(query @ vectors.T), axis=1)[:, :3]

    index = faiss.IndexFlatIP(2)
    index.add(vectors)
    _, actual = index.search(query, 3)

    np.testing.assert_array_equal(actual, expected)


def test_requested_identity_separates_text_formats(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id":"0","contents":"Title text"}\n', encoding="utf-8")
    artifact = {
        "weights": {"sha256": "action-sha"},
        "source": {
            "checkpoint": {"sha256": "checkpoint-sha"},
            "checkpoint_name": "best",
            "qrag_implementation": {"sha256": "implementation-sha"},
        },
        "encoder": {
            "model_name": "Alibaba-NLP/gte-multilingual-base",
            "requested_revision": "main",
            "resolved_revision": "commit",
            "positions_processor": "none",
        },
    }
    common = dict(
        corpus=corpus,
        tar_member=None,
        allow_missing_id=False,
        allow_id_row_mismatch=False,
        max_length=256,
        model_dtype="float16",
        shard_size=100_000,
        cache_dtype="float32",
        batch_size=256,
        multi_process_chunk_size=1_000,
    )
    raw = SimpleNamespace(**common, text_format="raw")
    qrag = SimpleNamespace(**common, text_format="qrag")

    raw_identity = builder.requested_build_identity(raw, artifact)
    qrag_identity = builder.requested_build_identity(qrag, artifact)

    assert raw_identity != qrag_identity
    assert raw_identity["corpus"]["text_format"] == "raw"
    assert qrag_identity["corpus"]["text_format"] == "qrag"
