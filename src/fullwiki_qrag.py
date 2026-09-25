#!/usr/bin/env python3
"""Run a trained Q-RAG policy against the full Wiki-18 action index.

The Q-RAG index contains action-tower vectors. Direct querying with the matching
state tower is a useful ablation because it computes the learned policy score
``Q(state, passage) = state_embed(state) @ action_embed(passage)`` exactly.
For production full-wiki retrieval, use the normalized GTE index to generate
top-k candidates and Q-RAG to rerank their cached action vectors. ``fixed``
keeps the initial candidate pool for every hop and is the default. ``refresh``
reruns GTE with the updated state at every hop and remains an experimental mode.

``--reranker none`` is the first-stage-only baseline: it keeps the GTE ranking
and loads neither the state tower nor the action shards. Without it there is no
way to tell how much of the final answer quality Q-RAG is responsible for. In
``fixed`` mode that baseline is prefix-consistent, so a single ``--steps 6`` run
also yields the 2- and 4-step results by truncation.

``--first-stage-query-length`` caps the first-stage query in tokens. It
defaults to the corpus encoding length recorded in the candidate index
manifest (256), so published runs are unaffected. It matters only in
``refresh``, where the query grows into ``question [SEP] chunk1 [SEP] ...``
and 256 tokens drop most of what the later hops appended.

``--log-candidates`` controls the per-step candidate dump inside
``retrieval_hops``. The default ``full`` reproduces the published runs; ``none``
keeps output files small, which matters because downstream reader/judge scripts
copy every input field into their own output.

The Wiki corpus is a large JSONL file.  A persistent uint64 byte-offset table
is built once so arbitrary FAISS row ids can be resolved without rescanning
the corpus for every question.

Examples:

    # One-time preparation (about 168 MiB for 21,015,324 rows).
    python src/fullwiki_qrag.py prepare \
      --index-dir datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw

    # Recommended GTE top-100 followed by two-step Q-RAG retrieval.
    python src/fullwiki_qrag.py retrieve \
      --index-dir datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw \
      --candidate-index-dir datasets/data_sources/full-wiki/wiki18-gte \
      --candidate-backend torch-cuda \
      --qrag-repo Q-RAG-feedback \
      --device cuda:1 \
      --query "Were Scott Derrickson and Ed Wood of the same nationality?" \
      --top-k 100 \
      --steps 2 \
      --mode fixed

    # HotpotQA full-wiki batch, compatible with eval_llm_openqa.py via
    # pred_idx/pred_texts.
    python src/fullwiki_qrag.py retrieve \
      --index-dir datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw \
      --candidate-index-dir datasets/data_sources/full-wiki/wiki18-gte \
      --candidate-backend torch-cuda \
      --qrag-repo Q-RAG-feedback \
      --device cuda:1 \
      --input datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json \
      --output runs/fullwiki_qrag.jsonl \
      --top-k 100 \
      --steps 2 \
      --mode fixed

``search`` is a different retriever, not a flag of ``retrieve``: the state
tower queries the whole action matrix at every hop, so there is no first
stage and no candidate pool from a dataset. ``--index-dir`` is the normalized
GTE index used as the action matrix ``M``, ``--qrag-repo`` must be
``Q-RAG_for_full-wiki``, and without ``--checkpoint`` the tower is stock GTE,
which is the zero-shot starting point the trained numbers are read against.

    # Zero-shot direct search, six hops, quota N=2.
    python src/fullwiki_qrag.py search \
      --index-dir datasets/data_sources/full-wiki/wiki18-gte \
      --qrag-repo full-wiki/Q-RAG_for_full-wiki \
      --title-table datasets/data_sources/full-wiki/wiki18-gte/corpus-title-ids.npy \
      --device cuda:1 \
      --input datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json \
      --output runs/fullwiki_search_zeroshot.jsonl \
      --top-k 100 --steps 6 --max-chunks-per-title 2
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import html
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator, Sequence
import unicodedata

import numpy as np

from build_index_wiki_gte import (
    atomic_write_json,
    discover_shards,
    read_json,
    require_faiss,
)
from build_index_wiki_qrag import (
    EXPECTED_DIMENSION,
    SHARDS_DIR,
    add_qrag_repo_to_path,
    load_run_config,
    parse_wiki_title,
    rows_from_shards,
)


LOG = logging.getLogger("fullwiki-qrag")

MANIFEST_FILE = "manifest.json"
DEFAULT_OFFSETS_FILE = "corpus-row-offsets.npy"
DEFAULT_OFFSETS_METADATA_FILE = "corpus-row-offsets.json"
STATE_PREFIX = "state_embed."
SUPPORTED_STATE_SOURCES = ("critic", "policy")
SUPPORTED_RERANKERS = ("qrag", "none")
SUPPORTED_CANDIDATE_LOGS = ("full", "topk", "none")
CANDIDATE_LOG_KEYS = (
    "candidate_idx",
    "candidate_scores",
    "first_stage_candidate_idx",
    "first_stage_scores",
)
TOPK_CANDIDATE_LOG = 10
DEFAULT_TITLE_TABLE_FILE = "corpus-title-ids.npy"
DIRECT_SEARCH_RETRIEVER = "qrag-state-direct"


def load_and_validate_manifest(index_dir: Path) -> dict[str, Any]:
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)

    index = manifest.get("index", {})
    encoder = manifest.get("encoder", {})
    corpus = manifest.get("corpus", {})
    problems: list[str] = []
    if index.get("type") != "faiss.IndexFlatIP":
        problems.append(f"index.type={index.get('type')!r}")
    if index.get("metric") != "raw_inner_product":
        problems.append(f"index.metric={index.get('metric')!r}")
    if index.get("normalization") != "none":
        problems.append(f"index.normalization={index.get('normalization')!r}")
    if int(index.get("dimension", -1)) != EXPECTED_DIMENSION:
        problems.append(f"index.dimension={index.get('dimension')!r}")
    if encoder.get("positions_processor") != "none":
        problems.append(
            f"encoder.positions_processor={encoder.get('positions_processor')!r}"
        )
    if corpus.get("faiss_id_mapping") != "zero-based JSONL row number":
        problems.append(
            f"corpus.faiss_id_mapping={corpus.get('faiss_id_mapping')!r}"
        )
    if problems:
        raise ValueError(
            "The index is incompatible with full-wiki Q-RAG retrieval: "
            + ", ".join(problems)
        )
    return manifest


def resolve_corpus_path(
    manifest: dict[str, Any],
    explicit: Path | None,
) -> Path:
    if explicit is not None:
        corpus = explicit.expanduser().resolve()
    else:
        recorded = manifest["corpus"].get("source_path")
        if not recorded:
            raise ValueError("Manifest has no corpus.source_path; pass --corpus")
        corpus = Path(recorded).expanduser().resolve()
    if not corpus.is_file():
        raise FileNotFoundError(f"Wiki corpus not found: {corpus}")
    return corpus


def offsets_metadata_path(offsets_path: Path) -> Path:
    if offsets_path.name == DEFAULT_OFFSETS_FILE:
        return offsets_path.with_name(DEFAULT_OFFSETS_METADATA_FILE)
    return offsets_path.with_suffix(offsets_path.suffix + ".json")


def corpus_stat_identity(corpus: Path) -> dict[str, Any]:
    stat = corpus.stat()
    return {
        "path": str(corpus.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def validate_offsets(
    corpus: Path,
    offsets_path: Path,
    expected_rows: int,
) -> np.ndarray:
    metadata_path = offsets_metadata_path(offsets_path)
    if not offsets_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Corpus offsets are missing: {offsets_path}. "
            "Run the prepare command first or omit --no-auto-prepare."
        )

    metadata = read_json(metadata_path)
    expected_identity = corpus_stat_identity(corpus)
    recorded_identity = metadata.get("corpus", {})
    for key in ("path", "size_bytes", "mtime_ns"):
        if recorded_identity.get(key) != expected_identity[key]:
            raise RuntimeError(
                f"Corpus changed since offsets were built ({key}); "
                f"rebuild {offsets_path}"
            )
    if int(metadata.get("rows", -1)) != expected_rows:
        raise RuntimeError(
            f"Offset metadata has {metadata.get('rows')} rows, "
            f"manifest expects {expected_rows}"
        )

    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    if offsets.dtype != np.dtype("<u8"):
        raise RuntimeError(f"Unexpected offsets dtype: {offsets.dtype}")
    if offsets.shape != (expected_rows + 1,):
        raise RuntimeError(
            f"Unexpected offsets shape: {offsets.shape}; "
            f"expected {(expected_rows + 1,)}"
        )
    if int(offsets[0]) != 0 or int(offsets[-1]) != corpus.stat().st_size:
        raise RuntimeError("Offset boundary values do not match the corpus")
    return offsets


def build_corpus_offsets(
    corpus: Path,
    offsets_path: Path,
    expected_rows: int,
    *,
    progress_every: int = 1_000_000,
) -> np.ndarray:
    """Build an atomic NumPy table containing every JSONL line boundary."""
    try:
        return validate_offsets(corpus, offsets_path, expected_rows)
    except FileNotFoundError:
        pass

    offsets_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = offsets_path.with_name(
        f"{offsets_path.stem}.tmp-{os.getpid()}{offsets_path.suffix}"
    )
    temporary_metadata = offsets_metadata_path(temporary)
    if temporary.exists() or temporary_metadata.exists():
        raise RuntimeError(f"Temporary offsets artifact already exists: {temporary}")

    LOG.info(
        "Building corpus offsets: rows=%d corpus=%s output=%s",
        expected_rows,
        corpus,
        offsets_path,
    )
    offsets = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype="<u8",
        shape=(expected_rows + 1,),
    )
    row = 0
    position = 0
    started = time.monotonic()
    try:
        with corpus.open("rb", buffering=16 * 1024 * 1024) as source:
            while True:
                line = source.readline()
                if not line:
                    break
                if row >= expected_rows:
                    raise RuntimeError(
                        f"Corpus has more than manifest rows={expected_rows}"
                    )
                offsets[row] = position
                position += len(line)
                row += 1
                if progress_every > 0 and row % progress_every == 0:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    LOG.info(
                        "Offset progress: %d/%d rows (%.0f rows/s)",
                        row,
                        expected_rows,
                        row / elapsed,
                    )
        if row != expected_rows:
            raise RuntimeError(
                f"Corpus has {row} rows, manifest expects {expected_rows}"
            )
        offsets[row] = position
        offsets.flush()
        del offsets
        if position != corpus.stat().st_size:
            raise RuntimeError(
                f"Read {position} bytes, corpus size is {corpus.stat().st_size}"
            )

        metadata = {
            "schema_version": 1,
            "corpus": corpus_stat_identity(corpus),
            "rows": expected_rows,
            "dtype": "<u8",
            "shape": [expected_rows + 1],
            "last_offset": position,
        }
        atomic_write_json(temporary_metadata, metadata)
        os.replace(temporary, offsets_path)
        os.replace(temporary_metadata, offsets_metadata_path(offsets_path))
    except BaseException:
        # Keep no valid-looking partial artifact.  The uniquely named temporary
        # file can be inspected or removed after an interrupted build.
        raise

    LOG.info(
        "Corpus offsets complete: rows=%d size=%.1f MiB",
        expected_rows,
        offsets_path.stat().st_size / (1024**2),
    )
    return validate_offsets(corpus, offsets_path, expected_rows)


class WikiCorpus:
    """Random row access for the Wiki-18 JSONL corpus."""

    def __init__(
        self,
        corpus: Path,
        offsets_path: Path,
        expected_rows: int,
        *,
        auto_prepare: bool,
    ) -> None:
        self.path = corpus
        if auto_prepare:
            self.offsets = build_corpus_offsets(
                corpus,
                offsets_path,
                expected_rows,
            )
        else:
            self.offsets = validate_offsets(
                corpus,
                offsets_path,
                expected_rows,
            )
        self.expected_rows = expected_rows

    def read_rows(self, row_ids: Sequence[int]) -> list[dict[str, Any]]:
        if not row_ids:
            return []
        result: list[dict[str, Any]] = []
        with self.path.open("rb", buffering=0) as source:
            for row_id_value in row_ids:
                row_id = int(row_id_value)
                if not 0 <= row_id < self.expected_rows:
                    raise IndexError(f"Corpus row outside range: {row_id}")
                start = int(self.offsets[row_id])
                end = int(self.offsets[row_id + 1])
                source.seek(start)
                raw = source.read(end - start)
                try:
                    item = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"Cannot decode corpus row {row_id} at byte {start}"
                    ) from error
                if str(item.get("id")) != str(row_id):
                    raise RuntimeError(
                        f"Corpus row/id mismatch: row={row_id}, id={item.get('id')!r}"
                    )
                if not isinstance(item.get("contents"), str):
                    raise RuntimeError(f"Corpus row {row_id} has no string contents")
                result.append(item)
        return result


def wiki_title(contents: str) -> str:
    title_line, _, _ = contents.partition("\n")
    return parse_wiki_title(title_line)


def normalize_title(title: str) -> str:
    """Compare titles across the HotpotQA and Wiki-18 Wikipedia dumps.

    HotpotQA ships 2017 titles that still carry HTML entities such as
    ``Procter &amp; Gamble``, while Wiki-18 stores the decoded 2018 form.
    Exact string equality therefore reports retrieval misses that never
    happened; on HotpotQA dev this understates gold-title coverage by
    roughly 4.6 percentage points.
    """
    decoded = html.unescape(title)
    folded = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(folded.replace("_", " ").split())


def trim_hop(hop: dict[str, Any], log_candidates: str) -> dict[str, Any]:
    """Drop or truncate the per-step candidate dumps before serialization."""
    if log_candidates == "full":
        return hop
    if log_candidates == "none":
        return {
            key: value for key, value in hop.items() if key not in CANDIDATE_LOG_KEYS
        }
    if log_candidates != "topk":
        raise ValueError(f"Unsupported candidate log level: {log_candidates}")
    return {
        key: (list(value[:TOPK_CANDIDATE_LOG]) if key in CANDIDATE_LOG_KEYS else value)
        for key, value in hop.items()
    }


def extract_state_dict(
    checkpoint: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    if source not in SUPPORTED_STATE_SOURCES:
        raise ValueError(f"Unsupported state source: {source}")
    if source not in checkpoint:
        raise KeyError(
            f"Checkpoint has no {source!r} section; available={list(checkpoint)}"
        )
    section = checkpoint[source]
    state = {
        key.removeprefix(STATE_PREFIX): value
        for key, value in section.items()
        if key.startswith(STATE_PREFIX)
    }
    if not state:
        raise ValueError(
            f"No keys with prefix {STATE_PREFIX!r} in checkpoint[{source!r}]"
        )
    return state


def state_config_path(manifest: dict[str, Any]) -> Path:
    """Locate the training run config copied next to the action artifact."""
    artifact = manifest["action_artifact"]
    # The manifest stores a relative artifact directory, while its nested
    # metadata records the original run config/checkpoint.
    artifact_directory = Path(manifest["encoder"]["artifact_directory"])
    index_dir = Path(manifest["_index_dir"])
    return index_dir / artifact_directory / artifact["run_config_file"]


def state_settings_from_config(config: Any) -> tuple[int, str, bool]:
    return (
        int(config.max_action_length_in_memory),
        str(config.envs.test_env.get("separator", " [SEP] ")),
        bool(config.envs.get("sort_by_index", False)),
    )


def load_state_settings(manifest: dict[str, Any]) -> tuple[int, str, bool]:
    """Read state-text settings without instantiating the state tower.

    ``--reranker none`` never scores with Q-RAG, so it must not pay for
    loading an 11 GiB checkpoint just to learn the separator.
    """
    return state_settings_from_config(load_run_config(state_config_path(manifest)))


def load_state_encoder(
    manifest: dict[str, Any],
    qrag_repo: Path,
    device: str,
    model_dtype: str,
    state_source: str,
) -> tuple[Any, Any, int, str, bool]:
    """Instantiate only the state tower, not the five-network PQN agent."""
    try:
        import torch
        from hydra.utils import instantiate
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "torch, transformers, and hydra-core are required"
        ) from error

    artifact = manifest["action_artifact"]
    config_path = state_config_path(manifest)
    tokenizer_path = config_path.parent / artifact["tokenizer_directory"]
    checkpoint_path = Path(artifact["source"]["checkpoint"]["path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Source Q-RAG checkpoint not found: {checkpoint_path}"
        )

    add_qrag_repo_to_path(qrag_repo)
    config = load_run_config(config_path)
    resolved_revision = artifact["encoder"].get("resolved_revision")
    if resolved_revision:
        config.algo.model.revision = resolved_revision

    previous_default_device = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        encoder = instantiate(config.algo.pqn.state_embed)
    finally:
        torch.set_default_device(previous_default_device)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    encoder.tokenizer = tokenizer

    LOG.info(
        "Loading %s state tower from memory-mapped checkpoint: %s",
        state_source,
        checkpoint_path,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    state = extract_state_dict(checkpoint, state_source)
    incompatible = encoder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict state-tower load failed: {incompatible}")
    del state
    del checkpoint

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if model_dtype not in dtype_by_name:
        raise ValueError(f"Unsupported model dtype: {model_dtype}")
    if device.startswith("cuda"):
        requested_device = torch.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device!r} was requested, but CUDA is unavailable."
            )
        requested_index = (
            requested_device.index
            if requested_device.index is not None
            else torch.cuda.current_device()
        )
        visible_count = torch.cuda.device_count()
        if requested_index < 0 or requested_index >= visible_count:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
            hint = (
                " When CUDA_VISIBLE_DEVICES selects one physical GPU, that GPU is "
                "renumbered to cuda:0 inside the process."
            )
            raise ValueError(
                f"CUDA device {device!r} is out of range: this process sees "
                f"{visible_count} CUDA device(s); "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}.{hint}"
            )
        torch.cuda.set_device(requested_device)
    encoder.to(device=device, dtype=dtype_by_name[model_dtype])
    encoder.eval()

    return (encoder, tokenizer, *state_settings_from_config(config))


def make_text_memories(
    questions: Sequence[str],
    selected_texts: Sequence[Sequence[str]],
    separator: str,
) -> list[Any]:
    from envs.utils import TextMemory

    memories = []
    for question, selected in zip(questions, selected_texts):
        state_text = separator.join([question, *selected])
        memories.append(
            TextMemory(
                item_ids=[],
                available_ids=set(),
                # stack_memory pads this field even though the state tower
                # does not read it. Keep one harmless sentinel action so the
                # padding helper never receives an empty sequence.
                available_mask=np.ones(1, dtype=bool),
                text=state_text,
                input_ids=None,
                attention_mask=None,
            )
        )
    return memories


def encode_states(
    encoder: Any,
    tokenizer: Any,
    questions: Sequence[str],
    selected_texts: Sequence[Sequence[str]],
    separator: str,
    max_state_segment_length: int,
    device: str,
) -> np.ndarray:
    """Encode states with the exact segment-wise Q-RAG stack_memory path."""
    try:
        import torch
        from envs.utils import stack_memory
    except ImportError as error:
        raise RuntimeError("Q-RAG and torch are required") from error

    memories = make_text_memories(questions, selected_texts, separator)
    batch = stack_memory(
        memories,
        tokenizer,
        max_length=max_state_segment_length,
        device=device,
    )
    with torch.inference_mode():
        embeddings = encoder(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
        )
    result = np.asarray(
        embeddings.detach().float().cpu().numpy(),
        dtype=np.float32,
        order="C",
    )
    if result.shape != (len(questions), EXPECTED_DIMENSION):
        raise RuntimeError(f"Unexpected state embedding shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("State encoder returned NaN or Inf")
    return result


class ActionShards:
    """Exact random access to the action vectors used to build FAISS."""

    def __init__(self, index_dir: Path, manifest: dict[str, Any]) -> None:
        cache_dtype = str(manifest["embedding_cache"]["cache_dtype"])
        shard_dir = index_dir / manifest["embedding_cache"].get(
            "directory",
            SHARDS_DIR,
        )
        self.shards = discover_shards(shard_dir, cache_dtype)
        if not self.shards:
            raise FileNotFoundError(f"No embedding shards found in {shard_dir}")
        if self.shards[-1].end != int(manifest["index"]["ntotal"]):
            raise RuntimeError(
                f"Embedding shards end at {self.shards[-1].end}, "
                f"index has {manifest['index']['ntotal']} rows"
            )

    def rows(self, row_ids: Sequence[int]) -> np.ndarray:
        return rows_from_shards(self.shards, [int(row_id) for row_id in row_ids])


def resolve_query_length(corpus_max_length: int, requested: int | None) -> int:
    """Token budget for the first-stage query.

    The corpus was encoded once with the manifest's ``max_length`` and those
    embeddings are on disk, so the document side is fixed. The query is
    encoded per call, and in ``refresh`` it grows into ``question [SEP]
    chunk1 [SEP] ...``; at 256 tokens the sixth hop keeps barely a quarter of
    what it appended. Raising only the query side is safe here: GTE uses rope
    positions with ``max_position_embeddings=8192``, so both sides stay in the
    same space.
    """
    length = corpus_max_length if requested is None else int(requested)
    if length <= 0:
        raise ValueError(f"First-stage query length must be positive: {length}")
    if length != corpus_max_length:
        LOG.info(
            "First-stage query length %d differs from the corpus encoding "
            "length %d; documents keep their cached embeddings",
            length,
            corpus_max_length,
        )
    return length


class GteFirstStage:
    """Normalized GTE candidate generator over the same Wiki-18 row ids."""

    def __init__(
        self,
        index_dir: Path,
        device: str,
        faiss_threads: int,
        trust_remote_code: bool,
        backend: str,
        query_length: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("sentence-transformers is required for GTE candidates") from error

        self.index_dir = index_dir.expanduser().resolve()
        if backend not in {"faiss-cpu", "torch-cuda"}:
            raise ValueError(f"Unsupported candidate backend: {backend}")
        self.backend = backend
        self.device = device
        manifest_path = self.index_dir / MANIFEST_FILE
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Candidate index manifest not found: {manifest_path}")
        self.manifest = read_json(manifest_path)
        index_metadata = self.manifest.get("index", {})
        encoder_metadata = self.manifest.get("encoder", {})
        if index_metadata.get("type") != "faiss.IndexFlatIP":
            raise ValueError(f"Unsupported candidate index: {index_metadata.get('type')!r}")
        if index_metadata.get("metric") != "inner_product_on_l2_normalized_vectors":
            raise ValueError(f"Candidate index is not normalized GTE: {index_metadata.get('metric')!r}")
        if int(index_metadata.get("dimension", -1)) != EXPECTED_DIMENSION:
            raise ValueError(f"Candidate dimension={index_metadata.get('dimension')!r}")

        model_name = str(encoder_metadata["model"])
        revision = str(
            encoder_metadata.get("resolved_revision")
            or encoder_metadata.get("requested_revision", "main")
        )
        LOG.info(
            "Loading pinned local GTE candidate encoder: %s revision=%s",
            model_name,
            revision,
        )
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        self.corpus_max_length = int(encoder_metadata["max_length"])
        self.query_length = resolve_query_length(self.corpus_max_length, query_length)
        self.model.max_seq_length = self.query_length
        model_dtype = str(encoder_metadata.get("model_dtype", "float32"))
        if model_dtype == "float16":
            self.model.half()
        elif model_dtype == "bfloat16":
            self.model.bfloat16()
        elif model_dtype != "float32":
            raise ValueError(f"Unsupported GTE model dtype: {model_dtype}")

        index_path = self.index_dir / index_metadata["file"]
        if not index_path.is_file():
            raise FileNotFoundError(f"Candidate FAISS index not found: {index_path}")
        self.ntotal = int(index_metadata["ntotal"])
        self.index = None
        self.torch_vectors = None
        if backend == "faiss-cpu":
            faiss = require_faiss()
            if faiss_threads > 0:
                faiss.omp_set_num_threads(faiss_threads)
            LOG.info(
                "Loading GTE candidate index (%.1f GiB): %s",
                index_path.stat().st_size / (1024**3),
                index_path,
            )
            self.index = faiss.read_index(str(index_path))
            if int(self.index.ntotal) != self.ntotal:
                raise RuntimeError("Candidate FAISS ntotal differs from its manifest")
        else:
            if not device.startswith("cuda"):
                raise ValueError("torch-cuda candidate backend requires a CUDA device")
            import torch
            shard_dir = self.index_dir / self.manifest["embedding_cache"]["directory"]
            shards = discover_shards(
                shard_dir,
                str(self.manifest["embedding_cache"]["cache_dtype"]),
            )
            if not shards or shards[-1].end != self.ntotal:
                raise RuntimeError("Candidate embedding shards are incomplete")
            LOG.info(
                "Loading exact float32 GTE matrix on %s: rows=%d (%.1f GiB)",
                device,
                self.ntotal,
                self.ntotal * EXPECTED_DIMENSION * 4 / (1024**3),
            )
            self.torch_vectors = torch.empty(
                (self.ntotal, EXPECTED_DIMENSION),
                dtype=torch.float32,
                device=device,
            )
            for shard_number, shard in enumerate(shards, start=1):
                cached = np.load(shard.path, mmap_mode="r", allow_pickle=False)
                staging = torch.from_numpy(
                    np.array(cached, dtype=np.float32, order="C", copy=True)
                )
                self.torch_vectors[shard.start:shard.end].copy_(staging)
                if shard_number % 25 == 0 or shard_number == len(shards):
                    LOG.info("GPU candidate matrix: %d/%d shards", shard_number, len(shards))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().float().cpu().numpy()
        result = np.asarray(embeddings, dtype=np.float32, order="C")
        norms = np.linalg.norm(result, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).tiny):
            raise RuntimeError("GTE returned invalid query embeddings")
        result /= norms[:, None]
        return result

    def search(self, texts: Sequence[str], top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = self.encode(texts)
        k = min(top_k, self.ntotal)
        if self.backend == "faiss-cpu":
            assert self.index is not None
            scores, row_ids = self.index.search(queries, k)
        else:
            import torch
            assert self.torch_vectors is not None
            with torch.inference_mode():
                query_tensor = torch.from_numpy(queries).to(self.device)
                score_tensor, id_tensor = torch.topk(
                    query_tensor @ self.torch_vectors.T,
                    k=k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
            scores = score_tensor.cpu().numpy()
            row_ids = id_tensor.cpu().numpy()
        if np.any(row_ids < 0):
            raise RuntimeError("Candidate search returned missing row ids")
        return scores, row_ids




class FullWikiQrag:
    """Direct action-index Q-RAG; mainly useful as a diagnostic ablation."""

    def __init__(
        self,
        index_dir: Path,
        manifest: dict[str, Any],
        corpus: WikiCorpus,
        state_encoder: Any,
        tokenizer: Any,
        max_state_segment_length: int,
        separator: str,
        sort_by_index: bool,
        device: str,
        faiss_threads: int,
        load_action_index: bool = True,
        log_candidates: str = "full",
    ) -> None:
        if log_candidates not in SUPPORTED_CANDIDATE_LOGS:
            raise ValueError(f"Unsupported candidate log level: {log_candidates}")
        self.index_dir = index_dir
        self.manifest = manifest
        self.corpus = corpus
        self.state_encoder = state_encoder
        self.tokenizer = tokenizer
        self.max_state_segment_length = max_state_segment_length
        self.separator = separator
        self.sort_by_index = sort_by_index
        self.device = device
        self.log_candidates = log_candidates
        self.reranker = "qrag"
        self._action_shards: ActionShards | None = None
        self.index = None
        if sort_by_index:
            LOG.warning(
                "Training used sort_by_index=true; full-wiki states use retrieval order"
            )
        if not load_action_index:
            return
        faiss = require_faiss()
        if faiss_threads > 0:
            faiss.omp_set_num_threads(faiss_threads)
        index_path = index_dir / manifest["index"]["file"]
        if not index_path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        LOG.info(
            "Loading exact Q-RAG action index (%.1f GiB): %s",
            index_path.stat().st_size / (1024**3),
            index_path,
        )
        self.index = faiss.read_index(str(index_path))
        if int(self.index.ntotal) != int(manifest["index"]["ntotal"]):
            raise RuntimeError("Q-RAG FAISS ntotal differs from manifest")

    @property
    def action_shards(self) -> ActionShards:
        if self._action_shards is None:
            self._action_shards = ActionShards(self.index_dir, self.manifest)
        return self._action_shards

    def encode(
        self,
        questions: Sequence[str],
        selected_texts: Sequence[Sequence[str]],
    ) -> np.ndarray:
        return encode_states(
            self.state_encoder,
            self.tokenizer,
            questions,
            selected_texts,
            self.separator,
            self.max_state_segment_length,
            self.device,
        )

    def search(
        self,
        state_vectors: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Direct Q-RAG action index is not loaded")
        scores, row_ids = self.index.search(
            np.asarray(state_vectors, dtype=np.float32, order="C"),
            min(top_k, int(self.index.ntotal)),
        )
        if np.any(row_ids < 0):
            raise RuntimeError("Q-RAG FAISS returned missing row ids")
        return scores, row_ids

    def retrieve_refresh(
        self,
        questions: Sequence[str],
        top_k: int,
        steps: int,
    ) -> list[dict[str, Any]]:
        selected_ids: list[list[int]] = [[] for _ in questions]
        selected_texts: list[list[str]] = [[] for _ in questions]
        selected_scores: list[list[float]] = [[] for _ in questions]
        hops: list[list[dict[str, Any]]] = [[] for _ in questions]
        for step in range(steps):
            scores, candidate_ids = self.search(
                self.encode(questions, selected_texts),
                top_k,
            )
            chosen_ids: list[int] = []
            chosen_scores: list[float] = []
            for row in range(len(questions)):
                already = set(selected_ids[row])
                position = next(
                    (
                        pos
                        for pos, value in enumerate(candidate_ids[row])
                        if int(value) not in already
                    ),
                    None,
                )
                if position is None:
                    raise RuntimeError(f"No unseen direct candidate at step {step}")
                chosen_ids.append(int(candidate_ids[row, position]))
                chosen_scores.append(float(scores[row, position]))
            chosen_rows = self.corpus.read_rows(chosen_ids)
            for row, item in enumerate(chosen_rows):
                selected_ids[row].append(chosen_ids[row])
                selected_texts[row].append(item["contents"])
                selected_scores[row].append(chosen_scores[row])
                hops[row].append(
                    trim_hop(
                        {
                            "step": step,
                            "candidate_idx": [
                                int(value) for value in candidate_ids[row]
                            ],
                            "candidate_scores": [
                                float(value) for value in scores[row]
                            ],
                            "selected_idx": chosen_ids[row],
                            "selected_score": chosen_scores[row],
                        },
                        self.log_candidates,
                    )
                )
        return [
            {
                "pred_idx": selected_ids[row],
                "pred_texts": selected_texts[row],
                "q_values": selected_scores[row],
                "retrieval_hops": hops[row],
                "candidate_retriever": "qrag-action-direct",
                "reranker": self.reranker,
            }
            for row in range(len(questions))
        ]

    def retrieve_fixed(
        self,
        questions: Sequence[str],
        top_k: int,
        steps: int,
    ) -> list[dict[str, Any]]:
        initial_states = self.encode(questions, [[] for _ in questions])
        _, candidate_ids = self.search(initial_states, top_k)
        flat_ids = [int(value) for value in candidate_ids.reshape(-1)]
        candidate_vectors = self.action_shards.rows(flat_ids).reshape(
            len(questions), candidate_ids.shape[1], EXPECTED_DIMENSION
        )
        selected_positions: list[list[int]] = [[] for _ in questions]
        selected_ids: list[list[int]] = [[] for _ in questions]
        selected_texts: list[list[str]] = [[] for _ in questions]
        selected_scores: list[list[float]] = [[] for _ in questions]
        hops: list[list[dict[str, Any]]] = [[] for _ in questions]
        for step in range(steps):
            states = self.encode(questions, selected_texts)
            scores = np.einsum("bd,bkd->bk", states, candidate_vectors, optimize=True)
            for row, positions in enumerate(selected_positions):
                if positions:
                    scores[row, positions] = -np.inf
            chosen_positions = np.argmax(scores, axis=1)
            chosen_ids = [
                int(candidate_ids[row, chosen_positions[row]])
                for row in range(len(questions))
            ]
            chosen_rows = self.corpus.read_rows(chosen_ids)
            for row, item in enumerate(chosen_rows):
                position = int(chosen_positions[row])
                selected_positions[row].append(position)
                selected_ids[row].append(chosen_ids[row])
                selected_texts[row].append(item["contents"])
                selected_scores[row].append(float(scores[row, position]))
                # Key order matches the published runs so that a full-log
                # rerun stays byte-identical.
                hop: dict[str, Any] = {"step": step}
                if self.log_candidates != "none":
                    order = np.argsort(-scores[row], stable=True)
                    finite = [
                        int(value)
                        for value in order
                        if np.isfinite(scores[row, value])
                    ]
                    hop["candidate_idx"] = [
                        int(candidate_ids[row, value]) for value in finite
                    ]
                    hop["candidate_scores"] = [
                        float(scores[row, value]) for value in finite
                    ]
                hop["selected_idx"] = chosen_ids[row]
                hop["selected_score"] = float(scores[row, position])
                hops[row].append(trim_hop(hop, self.log_candidates))
        return [
            {
                "pred_idx": selected_ids[row],
                "pred_texts": selected_texts[row],
                "q_values": selected_scores[row],
                "retrieval_hops": hops[row],
                "candidate_retriever": "qrag-action-direct",
                "reranker": self.reranker,
            }
            for row in range(len(questions))
        ]

    def retrieve(
        self,
        questions: Sequence[str],
        top_k: int,
        steps: int,
        mode: str,
    ) -> list[dict[str, Any]]:
        if mode == "refresh":
            return self.retrieve_refresh(questions, top_k, steps)
        if mode == "fixed":
            return self.retrieve_fixed(questions, top_k, steps)
        raise ValueError(f"Unknown retrieval mode: {mode}")


class TwoStageFullWikiQrag(FullWikiQrag):
    """GTE top-k candidate generation followed by trained Q-RAG scoring."""

    def __init__(
        self,
        index_dir: Path,
        manifest: dict[str, Any],
        corpus: WikiCorpus,
        state_encoder: Any,
        tokenizer: Any,
        max_state_segment_length: int,
        separator: str,
        sort_by_index: bool,
        device: str,
        faiss_threads: int,
        candidate_index_dir: Path,
        candidate_backend: str,
        trust_remote_code: bool,
        max_chunks_per_title: int | None,
        reranker: str = "qrag",
        log_candidates: str = "full",
        first_stage_query_length: int | None = None,
    ) -> None:
        super().__init__(
            index_dir,
            manifest,
            corpus,
            state_encoder,
            tokenizer,
            max_state_segment_length,
            separator,
            sort_by_index,
            device,
            faiss_threads,
            load_action_index=False,
            log_candidates=log_candidates,
        )
        if reranker not in SUPPORTED_RERANKERS:
            raise ValueError(f"Unsupported reranker: {reranker}")
        self.reranker = reranker
        self.first_stage = GteFirstStage(
            candidate_index_dir,
            device,
            faiss_threads,
            trust_remote_code,
            candidate_backend,
            first_stage_query_length,
        )
        if max_chunks_per_title is not None and max_chunks_per_title < 1:
            raise ValueError(
                f"max_chunks_per_title must be positive: {max_chunks_per_title}"
            )
        self.max_chunks_per_title = max_chunks_per_title
        qrag_corpus = manifest["corpus"]
        candidate_corpus = self.first_stage.manifest["corpus"]
        for key in ("rows", "sha256", "faiss_id_mapping"):
            if candidate_corpus.get(key) != qrag_corpus.get(key):
                raise RuntimeError(
                    f"Candidate/Q-RAG corpus mismatch for {key}: "
                    f"{candidate_corpus.get(key)!r} != {qrag_corpus.get(key)!r}"
                )

    def _state_texts(
        self,
        questions: Sequence[str],
        selected_texts: Sequence[Sequence[str]],
    ) -> list[str]:
        return [
            self.separator.join([question, *selected])
            for question, selected in zip(questions, selected_texts)
        ]

    def _retrieve_two_stage(
        self,
        questions: Sequence[str],
        top_k: int,
        steps: int,
        refresh: bool,
    ) -> list[dict[str, Any]]:
        batch_size = len(questions)
        selected_ids: list[list[int]] = [[] for _ in questions]
        selected_texts: list[list[str]] = [[] for _ in questions]
        selected_scores: list[list[float]] = [[] for _ in questions]
        hops: list[list[dict[str, Any]]] = [[] for _ in questions]
        fixed_ids = fixed_scores = fixed_vectors = fixed_titles = None

        for step in range(steps):
            if refresh or fixed_ids is None:
                first_scores, candidate_ids = self.first_stage.search(
                    self._state_texts(questions, selected_texts), top_k
                )
                flat_ids = [int(value) for value in candidate_ids.reshape(-1)]
                if self.reranker == "none":
                    # No Q-RAG scoring, so the action shards stay untouched.
                    candidate_vectors = None
                else:
                    candidate_vectors = self.action_shards.rows(flat_ids).reshape(
                        batch_size, candidate_ids.shape[1], EXPECTED_DIMENSION
                    )
                candidate_rows = self.corpus.read_rows(flat_ids)
                candidate_titles = np.asarray(
                    [wiki_title(item["contents"]) for item in candidate_rows],
                    dtype=object,
                ).reshape(batch_size, candidate_ids.shape[1])
                if not refresh:
                    fixed_ids = candidate_ids
                    fixed_scores = first_scores
                    fixed_vectors = candidate_vectors
                    fixed_titles = candidate_titles
            else:
                candidate_ids = fixed_ids
                first_scores = fixed_scores
                candidate_vectors = fixed_vectors
                candidate_titles = fixed_titles
            assert candidate_ids is not None
            assert first_scores is not None
            assert candidate_titles is not None

            if self.reranker == "none":
                # First-stage order is already descending, so argmax over a
                # copy of the GTE scores walks the pool top-down while the
                # exclusions below still apply. The copy matters: mutating
                # first_scores would corrupt the logged first-stage dump and
                # leak -inf across steps in fixed mode.
                qrag_scores = np.array(first_scores, dtype=np.float32, copy=True)
            else:
                assert candidate_vectors is not None
                states = self.encode(questions, selected_texts)
                qrag_scores = np.einsum(
                    "bd,bkd->bk", states, candidate_vectors, optimize=True
                )
            for row, already in enumerate(selected_ids):
                if already:
                    qrag_scores[row, np.isin(candidate_ids[row], already)] = -np.inf
            if self.max_chunks_per_title is not None:
                # Титул закрывается не первым взятым чанком, а исчерпанием
                # квоты: gold-предложение регулярно лежит во втором чанке той
                # же статьи, и жёсткий запрет делает его недостижимым. При
                # квоте 1 условие вырождается в прежнее «титул уже взят».
                for row, texts in enumerate(selected_texts):
                    if not texts:
                        continue
                    taken = Counter(wiki_title(text) for text in texts)
                    exhausted = [
                        title
                        for title, count in taken.items()
                        if count >= self.max_chunks_per_title
                    ]
                    if exhausted:
                        qrag_scores[
                            row, np.isin(candidate_titles[row], exhausted)
                        ] = -np.inf

            chosen_positions = np.argmax(qrag_scores, axis=1)
            chosen_ids = [
                int(candidate_ids[row, chosen_positions[row]])
                for row in range(batch_size)
            ]
            chosen_rows = self.corpus.read_rows(chosen_ids)
            for row, item in enumerate(chosen_rows):
                position = int(chosen_positions[row])
                chosen_score = float(qrag_scores[row, position])
                if not np.isfinite(chosen_score):
                    raise RuntimeError(f"No unseen Q-RAG candidate at step {step}")
                selected_ids[row].append(chosen_ids[row])
                selected_texts[row].append(item["contents"])
                selected_scores[row].append(chosen_score)
                # Key order matches the published runs so that a full-log
                # rerun stays byte-identical.
                hop: dict[str, Any] = {"step": step}
                if self.log_candidates != "none":
                    order = np.argsort(-qrag_scores[row], stable=True)
                    finite = [
                        int(value)
                        for value in order
                        if np.isfinite(qrag_scores[row, value])
                    ]
                    hop["candidate_idx"] = [
                        int(candidate_ids[row, value]) for value in finite
                    ]
                    hop["candidate_scores"] = [
                        float(qrag_scores[row, value]) for value in finite
                    ]
                    hop["first_stage_candidate_idx"] = [
                        int(value) for value in candidate_ids[row]
                    ]
                    hop["first_stage_scores"] = [
                        float(value) for value in first_scores[row]
                    ]
                hop["selected_idx"] = chosen_ids[row]
                hop["selected_score"] = chosen_score
                hops[row].append(trim_hop(hop, self.log_candidates))
        return [
            {
                "pred_idx": selected_ids[row],
                "pred_texts": selected_texts[row],
                "q_values": selected_scores[row],
                "retrieval_hops": hops[row],
                "candidate_retriever": "gte-multilingual-base",
                "reranker": self.reranker,
            }
            for row in range(batch_size)
        ]

    def retrieve_refresh(
        self, questions: Sequence[str], top_k: int, steps: int
    ) -> list[dict[str, Any]]:
        return self._retrieve_two_stage(questions, top_k, steps, refresh=True)

    def retrieve_fixed(
        self, questions: Sequence[str], top_k: int, steps: int
    ) -> list[dict[str, Any]]:
        return self._retrieve_two_stage(questions, top_k, steps, refresh=False)


def load_and_validate_matrix_manifest(index_dir: Path) -> dict[str, Any]:
    """Проверить, что индекс годится матрицей действий для прямого поиска.

    Это ``wiki18-gte``, а не Q-RAG-индекс: action-башня заморожена и побитово
    равна стоковой GTE, поэтому векторы действий берутся прямо из его шардов.
    Проверки поэтому другие, чем у :func:`load_and_validate_manifest`, — там
    ждут ненормированный Q-RAG-индекс.
    """
    manifest_path = index_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Index manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)

    index = manifest.get("index", {})
    corpus = manifest.get("corpus", {})
    problems: list[str] = []
    if index.get("metric") != "inner_product_on_l2_normalized_vectors":
        problems.append(f"index.metric={index.get('metric')!r}")
    if int(index.get("dimension", -1)) != EXPECTED_DIMENSION:
        problems.append(f"index.dimension={index.get('dimension')!r}")
    if corpus.get("faiss_id_mapping") != "zero-based JSONL row number":
        problems.append(f"corpus.faiss_id_mapping={corpus.get('faiss_id_mapping')!r}")
    if problems:
        raise ValueError(
            "The index cannot serve as a direct-search action matrix: "
            + ", ".join(problems)
        )
    return manifest


def load_action_matrix(index_dir: Path, manifest: dict[str, Any], device: str) -> Any:
    """Загрузить ``M`` на устройство из шардов индекса (21 015 324 × 768, 60 ГиБ)."""
    import torch

    shard_dir = index_dir / manifest["embedding_cache"]["directory"]
    shards = discover_shards(shard_dir, str(manifest["embedding_cache"]["cache_dtype"]))
    rows = int(manifest["index"]["ntotal"])
    if not shards or shards[-1].end != rows:
        raise RuntimeError(f"Embedding shards in {shard_dir} are incomplete")
    LOG.info(
        "Loading exact float32 action matrix on %s: rows=%d (%.1f GiB)",
        device,
        rows,
        rows * EXPECTED_DIMENSION * 4 / (1024**3),
    )
    matrix = torch.empty((rows, EXPECTED_DIMENSION), dtype=torch.float32, device=device)
    for number, shard in enumerate(shards, start=1):
        cached = np.load(shard.path, mmap_mode="r", allow_pickle=False)
        staging = torch.from_numpy(np.array(cached, dtype=np.float32, order="C", copy=True))
        matrix[shard.start:shard.end].copy_(staging)
        if number % 25 == 0 or number == len(shards):
            LOG.info("Action matrix: %d/%d shards", number, len(shards))
    return matrix


def load_title_ids(path: Path, expected_rows: int, device: str) -> Any:
    """Таблица титулов из ``build_title_table.py``: ``int32`` на строку индекса."""
    import torch

    if not path.is_file():
        raise FileNotFoundError(
            f"Title table not found: {path}. Build it with build_title_table.py."
        )
    table = np.load(path, allow_pickle=False)
    if table.shape != (expected_rows,):
        raise ValueError(
            f"Title table has {table.shape} rows, the matrix has {expected_rows}"
        )
    return torch.from_numpy(np.ascontiguousarray(table)).to(device)


def require_search_qrag_repo(qrag_repo: Path) -> None:
    """Убедиться, что подключена копия ``Q-RAG_for_full-wiki``, а не оригинал.

    В дереве два экземпляра ``rl/`` с разным пулингом: read-only оригинал
    считает masked mean и делит на 10, копия — CLS с L2-нормировкой, то есть
    ровно то, чем собран ``wiki18-gte``. Подключить оригинал молча значит
    получить состояние в чужом пространстве и провалить всю фазу 2. Отличает
    их сигнатура: параметр ``normalize`` появился только в копии.
    """
    import inspect

    from rl.bert_predictor import BertPredictor

    if "normalize" not in inspect.signature(BertPredictor.__init__).parameters:
        raise RuntimeError(
            f"{qrag_repo} looks like the read-only Q-RAG original: its "
            "BertPredictor has no `normalize` parameter and pools with masked "
            "mean over /10, so its state vectors are not in the wiki18-gte "
            "space. Point --qrag-repo at Q-RAG_for_full-wiki."
        )


def build_search_state_encoder(
    manifest: dict[str, Any],
    qrag_repo: Path,
    device: str,
    model_dtype: str,
    checkpoint: Path | None,
    state_config: Path | None,
    state_source: str,
) -> tuple[Any, Any, int, str]:
    """State-башня прямого поиска: из чекпоинта или стоковая GTE (zero-shot).

    Zero-shot нужен лестнице фазы 2: без стартовой точки обученную цифру не
    прочитать — 33% после обучения означают разное, если старт был 29% или 24%.
    Собирается он той же архитектурой и тем же путём кодирования, что и
    обучаемая башня, иначе сравнивались бы два разных пайплайна.
    """
    try:
        import torch
        from hydra.utils import instantiate
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("torch, transformers, and hydra-core are required") from error

    add_qrag_repo_to_path(qrag_repo)
    require_search_qrag_repo(qrag_repo)
    from rl.bert_predictor import BertPredictor

    encoder_metadata = manifest["encoder"]
    model_name = str(encoder_metadata["model"])
    revision = str(
        encoder_metadata.get("resolved_revision")
        or encoder_metadata.get("requested_revision", "main")
    )

    if state_config is None and checkpoint is not None:
        candidate = checkpoint.parent / "config.yaml"
        state_config = candidate if candidate.is_file() else None

    separator = " [SEP] "
    max_state_segment_length = int(encoder_metadata["max_length"])
    previous_default_device = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        if state_config is not None:
            config = load_run_config(state_config)
            encoder = instantiate(config.algo.pqn.state_embed)
            max_state_segment_length = int(config.max_action_length_in_memory)
            separator = str(config.envs.get("separator", separator))
        else:
            # Стоковая GTE в архитектуре обучаемой башни: те же 12 слоёв, тот
            # же CLS-пулинг, normalize=false — на ранжирование норма запроса
            # не влияет, а путь кодирования обязан совпадать с обучением.
            LOG.info("Zero-shot state tower: stock %s revision=%s", model_name, revision)
            encoder = BertPredictor(
                bert=AutoModel.from_pretrained(
                    model_name, revision=revision, trust_remote_code=True
                ),
                num_hidden_layers=12,
                tokenizer=AutoTokenizer.from_pretrained(
                    model_name,
                    revision=revision,
                    trust_remote_code=True,
                    local_files_only=True,
                ),
                model_dim=EXPECTED_DIMENSION,
                output_size=EXPECTED_DIMENSION // 2,
                n_output=1,
                normalize=False,
            )
    finally:
        torch.set_default_device(previous_default_device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
        local_files_only=True,
    )
    encoder.tokenizer = tokenizer

    if checkpoint is not None:
        LOG.info("Loading %s state tower from %s", state_source, checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        state = extract_state_dict(loaded, state_source)
        incompatible = encoder.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict state-tower load failed: {incompatible}")
        del state, loaded

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if model_dtype not in dtype_by_name:
        raise ValueError(f"Unsupported model dtype: {model_dtype}")
    encoder.to(device=device, dtype=dtype_by_name[model_dtype])
    encoder.eval()
    return encoder, tokenizer, max_state_segment_length, separator


def verify_index_alignment(
    encoder: Any,
    tokenizer: Any,
    corpus: WikiCorpus,
    matrix: Any,
    row_id: int,
    max_length: int,
    device: str,
    minimum_cosine: float = 0.99,
) -> float:
    """Сверить башню с индексом на одной строке корпуса.

    В zero-shot башня побитово равна стоковой GTE, поэтому нормированный выход
    обязан совпасть со строкой ``M`` с косинусом около 0.999999 — остаток
    объясняется тем, что база кодировалась в fp16. Любой другой пулинг
    (masked mean read-only оригинала) провалит проверку сразу, а не через
    полчаса прогона.
    """
    import torch

    text = corpus.read_rows([row_id])[0]["contents"]
    tokens = tokenizer(
        [text],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        embedding = encoder(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
        )
    embedding = torch.nn.functional.normalize(embedding.float(), dim=-1)
    cosine = float((embedding[0] * matrix[row_id].float()).sum())
    if cosine < minimum_cosine:
        raise RuntimeError(
            f"State tower disagrees with the action matrix on row {row_id}: "
            f"cosine={cosine:.6f} < {minimum_cosine}. The tower is not the "
            "encoder that built wiki18-gte -- check --qrag-repo and the "
            "pooling in rl/bert_predictor.py."
        )
    LOG.info("Index alignment on row %d: cosine=%.6f", row_id, cosine)
    return cosine


class DirectSearchRetriever:
    """Итеративный плотный ретривер: state-башня ищет по всем строкам ``M``.

    Один прямой проход башни даёт и запрос поиска, и все Q-скоры: скор
    ``s @ M[i]`` и оценка ``Q(s, a_i)`` — одно и то же число, рассогласоваться
    нечему. Маска применяется **до** ``topk``: запрос содержит текст уже
    выбранного чанка, поэтому его соседи по статье оказываются ближайшими
    соседями запроса и без маски забили бы пул целиком.
    """

    def __init__(
        self,
        matrix: Any,
        title_ids: Any,
        corpus: WikiCorpus,
        encoder: Any,
        tokenizer: Any,
        separator: str,
        max_state_segment_length: int,
        device: str,
        max_chunks_per_title: int | None,
        log_candidates: str = "full",
    ) -> None:
        if log_candidates not in SUPPORTED_CANDIDATE_LOGS:
            raise ValueError(f"Unsupported candidate log level: {log_candidates}")
        if max_chunks_per_title is not None and max_chunks_per_title < 1:
            raise ValueError(
                f"max_chunks_per_title must be positive: {max_chunks_per_title}"
            )
        self.matrix = matrix
        self.title_ids = title_ids
        self.corpus = corpus
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.separator = separator
        self.max_state_segment_length = max_state_segment_length
        self.device = device
        self.max_chunks_per_title = max_chunks_per_title
        self.log_candidates = log_candidates

    def encode(
        self,
        questions: Sequence[str],
        selected_texts: Sequence[Sequence[str]],
    ) -> np.ndarray:
        return encode_states(
            self.encoder,
            self.tokenizer,
            questions,
            selected_texts,
            self.separator,
            self.max_state_segment_length,
            self.device,
        )

    def search(
        self,
        states: np.ndarray,
        top_k: int,
        blocked_rows: Sequence[Sequence[int]],
        blocked_titles: Sequence[Sequence[int]],
    ) -> tuple[Any, Any]:
        import torch

        with torch.inference_mode():
            queries = torch.from_numpy(states).to(self.device)
            scores = queries @ self.matrix.T
            for row, ids in enumerate(blocked_rows):
                if ids:
                    scores[row, torch.as_tensor(ids, device=self.device)] = -float("inf")
            for row, titles in enumerate(blocked_titles):
                for title_id in titles:
                    scores[row].masked_fill_(self.title_ids == int(title_id), -float("inf"))
            top_k = min(top_k, scores.shape[1])
            # Замаскированная строка не должна добивать пул до K: она попала бы
            # в лог кандидатов как доступная и завысила бы всё, что по этому
            # логу считается.
            available = int(torch.isfinite(scores).sum(dim=1).min())
            if available < top_k:
                raise RuntimeError(
                    f"Only {available} unmasked rows left, top_k={top_k}"
                )
            top_scores, top_ids = torch.topk(
                scores, top_k, dim=1, largest=True, sorted=True
            )
        return top_scores.float().cpu().numpy(), top_ids.cpu().numpy()

    def retrieve(
        self,
        questions: Sequence[str],
        top_k: int,
        steps: int,
        mode: str = "direct",
    ) -> list[dict[str, Any]]:
        if mode != "direct":
            raise ValueError(f"Direct search supports only mode='direct', got {mode!r}")
        batch = len(questions)
        selected_ids: list[list[int]] = [[] for _ in questions]
        selected_texts: list[list[str]] = [[] for _ in questions]
        selected_scores: list[list[float]] = [[] for _ in questions]
        taken_titles: list[Counter] = [Counter() for _ in questions]
        hops: list[list[dict[str, Any]]] = [[] for _ in questions]

        for step in range(steps):
            states = self.encode(questions, selected_texts)
            blocked_titles = [
                [
                    title_id
                    for title_id, count in counter.items()
                    if self.max_chunks_per_title is not None
                    and count >= self.max_chunks_per_title
                ]
                for counter in taken_titles
            ]
            scores, candidate_ids = self.search(
                states, top_k, selected_ids, blocked_titles
            )
            chosen_ids = [int(candidate_ids[row, 0]) for row in range(batch)]
            chosen_rows = self.corpus.read_rows(chosen_ids)
            for row, item in enumerate(chosen_rows):
                chosen_score = float(scores[row, 0])
                if not np.isfinite(chosen_score):
                    raise RuntimeError(f"No unmasked candidate left at step {step}")
                selected_ids[row].append(chosen_ids[row])
                selected_texts[row].append(item["contents"])
                selected_scores[row].append(chosen_score)
                taken_titles[row][int(self.title_ids[chosen_ids[row]])] += 1
                hop: dict[str, Any] = {"step": step}
                if self.log_candidates != "none":
                    hop["candidate_idx"] = [int(value) for value in candidate_ids[row]]
                    hop["candidate_scores"] = [float(value) for value in scores[row]]
                hop["selected_idx"] = chosen_ids[row]
                hop["selected_score"] = chosen_score
                hops[row].append(trim_hop(hop, self.log_candidates))
        return [
            {
                "pred_idx": selected_ids[row],
                "pred_texts": selected_texts[row],
                "q_values": selected_scores[row],
                "retrieval_hops": hops[row],
                "candidate_retriever": DIRECT_SEARCH_RETRIEVER,
                "reranker": DIRECT_SEARCH_RETRIEVER,
            }
            for row in range(batch)
        ]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield item


def load_input_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        first = source.read(1)
    if first == "[":
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise ValueError(f"Expected a JSON array of objects: {path}")
        return data
    if first == "{":
        return list(iter_jsonl(path))
    raise ValueError(f"Unsupported input format: {path}")


def sample_id(sample: dict[str, Any], fallback: int) -> str:
    return str(sample.get("id", sample.get("_id", fallback)))


def supporting_fact_texts(sample: dict[str, Any]) -> list[str]:
    supporting = sample.get("supporting_facts")
    context = sample.get("context")
    if not isinstance(supporting, list) or not isinstance(context, list):
        return []
    by_title = {
        str(title): sentences
        for title, sentences in context
        if isinstance(sentences, list)
    }
    texts = []
    for fact in supporting:
        if not isinstance(fact, list) or len(fact) != 2:
            continue
        title, sentence_index = str(fact[0]), int(fact[1])
        sentences = by_title.get(title, [])
        if 0 <= sentence_index < len(sentences):
            texts.append(f"{title} {sentences[sentence_index]}")
    return texts


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def title_metrics(
    gold_titles: Sequence[str],
    retrieved_titles: Sequence[str],
) -> dict[str, float]:
    """Gold-title recall/EM, both as originally published and normalized.

    ``*_exact`` keeps the originally published definition reproducible; the
    headline metric normalizes both sides because HotpotQA and Wiki-18 come
    from different Wikipedia dumps. See :func:`normalize_title`.
    """
    exact_gold = set(gold_titles)
    exact_found = set(retrieved_titles)
    normal_gold = {normalize_title(title) for title in gold_titles}
    normal_found = {normalize_title(title) for title in retrieved_titles}
    return {
        "title_recall_exact": (
            len(exact_gold & exact_found) / len(exact_gold) if exact_gold else 1.0
        ),
        "title_em_exact": float(exact_gold <= exact_found),
        "title_recall": (
            len(normal_gold & normal_found) / len(normal_gold) if normal_gold else 1.0
        ),
        "title_em": float(normal_gold <= normal_found),
    }


def add_eval_fields(
    sample: dict[str, Any],
    retrieval: dict[str, Any],
    index: int,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    question = sample.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {index} has no non-empty question")
    result = {
        "id": sample_id(sample, index),
        "question": question,
        "answer": sample.get("answer"),
        "sf_idx": sample.get("sf_idx", []),
        "supporting_facts": sample.get("supporting_facts", []),
        **retrieval,
        "retrieval_mode": mode,
        "candidate_pool_size": top_k,
    }
    sf_texts = supporting_fact_texts(sample)
    if sf_texts:
        result["sf_texts"] = sf_texts
    supporting = sample.get("supporting_facts")
    if isinstance(supporting, list):
        gold_titles = ordered_unique(
            str(fact[0]) for fact in supporting if isinstance(fact, list) and fact
        )
        retrieved_titles = [wiki_title(text) for text in retrieval["pred_texts"]]
        result["gold_titles"] = gold_titles
        result["retrieved_titles"] = retrieved_titles
        result.update(title_metrics(gold_titles, retrieved_titles))
    return result


def batched(
    items: Sequence[dict[str, Any]], batch_size: int
) -> Iterator[tuple[int, Sequence[dict[str, Any]]]]:
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def chunks_per_title(args: argparse.Namespace) -> int | None:
    """Квота чанков на одну статью из флагов CLI. ``None`` — без квоты.

    Два флага вместо одного: ``--dedupe-titles/--no-dedupe-titles`` — это
    прежний вкл/выкл, он остался ради ранов и cmd.sh, написанных до квоты, а
    ``--max-chunks-per-title`` задаёт её величину. Значение 1 воспроизводит
    прежнее поведение чанк в чанк.
    """
    if not args.dedupe_titles:
        return None
    return int(getattr(args, "max_chunks_per_title", 1))


def prepare_runtime(
    args: argparse.Namespace,
    *,
    load_model_and_index: bool,
) -> tuple[dict[str, Any], WikiCorpus, FullWikiQrag | None]:
    index_dir = args.index_dir.expanduser().resolve()
    manifest = load_and_validate_manifest(index_dir)
    manifest["_index_dir"] = str(index_dir)
    corpus_path = resolve_corpus_path(manifest, args.corpus)
    offsets_path = (
        args.offsets.expanduser().resolve()
        if args.offsets is not None
        else index_dir / DEFAULT_OFFSETS_FILE
    )
    corpus = WikiCorpus(
        corpus_path,
        offsets_path,
        int(manifest["corpus"]["rows"]),
        auto_prepare=not getattr(args, "no_auto_prepare", False),
    )
    if not load_model_and_index:
        return manifest, corpus, None
    qrag_repo = args.qrag_repo.expanduser().resolve()
    if not (qrag_repo / "rl" / "bert_predictor.py").is_file():
        raise FileNotFoundError(f"Q-RAG repository not found: {qrag_repo}")
    reranker = getattr(args, "reranker", "qrag")
    if reranker == "none":
        LOG.info("Reranker disabled: skipping the Q-RAG state tower entirely")
        state_encoder = tokenizer = None
        max_length, separator, sort_by_index = load_state_settings(manifest)
    else:
        (
            state_encoder,
            tokenizer,
            max_length,
            separator,
            sort_by_index,
        ) = load_state_encoder(
            manifest,
            qrag_repo,
            args.device,
            args.model_dtype,
            args.state_source,
        )
    common = (
        index_dir,
        manifest,
        corpus,
        state_encoder,
        tokenizer,
        max_length,
        separator,
        sort_by_index,
        args.device,
        args.faiss_threads,
    )
    log_candidates = getattr(args, "log_candidates", "full")
    if args.candidate_index_dir is None:
        runner: FullWikiQrag = FullWikiQrag(*common, log_candidates=log_candidates)
    else:
        runner = TwoStageFullWikiQrag(
            *common,
            candidate_index_dir=args.candidate_index_dir.expanduser().resolve(),
            candidate_backend=args.candidate_backend,
            trust_remote_code=args.trust_remote_code,
            max_chunks_per_title=chunks_per_title(args),
            reranker=reranker,
            log_candidates=log_candidates,
            first_stage_query_length=getattr(
                args, "first_stage_query_length", None
            ),
        )
    return manifest, corpus, runner


def command_prepare(args: argparse.Namespace) -> int:
    manifest, _, _ = prepare_runtime(args, load_model_and_index=False)
    LOG.info(
        "Ready: corpus rows=%d offsets=%s",
        manifest["corpus"]["rows"],
        args.offsets or (args.index_dir / DEFAULT_OFFSETS_FILE),
    )
    return 0


def command_retrieve(args: argparse.Namespace) -> int:
    if args.top_k <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--top-k, --steps, and --batch-size must be positive")
    if args.steps > args.top_k and args.mode == "fixed":
        raise ValueError("fixed mode requires --steps <= --top-k")
    if args.reranker == "none" and args.candidate_index_dir is None:
        raise ValueError(
            "--reranker none has nothing to rank without --candidate-index-dir"
        )
    if args.max_chunks_per_title < 1:
        raise ValueError("--max-chunks-per-title must be positive")
    if not args.dedupe_titles and args.max_chunks_per_title != 1:
        raise ValueError(
            "--no-dedupe-titles already lifts every per-title limit; "
            "--max-chunks-per-title only makes sense with --dedupe-titles"
        )
    if args.query is not None:
        samples = [{"id": "query-0", "question": args.query}]
    else:
        samples = load_input_samples(args.input.expanduser().resolve())
    if args.max_samples is not None:
        if args.max_samples < 0:
            raise ValueError("--max-samples cannot be negative")
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No input samples")
    _, _, runner = prepare_runtime(args, load_model_and_index=True)
    assert runner is not None
    output_path = args.output.expanduser().resolve() if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        destination = (
            sys.stdout
            if output_path is None
            else stack.enter_context(output_path.open("w", encoding="utf-8"))
        )
        processed = 0
        title_recalls: list[float] = []
        title_ems: list[float] = []
        title_ems_exact: list[float] = []
        for start, batch in batched(samples, args.batch_size):
            questions = [str(sample["question"]) for sample in batch]
            retrievals = runner.retrieve(questions, args.top_k, args.steps, args.mode)
            for offset, (sample, retrieval) in enumerate(zip(batch, retrievals)):
                result = add_eval_fields(
                    sample, retrieval, start + offset, args.mode, args.top_k
                )
                if "title_recall" in result:
                    title_recalls.append(float(result["title_recall"]))
                    title_ems.append(float(result["title_em"]))
                    title_ems_exact.append(float(result["title_em_exact"]))
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
                processed += 1
            LOG.info("Retrieved %d/%d samples", processed, len(samples))
    if title_recalls:
        LOG.info(
            "Gold-title retrieval over %d samples: recall=%.4f EM=%.4f "
            "(exact-match EM=%.4f)",
            len(title_recalls),
            float(np.mean(title_recalls)),
            float(np.mean(title_ems)),
            float(np.mean(title_ems_exact)),
        )
    if output_path is not None:
        LOG.info("Wrote %d records to %s", processed, output_path)
    return 0


def write_retrieval_records(
    samples: Sequence[dict[str, Any]],
    runner: Any,
    args: argparse.Namespace,
    mode: str,
) -> int:
    """Общий вывод ``retrieval.jsonl`` для батчей любого ретривера.

    Тело почти повторяет цикл внутри :func:`command_retrieve`, и это
    сознательно: `retrieve` дважды правился с побайтовой регрессией по
    docs/pipeline.md §7, и переписать его под общий помощник значит потребовать
    третью — на GPU, ради нулевого выигрыша. Новый режим пользуется отдельной
    копией, старый остаётся неприкосновенным.
    """
    output_path = args.output.expanduser().resolve() if args.output else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    title_recalls: list[float] = []
    title_ems: list[float] = []
    title_ems_exact: list[float] = []
    with ExitStack() as stack:
        destination = (
            sys.stdout
            if output_path is None
            else stack.enter_context(output_path.open("w", encoding="utf-8"))
        )
        for start, batch in batched(samples, args.batch_size):
            questions = [str(sample["question"]) for sample in batch]
            retrievals = runner.retrieve(questions, args.top_k, args.steps, mode)
            for offset, (sample, retrieval) in enumerate(zip(batch, retrievals)):
                result = add_eval_fields(
                    sample, retrieval, start + offset, mode, args.top_k
                )
                if "title_recall" in result:
                    title_recalls.append(float(result["title_recall"]))
                    title_ems.append(float(result["title_em"]))
                    title_ems_exact.append(float(result["title_em_exact"]))
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                destination.flush()
                processed += 1
            LOG.info("Retrieved %d/%d samples", processed, len(samples))
    if title_recalls:
        LOG.info(
            "Gold-title retrieval over %d samples: recall=%.4f EM=%.4f "
            "(exact-match EM=%.4f)",
            len(title_recalls),
            float(np.mean(title_recalls)),
            float(np.mean(title_ems)),
            float(np.mean(title_ems_exact)),
        )
    if output_path is not None:
        LOG.info("Wrote %d records to %s", processed, output_path)
    return processed


def command_search(args: argparse.Namespace) -> int:
    """Итеративный прямой поиск state-башней по всем строкам матрицы."""
    if args.top_k <= 0 or args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("--top-k, --steps, and --batch-size must be positive")
    if args.max_chunks_per_title < 1:
        raise ValueError("--max-chunks-per-title must be positive")
    if not args.device.startswith("cuda"):
        LOG.warning(
            "Direct search multiplies the query by the whole matrix; on %s this "
            "will be extremely slow",
            args.device,
        )
    if args.query is not None:
        samples = [{"id": "query-0", "question": args.query}]
    else:
        samples = load_input_samples(args.input.expanduser().resolve())
    if args.max_samples is not None:
        if args.max_samples < 0:
            raise ValueError("--max-samples cannot be negative")
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No input samples")

    index_dir = args.index_dir.expanduser().resolve()
    manifest = load_and_validate_matrix_manifest(index_dir)
    corpus_path = resolve_corpus_path(manifest, args.corpus)
    offsets_path = (
        args.offsets.expanduser().resolve()
        if args.offsets is not None
        else index_dir / DEFAULT_OFFSETS_FILE
    )
    rows = int(manifest["corpus"]["rows"])
    corpus = WikiCorpus(
        corpus_path, offsets_path, rows, auto_prepare=not args.no_auto_prepare
    )

    qrag_repo = args.qrag_repo.expanduser().resolve()
    if not (qrag_repo / "rl" / "bert_predictor.py").is_file():
        raise FileNotFoundError(f"Q-RAG repository not found: {qrag_repo}")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    encoder, tokenizer, max_state_segment_length, separator = build_search_state_encoder(
        manifest,
        qrag_repo,
        args.device,
        args.model_dtype,
        checkpoint,
        args.state_config.expanduser().resolve() if args.state_config else None,
        args.state_source,
    )

    title_table_path = (
        args.title_table.expanduser().resolve()
        if args.title_table is not None
        else index_dir / DEFAULT_TITLE_TABLE_FILE
    )
    title_ids = load_title_ids(title_table_path, rows, args.device)
    matrix = load_action_matrix(index_dir, manifest, args.device)

    if checkpoint is None and args.verify_alignment:
        # Только zero-shot: у обученной башни расхождение с индексом ожидаемо
        # и как раз является предметом измерения.
        verify_index_alignment(
            encoder,
            tokenizer,
            corpus,
            matrix,
            args.verify_row,
            int(manifest["encoder"]["max_length"]),
            args.device,
        )

    runner = DirectSearchRetriever(
        matrix=matrix,
        title_ids=title_ids,
        corpus=corpus,
        encoder=encoder,
        tokenizer=tokenizer,
        separator=separator,
        max_state_segment_length=max_state_segment_length,
        device=args.device,
        max_chunks_per_title=args.max_chunks_per_title,
        log_candidates=args.log_candidates,
    )
    LOG.info(
        "Direct search: steps=%d top_k=%d max_chunks_per_title=%d state=%s",
        args.steps,
        args.top_k,
        args.max_chunks_per_title,
        checkpoint or "zero-shot (stock GTE)",
    )
    write_retrieval_records(samples, runner, args, mode="direct")
    return 0


def add_shared_index_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--offsets", type=Path, default=None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve from the full Wiki-18 corpus with trained Q-RAG"
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    add_shared_index_arguments(prepare_parser)
    prepare_parser.set_defaults(handler=command_prepare)

    retrieve_parser = subparsers.add_parser("retrieve")
    add_shared_index_arguments(retrieve_parser)
    retrieve_parser.add_argument("--qrag-repo", type=Path, required=True)
    retrieve_parser.add_argument("--candidate-index-dir", type=Path, default=None)
    retrieve_parser.add_argument(
        "--candidate-backend",
        choices=("faiss-cpu", "torch-cuda"),
        default="faiss-cpu",
    )
    retrieve_parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    retrieve_parser.add_argument("--device", default="cuda:0")
    retrieve_parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    retrieve_parser.add_argument(
        "--state-source", choices=SUPPORTED_STATE_SOURCES, default="critic"
    )
    retrieve_parser.add_argument(
        "--mode", choices=("refresh", "fixed"), default="fixed"
    )
    retrieve_parser.add_argument(
        "--reranker",
        choices=SUPPORTED_RERANKERS,
        default="qrag",
        help=(
            "'none' is the first-stage-only baseline: keep the GTE ranking and "
            "skip Q-RAG scoring, the state tower, and the action shards"
        ),
    )
    retrieve_parser.add_argument(
        "--log-candidates",
        choices=SUPPORTED_CANDIDATE_LOGS,
        default="full",
        help=(
            "per-step candidate dump in retrieval_hops; 'none' keeps output "
            "files small, 'full' reproduces the published runs"
        ),
    )
    retrieve_parser.add_argument(
        "--dedupe-titles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    retrieve_parser.add_argument(
        "--max-chunks-per-title",
        type=int,
        default=1,
        help=(
            "how many chunks of one article the selection may take. 1 is the "
            "published behaviour: a title is banned as soon as it is picked. "
            "Wiki-18 cuts articles every 100 words, so the gold sentence often "
            "sits in a neighbouring chunk of an article already taken -- raise "
            "this to let the selection reach it, at the cost of covering fewer "
            "distinct titles on the same chunk budget"
        ),
    )
    retrieve_parser.add_argument(
        "--first-stage-query-length",
        type=int,
        default=None,
        help=(
            "token budget for the first-stage query; default is the corpus "
            "encoding length from the candidate index manifest (256), which "
            "keeps published runs unchanged. Raise it for --mode refresh, "
            "where the query is 'question [SEP] chunk1 [SEP] ...' and 256 "
            "tokens truncate away most of what later hops added"
        ),
    )
    retrieve_parser.add_argument("--top-k", type=int, default=100)
    retrieve_parser.add_argument("--steps", type=int, default=2)
    retrieve_parser.add_argument("--batch-size", type=int, default=16)
    retrieve_parser.add_argument("--faiss-threads", type=int, default=0)
    retrieve_parser.add_argument("--max-samples", type=int, default=None)
    retrieve_parser.add_argument("--no-auto-prepare", action="store_true")
    input_group = retrieve_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--query")
    input_group.add_argument("--input", type=Path)
    retrieve_parser.add_argument("--output", type=Path, default=None)
    retrieve_parser.set_defaults(handler=command_retrieve)

    # Линия A: state-башня сама ищет по всем строкам матрицы. Отдельная
    # подкоманда, а не флаг у `retrieve`: индекс тут другой (wiki18-gte как
    # матрица действий), кандидатов первой стадии нет вовсе, и любое
    # пересечение с флагами `retrieve` меняло бы его поведение.
    search_parser = subparsers.add_parser(
        "search",
        help=(
            "iterative dense retrieval: the state tower queries the whole "
            "action matrix at every hop"
        ),
    )
    add_shared_index_arguments(search_parser)
    search_parser.add_argument(
        "--qrag-repo",
        type=Path,
        required=True,
        help=(
            "must be Q-RAG_for_full-wiki: the read-only original pools with "
            "masked mean and its state vectors are in another space"
        ),
    )
    search_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="model_*.pt of a line-A run; omit for the zero-shot stock GTE tower",
    )
    search_parser.add_argument(
        "--state-config",
        type=Path,
        default=None,
        help=(
            "training config.yaml describing the state tower; defaults to the "
            "one saved next to --checkpoint"
        ),
    )
    search_parser.add_argument(
        "--title-table",
        type=Path,
        default=None,
        help=(
            "int32 table from build_title_table.py; defaults to "
            f"<index-dir>/{DEFAULT_TITLE_TABLE_FILE}"
        ),
    )
    search_parser.add_argument(
        "--state-source", choices=SUPPORTED_STATE_SOURCES, default="critic"
    )
    search_parser.add_argument("--device", default="cuda:0")
    search_parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    search_parser.add_argument(
        "--log-candidates",
        choices=SUPPORTED_CANDIDATE_LOGS,
        default="none",
        help=(
            "per-step candidate dump; 'none' by default because a six-hop run "
            "over 7405 questions dumps 4.4M candidate ids otherwise"
        ),
    )
    search_parser.add_argument("--top-k", type=int, default=100)
    search_parser.add_argument("--steps", type=int, default=6)
    search_parser.add_argument(
        "--max-chunks-per-title",
        type=int,
        default=2,
        help="quota N: how many chunks of one article a single episode may take",
    )
    search_parser.add_argument("--batch-size", type=int, default=16)
    search_parser.add_argument("--max-samples", type=int, default=None)
    search_parser.add_argument("--no-auto-prepare", action="store_true")
    search_parser.add_argument(
        "--verify-alignment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "zero-shot only: check that the tower reproduces a stored matrix "
            "row, which catches a wrong rl/ copy before the run starts"
        ),
    )
    search_parser.add_argument("--verify-row", type=int, default=0)
    search_input = search_parser.add_mutually_exclusive_group(required=True)
    search_input.add_argument("--query")
    search_input.add_argument("--input", type=Path)
    search_parser.add_argument("--output", type=Path, default=None)
    search_parser.set_defaults(handler=command_search)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Failed: %s", error)
        raise
