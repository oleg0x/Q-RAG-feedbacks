#!/usr/bin/env python3
"""Build a restartable Wiki-18 FAISS index from a trained Q-RAG action tower.

The script deliberately does not instantiate or load the complete PQN agent.
It extracts ``checkpoint["critic"]["action_embed.*"]`` into a compact,
auditable artifact and then uses only that tower for passage encoding.

The index preserves the scoring geometry used by Q-RAG:

* masked mean pooling from ``BertPredictor``;
* division by 10 from ``BertPredictor.forward``;
* no L2 normalization;
* inner-product search with ``faiss.IndexFlatIP``.

Phases:

1. ``extract``: create ``action-encoder/`` from the training checkpoint.
2. ``embed``: stream the corpus and write restartable NumPy shards.
3. ``index``: assemble raw vectors into an exact FlatIP index.
4. ``verify``: verify row mapping, values, norms, and shard/index equality.

Example:

    python src/build_index_wiki_qrag.py \
      --run-dir Q-RAG-feedback/runs/Jul24_16-15-51_QRAG_HotPotQA+2WikiMultihopQA \
      --checkpoint best \
      --corpus datasets/data_sources/full-wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
      --output-dir datasets/data_sources/full-wiki/wiki18-qrag-jul24-best-raw \
      --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
      --model-dtype float32 \
      --cache-dtype float32 \
      --text-format raw

The default ``--text-format raw`` preserves the upstream Wiki-18
``"Title"\\npassage`` representation. ``--text-format qrag`` explicitly converts it
to the HotpotQA/2Wiki action format ``Title passage`` for A/B experiments.
The selected transform is recorded in build-state and manifest files, so
shards from the two modes cannot be mixed.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import importlib.metadata
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Iterator, Sequence

import numpy as np

from build_index_wiki_gte import (
    Shard,
    atomic_write_json,
    corpus_identity,
    discover_shards,
    iter_corpus,
    merge_truncation_estimate,
    parse_devices,
    read_json,
    require_faiss,
    save_shard,
    sha256_file,
    utc_now,
)


LOG = logging.getLogger("wiki-qrag-index")

STATE_FILE = "build-state.json"
MANIFEST_FILE = "manifest.json"
SHARDS_DIR = "embedding-shards"
ACTION_ARTIFACT_DIR = "action-encoder"
ACTION_WEIGHTS_FILE = "model.safetensors"
ACTION_METADATA_FILE = "metadata.json"
ACTION_CONFIG_FILE = "run-config.yaml"
ACTION_TOKENIZER_DIR = "tokenizer"

CHECKPOINT_SECTION = "critic"
ACTION_PREFIX = "action_embed."
EXPECTED_DIMENSION = 768
POOLING = "masked_mean"
OUTPUT_SCALE = 0.1
NORMALIZATION = "none"


_WORKER_ENCODER: Any | None = None
_WORKER_DEVICE: str | None = None
_WORKER_BATCH_SIZE: int | None = None
_WORKER_MAX_LENGTH: int | None = None


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    with source.open("rb") as input_stream, temporary.open("wb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    os.replace(temporary, destination)


def parse_wiki_title(title_line: str) -> str:
    """Decode the quoted title used by the Wiki-18 ``contents`` field."""
    title = title_line.strip()
    if len(title) >= 2 and title.startswith('"') and title.endswith('"'):
        try:
            decoded = json.loads(title)
        except json.JSONDecodeError:
            decoded = title[1:-1]
        if isinstance(decoded, str):
            return decoded
    return title


def transform_contents(contents: str, text_format: str) -> str:
    if text_format == "raw":
        return contents
    if text_format != "qrag":
        raise ValueError(f"Unknown text format: {text_format}")

    title_line, separator, passage = contents.partition("\n")
    if not separator:
        return contents.strip()
    title = parse_wiki_title(title_line)
    return f"{title} {passage.lstrip()}".strip()


def checkpoint_path(run_dir: Path, checkpoint: str) -> Path:
    path = run_dir / f"model_{checkpoint}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def infer_qrag_repo(run_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
    else:
        candidate = run_dir.parent.parent.resolve()
    required = candidate / "rl" / "bert_predictor.py"
    if not required.is_file():
        raise ValueError(
            f"Cannot find Q-RAG source at {candidate}. "
            "Pass it explicitly with --qrag-repo."
        )
    return candidate


def load_run_config(config_path: Path) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("omegaconf is required for action extraction") from error
    return OmegaConf.load(config_path)


def config_value(config: Any, dotted_key: str) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("omegaconf is required for config access") from error
    value = OmegaConf.select(config, dotted_key)
    if value is None:
        raise ValueError(f"Missing config value: {dotted_key}")
    return value


def validate_run_config(config: Any) -> dict[str, Any]:
    positions_processor = str(config_value(config, "envs.positions_processor"))
    if positions_processor != "none":
        raise ValueError(
            "A static global action index is valid only for "
            f"positions_processor=none, got {positions_processor!r}"
        )

    model_name = str(config_value(config, "algo.model.model_name"))
    revision = str(config_value(config, "algo.model.revision"))
    configured_max_length = int(config_value(config, "max_action_length"))
    configured_dimension = int(config_value(config, "algo.model.predictor.input_dim"))
    if configured_dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected Q-RAG action dimension {EXPECTED_DIMENSION}, "
            f"config declares {configured_dimension}"
        )
    return {
        "positions_processor": positions_processor,
        "model_name": model_name,
        "requested_revision": revision,
        "configured_max_length": configured_max_length,
        "dimension": configured_dimension,
    }


def extract_action_state_dict(critic_state: dict[str, Any]) -> dict[str, Any]:
    action_state = {
        key.removeprefix(ACTION_PREFIX): value
        for key, value in critic_state.items()
        if key.startswith(ACTION_PREFIX)
    }
    if not action_state:
        raise ValueError(
            f"No keys with prefix {ACTION_PREFIX!r} in checkpoint[{CHECKPOINT_SECTION!r}]"
        )
    unexpected = [key for key in action_state if not key.startswith("model.")]
    if unexpected:
        raise ValueError(
            "Unexpected action state layout; first unexpected keys: "
            f"{unexpected[:5]}"
        )
    return action_state


def add_qrag_repo_to_path(qrag_repo: Path) -> None:
    qrag_string = str(qrag_repo)
    if qrag_string not in sys.path:
        sys.path.insert(0, qrag_string)


def instantiate_action_embedder(config: Any, qrag_repo: Path) -> Any:
    try:
        import torch
        from hydra.utils import instantiate
    except ImportError as error:
        raise RuntimeError("torch and hydra-core are required") from error

    add_qrag_repo_to_path(qrag_repo)
    torch.set_default_device("cpu")
    embedder = instantiate(config.algo.pqn.action_embed)
    return embedder


def tensor_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required") from error

    tensors = [value for value in state.values() if isinstance(value, torch.Tensor)]
    floating = [value for value in tensors if value.dtype.is_floating_point]
    return {
        "keys": len(state),
        "tensors": len(tensors),
        "floating_tensors": len(floating),
        "elements": sum(value.numel() for value in tensors),
        "storage_bytes": sum(value.numel() * value.element_size() for value in tensors),
        "dtypes": sorted({str(value.dtype) for value in tensors}),
    }


def extract_action_artifact(args: argparse.Namespace) -> dict[str, Any]:
    """Create and atomically publish the compact action-tower artifact."""
    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("torch and safetensors are required") from error

    artifact_dir = args.output_dir / ACTION_ARTIFACT_DIR
    metadata_path = artifact_dir / ACTION_METADATA_FILE
    if artifact_dir.exists():
        if not metadata_path.is_file():
            raise RuntimeError(
                f"Incomplete action artifact at {artifact_dir}; use a new output-dir"
            )
        metadata = read_json(metadata_path)
        required = [
            artifact_dir / ACTION_WEIGHTS_FILE,
            artifact_dir / ACTION_CONFIG_FILE,
            artifact_dir / ACTION_TOKENIZER_DIR,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Incomplete action artifact; missing: {missing}")
        expected_checkpoint = checkpoint_path(args.run_dir, args.checkpoint)
        expected_config = args.run_dir / "config.yaml"
        expected_implementation = args.qrag_repo / "rl" / "bert_predictor.py"
        source = metadata.get("source", {})
        if source.get("run_dir") != str(args.run_dir):
            raise RuntimeError(
                "Existing action artifact belongs to another run-dir; "
                "use a new --output-dir"
            )
        if source.get("checkpoint_name") != args.checkpoint:
            raise RuntimeError(
                "Existing action artifact belongs to another checkpoint; "
                "use a new --output-dir"
            )
        for label, current_path, recorded in (
            ("checkpoint", expected_checkpoint, source.get("checkpoint", {})),
            ("config", expected_config, source.get("config", {})),
            (
                "Q-RAG implementation",
                expected_implementation,
                source.get("qrag_implementation", {}),
            ),
        ):
            stat = current_path.stat()
            if (
                recorded.get("path") != str(current_path.resolve())
                or int(recorded.get("size_bytes", -1)) != stat.st_size
                or int(recorded.get("mtime_ns", -1)) != stat.st_mtime_ns
            ):
                raise RuntimeError(
                    f"{label} changed since action extraction; "
                    "use a new --output-dir"
                )
        return metadata

    conflicting = [
        path
        for path in (
            args.output_dir / STATE_FILE,
            args.output_dir / MANIFEST_FILE,
            args.output_dir / SHARDS_DIR,
        )
        if path.exists()
    ]
    if conflicting:
        raise RuntimeError(
            "Output directory already contains another index build but no "
            f"Q-RAG action artifact: {conflicting}. Use a new --output-dir."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary = args.output_dir / f"{ACTION_ARTIFACT_DIR}.tmp-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"Temporary artifact directory already exists: {temporary}")
    temporary.mkdir(parents=True)

    config_path = args.run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    source_checkpoint = checkpoint_path(args.run_dir, args.checkpoint)
    config = load_run_config(config_path)
    run_metadata = validate_run_config(config)

    LOG.info("Loading checkpoint metadata with mmap: %s", source_checkpoint)
    checkpoint = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if CHECKPOINT_SECTION not in checkpoint:
        raise KeyError(
            f"Checkpoint has no section {CHECKPOINT_SECTION!r}; "
            f"available={list(checkpoint)}"
        )
    action_state = extract_action_state_dict(checkpoint[CHECKPOINT_SECTION])

    LOG.info("Instantiating one action tower for strict validation")
    embedder = instantiate_action_embedder(config, args.qrag_repo)
    incompatible = embedder.load_state_dict(action_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict action load failed: {incompatible}")
    embedder.eval()

    actual_state = {
        key: value.detach().cpu().contiguous()
        for key, value in embedder.state_dict().items()
    }
    summary = tensor_state_summary(actual_state)
    if summary["elements"] != tensor_state_summary(action_state)["elements"]:
        raise RuntimeError("Action state changed during strict validation")

    weights_path = temporary / ACTION_WEIGHTS_FILE
    LOG.info("Writing compact action weights: %s", weights_path)
    save_file(
        actual_state,
        str(weights_path),
        metadata={
            "format": "pt",
            "checkpoint_section": CHECKPOINT_SECTION,
            "state_dict_prefix": ACTION_PREFIX,
        },
    )
    atomic_copy(config_path, temporary / ACTION_CONFIG_FILE)
    tokenizer_dir = temporary / ACTION_TOKENIZER_DIR
    embedder.tokenizer.save_pretrained(tokenizer_dir)

    inner_model = getattr(getattr(embedder, "model", None), "model", None)
    inner_config = getattr(inner_model, "config", None)
    resolved_revision = getattr(inner_config, "_commit_hash", None)
    source_checkpoint_identity = file_identity(source_checkpoint)
    source_config_identity = file_identity(config_path)
    implementation_identity = file_identity(
        args.qrag_repo / "rl" / "bert_predictor.py"
    )
    action_weights_identity = file_identity(weights_path)
    action_weights_identity["path"] = ACTION_WEIGHTS_FILE

    metadata = {
        "schema_version": 1,
        "created_at": utc_now(),
        "source": {
            "run_dir": str(args.run_dir),
            "checkpoint_name": args.checkpoint,
            "checkpoint": source_checkpoint_identity,
            "config": source_config_identity,
            "qrag_implementation": implementation_identity,
            "checkpoint_section": CHECKPOINT_SECTION,
            "state_dict_prefix": ACTION_PREFIX,
        },
        "encoder": {
            **run_metadata,
            "resolved_revision": resolved_revision,
            "wrapper": type(embedder).__name__,
            "pooling": POOLING,
            "output_scale": OUTPUT_SCALE,
            "normalization": NORMALIZATION,
            "output_dimension": EXPECTED_DIMENSION,
            "unused_state_keys": ["model.head.weight", "model.head.bias"],
        },
        "weights": {
            **action_weights_identity,
            **summary,
            "file": ACTION_WEIGHTS_FILE,
        },
        "tokenizer_directory": ACTION_TOKENIZER_DIR,
        "run_config_file": ACTION_CONFIG_FILE,
        "software": {
            "python": sys.version.split()[0],
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "hydra_core": package_version("hydra-core"),
            "omegaconf": package_version("omegaconf"),
            "safetensors": package_version("safetensors"),
        },
    }
    atomic_write_json(temporary / ACTION_METADATA_FILE, metadata)
    os.replace(temporary, artifact_dir)
    LOG.info("Action artifact published: %s", artifact_dir)
    return metadata


def action_artifact_metadata(output_dir: Path) -> dict[str, Any]:
    artifact_dir = output_dir / ACTION_ARTIFACT_DIR
    metadata_path = artifact_dir / ACTION_METADATA_FILE
    if not metadata_path.is_file():
        raise RuntimeError(
            f"Action artifact not found at {artifact_dir}; run --phase extract"
        )
    metadata = read_json(metadata_path)
    weights_path = artifact_dir / metadata["weights"]["file"]
    if not weights_path.is_file():
        raise RuntimeError(f"Action weights not found: {weights_path}")
    if weights_path.stat().st_size != int(metadata["weights"]["size_bytes"]):
        raise RuntimeError(f"Action weights size changed: {weights_path}")
    actual_sha256 = sha256_file(weights_path)
    if actual_sha256 != metadata["weights"]["sha256"]:
        raise RuntimeError(f"Action weights checksum changed: {weights_path}")
    return metadata


def load_action_encoder(
    artifact_dir: Path,
    qrag_repo: Path,
    device: str,
    model_dtype: str,
) -> Any:
    try:
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "torch, transformers, and safetensors are required"
        ) from error

    metadata = read_json(artifact_dir / ACTION_METADATA_FILE)
    config = load_run_config(artifact_dir / ACTION_CONFIG_FILE)
    resolved_revision = metadata["encoder"].get("resolved_revision")
    if resolved_revision:
        config.algo.model.revision = resolved_revision
    embedder = instantiate_action_embedder(config, qrag_repo)
    artifact_tokenizer = AutoTokenizer.from_pretrained(
        artifact_dir / ACTION_TOKENIZER_DIR,
        local_files_only=True,
    )
    embedder.tokenizer = artifact_tokenizer
    embedder.model.tokenizer = artifact_tokenizer
    state = load_file(str(artifact_dir / ACTION_WEIGHTS_FILE), device="cpu")
    incompatible = embedder.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict artifact load failed: {incompatible}")
    del state

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if model_dtype not in dtype_by_name:
        raise ValueError(f"Unknown model dtype: {model_dtype}")
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    embedder.to(device=device, dtype=dtype_by_name[model_dtype])
    embedder.eval()
    return embedder


def pad_token_batch(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    device: str,
) -> dict[str, Any]:
    """Match Q-RAG ``stack_text_list`` power-of-two padding."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required") from error

    tokens = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = tokens["input_ids"]
    attention_masks = tokens["attention_mask"]
    if not input_ids:
        raise ValueError("Cannot encode an empty text batch")
    longest = max(len(ids) for ids in input_ids)
    padded_length = 1 << max(0, longest - 1).bit_length()
    if padded_length > max_length:
        padded_length = max_length

    pad_token_id = int(tokenizer.pad_token_id)
    padding_side = tokenizer.padding_side
    ids_tensor = torch.full(
        (len(input_ids), padded_length),
        pad_token_id,
        dtype=torch.int32,
        device=device,
    )
    mask_tensor = torch.zeros(
        (len(input_ids), padded_length),
        dtype=torch.int32,
        device=device,
    )
    for row, (ids, mask) in enumerate(zip(input_ids, attention_masks)):
        ids = ids[:padded_length]
        mask = mask[:padded_length]
        values = torch.tensor(ids, dtype=torch.int32, device=device)
        mask_values = torch.tensor(mask, dtype=torch.int32, device=device)
        if padding_side == "left":
            ids_tensor[row, -len(ids) :] = values
            mask_tensor[row, -len(mask) :] = mask_values
        elif padding_side == "right":
            ids_tensor[row, : len(ids)] = values
            mask_tensor[row, : len(mask)] = mask_values
        else:
            raise ValueError(f"Unsupported tokenizer padding_side={padding_side!r}")
    return {"input_ids": ids_tensor, "attention_mask": mask_tensor}


def encode_with_action_tower(
    encoder: Any,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required") from error

    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            token_batch = pad_token_batch(
                encoder.tokenizer,
                batch_texts,
                max_length,
                device,
            )
            positions = torch.zeros(
                len(batch_texts),
                dtype=torch.float32,
                device=device,
            )
            embeddings = encoder(**token_batch, positions=positions)["rope"]
            embeddings = embeddings.detach().float().cpu().numpy()
            outputs.append(np.asarray(embeddings, dtype=np.float32, order="C"))

    result = np.concatenate(outputs, axis=0)
    if result.shape != (len(texts), EXPECTED_DIMENSION):
        raise RuntimeError(f"Unexpected action embedding shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("Action tower returned NaN or Inf")
    norms = np.linalg.norm(result, axis=1)
    if np.any(norms <= np.finfo(np.float32).tiny):
        raise RuntimeError("Action tower returned a zero vector")
    return result


def _worker_initialize(
    device_queue: Any,
    artifact_dir: str,
    qrag_repo: str,
    model_dtype: str,
    batch_size: int,
    max_length: int,
) -> None:
    global _WORKER_ENCODER
    global _WORKER_DEVICE
    global _WORKER_BATCH_SIZE
    global _WORKER_MAX_LENGTH

    device = device_queue.get()
    _WORKER_DEVICE = device
    _WORKER_BATCH_SIZE = batch_size
    _WORKER_MAX_LENGTH = max_length
    _WORKER_ENCODER = load_action_encoder(
        Path(artifact_dir),
        Path(qrag_repo),
        device,
        model_dtype,
    )


def _worker_encode(task: tuple[int, list[str]]) -> tuple[int, np.ndarray]:
    task_index, texts = task
    if (
        _WORKER_ENCODER is None
        or _WORKER_DEVICE is None
        or _WORKER_BATCH_SIZE is None
        or _WORKER_MAX_LENGTH is None
    ):
        raise RuntimeError("Q-RAG worker was not initialized")
    embeddings = encode_with_action_tower(
        _WORKER_ENCODER,
        texts,
        _WORKER_BATCH_SIZE,
        _WORKER_MAX_LENGTH,
        _WORKER_DEVICE,
    )
    return task_index, embeddings


class ActionEncoderPool:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.devices = parse_devices(args.device)
        self.local_encoder: Any | None = None
        self.pool: Any | None = None
        self.context: Any | None = None

    def __enter__(self) -> "ActionEncoderPool":
        artifact_dir = self.args.output_dir / ACTION_ARTIFACT_DIR
        if len(self.devices) == 1:
            self.local_encoder = load_action_encoder(
                artifact_dir,
                self.args.qrag_repo,
                self.devices[0],
                self.args.model_dtype,
            )
            return self

        if any(not device.startswith("cuda") for device in self.devices):
            raise ValueError("Multi-process encoding currently requires CUDA devices")
        self.context = mp.get_context("spawn")
        device_queue = self.context.Queue()
        for device in self.devices:
            device_queue.put(device)
        self.pool = self.context.Pool(
            processes=len(self.devices),
            initializer=_worker_initialize,
            initargs=(
                device_queue,
                str(artifact_dir),
                str(self.args.qrag_repo),
                self.args.model_dtype,
                self.args.batch_size,
                self.args.max_length,
            ),
        )
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.local_encoder is not None:
            return encode_with_action_tower(
                self.local_encoder,
                texts,
                self.args.batch_size,
                self.args.max_length,
                self.devices[0],
            )
        if self.pool is None:
            raise RuntimeError("ActionEncoderPool is not initialized")

        chunk_size = self.args.multi_process_chunk_size
        tasks = [
            (index, list(texts[start : start + chunk_size]))
            for index, start in enumerate(range(0, len(texts), chunk_size))
        ]
        results = self.pool.map(_worker_encode, tasks)
        results.sort(key=lambda item: item[0])
        embeddings = np.concatenate([item[1] for item in results], axis=0)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Multi-GPU encoder returned {len(embeddings)} rows, "
                f"expected {len(texts)}"
            )
        return embeddings

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.pool is not None:
            if exc_type is None:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
        self.pool = None
        self.local_encoder = None


def load_artifact_tokenizer(output_dir: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required") from error
    path = output_dir / ACTION_ARTIFACT_DIR / ACTION_TOKENIZER_DIR
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def estimate_truncation(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    sample_size: int,
) -> dict[str, int] | None:
    if sample_size <= 0 or not texts:
        return None
    count = min(sample_size, len(texts))
    if count == len(texts):
        sample = list(texts)
    else:
        indices = np.linspace(0, len(texts) - 1, num=count, dtype=np.int64)
        sample = [texts[int(index)] for index in indices]
    tokenized = tokenizer(
        sample,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    return {
        "sampled_passages": len(lengths),
        "would_truncate": sum(length > max_length for length in lengths),
        "max_observed_tokens": max(lengths, default=0),
    }


def requested_build_identity(
    args: argparse.Namespace,
    artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus": {
            **corpus_identity(args.corpus),
            "tar_member": args.tar_member,
            "require_id": not args.allow_missing_id,
            "validate_id_matches_row": (
                not args.allow_missing_id and not args.allow_id_row_mismatch
            ),
            "text_format": args.text_format,
        },
        "encoder": {
            "artifact_sha256": artifact_metadata["weights"]["sha256"],
            "source_checkpoint_sha256": artifact_metadata["source"]["checkpoint"][
                "sha256"
            ],
            "qrag_implementation_sha256": artifact_metadata["source"][
                "qrag_implementation"
            ]["sha256"],
            "source_checkpoint_name": artifact_metadata["source"]["checkpoint_name"],
            "checkpoint_section": CHECKPOINT_SECTION,
            "state_dict_prefix": ACTION_PREFIX,
            "model": artifact_metadata["encoder"]["model_name"],
            "requested_revision": artifact_metadata["encoder"]["requested_revision"],
            "resolved_revision": artifact_metadata["encoder"].get(
                "resolved_revision"
            ),
            "positions_processor": artifact_metadata["encoder"][
                "positions_processor"
            ],
            "pooling": POOLING,
            "output_scale": OUTPUT_SCALE,
            "normalization": NORMALIZATION,
            "dimension": EXPECTED_DIMENSION,
            "max_length": args.max_length,
            "model_dtype": args.model_dtype,
        },
        "shards": {
            "rows_per_shard": args.shard_size,
            "cache_dtype": args.cache_dtype,
            "inference_batch_size": args.batch_size,
            "multi_process_chunk_size": args.multi_process_chunk_size,
            "padding": "power_of_two",
        },
    }


def assert_compatible_state(
    state: dict[str, Any],
    expected_identity: dict[str, Any],
) -> None:
    actual = state.get("build_identity")
    if actual != expected_identity:
        raise RuntimeError(
            "Existing build parameters do not match this request. "
            "Use the original parameters or a new --output-dir.\n"
            f"Existing: {json.dumps(actual, ensure_ascii=False, sort_keys=True)}\n"
            f"Requested: {json.dumps(expected_identity, ensure_ascii=False, sort_keys=True)}"
        )


def iter_transformed_corpus(args: argparse.Namespace) -> Iterator[tuple[int, str]]:
    for row_id, contents in iter_corpus(
        args.corpus,
        require_id=not args.allow_missing_id,
        validate_id_matches_row=(
            not args.allow_missing_id and not args.allow_id_row_mismatch
        ),
        tar_member=args.tar_member,
    ):
        transformed = transform_contents(contents, args.text_format)
        if not transformed:
            raise ValueError(f"Text transform produced an empty passage at row {row_id}")
        yield row_id, transformed


def encode_corpus(
    args: argparse.Namespace,
    artifact_metadata: dict[str, Any],
) -> tuple[list[Shard], dict[str, Any]]:
    output_dir = args.output_dir
    shards_dir = output_dir / SHARDS_DIR
    state_path = output_dir / STATE_FILE
    build_identity = requested_build_identity(args, artifact_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = read_json(state_path)
        assert_compatible_state(state, build_identity)
    else:
        orphaned = list(shards_dir.glob("part-*.npy"))
        if orphaned:
            raise RuntimeError(
                f"Shards exist at {shards_dir}, but {state_path} is missing"
            )
        state = {
            "build_identity": build_identity,
            "created_at": utc_now(),
            "embedding_complete": False,
            "completed_rows": 0,
        }
        atomic_write_json(state_path, state)

    shards = discover_shards(shards_dir, args.cache_dtype)
    completed_rows = shards[-1].end if shards else 0
    recorded_rows = int(state.get("completed_rows", 0))
    if recorded_rows > completed_rows:
        raise RuntimeError(
            f"State records {recorded_rows} rows, shards contain {completed_rows}"
        )
    if recorded_rows < completed_rows:
        LOG.warning(
            "Recovering build-state from atomically written shards: %d -> %d",
            recorded_rows,
            completed_rows,
        )
        state.update({"updated_at": utc_now(), "completed_rows": completed_rows})
        atomic_write_json(state_path, state)

    if state.get("embedding_complete"):
        if int(state.get("total_rows", -1)) != completed_rows:
            raise RuntimeError("Inconsistent completed build-state")
        LOG.info("Embedding phase already complete: %d rows", completed_rows)
        return shards, state

    tokenizer = load_artifact_tokenizer(output_dir)
    buffer: list[str] = []
    buffer_start = completed_rows
    total_rows = 0

    with ActionEncoderPool(args) as encoder_pool:

        def flush_buffer() -> None:
            nonlocal buffer, buffer_start, shards, state
            if not buffer:
                return
            truncation = estimate_truncation(
                tokenizer,
                buffer,
                args.max_length,
                args.truncation_samples_per_shard,
            )
            state["truncation_estimate"] = merge_truncation_estimate(
                state.get("truncation_estimate"),
                truncation,
            )
            LOG.info(
                "Encoding rows [%d, %d), batch_size=%d, devices=%s",
                buffer_start,
                buffer_start + len(buffer),
                args.batch_size,
                args.device,
            )
            embeddings = encoder_pool.encode(buffer)
            shard = save_shard(
                shards_dir,
                buffer_start,
                embeddings,
                args.cache_dtype,
            )
            shards.append(shard)
            norms = np.linalg.norm(embeddings, axis=1)
            state.update(
                {
                    "updated_at": utc_now(),
                    "completed_rows": shard.end,
                    "dimension": shard.dimension,
                    "latest_shard_norms": {
                        "min": float(norms.min()),
                        "mean": float(norms.mean()),
                        "max": float(norms.max()),
                    },
                }
            )
            atomic_write_json(state_path, state)
            LOG.info("Saved %s", shard.path)
            buffer = []
            buffer_start = shard.end

        for row_id, text in iter_transformed_corpus(args):
            total_rows = row_id + 1
            if row_id < completed_rows:
                continue
            expected = buffer_start + len(buffer)
            if row_id != expected:
                raise RuntimeError(
                    f"Corpus order changed: row={row_id}, expected={expected}"
                )
            buffer.append(text)
            if len(buffer) >= args.shard_size:
                flush_buffer()
        if total_rows < completed_rows:
            raise RuntimeError(
                f"Corpus has {total_rows} rows, shards already have {completed_rows}"
            )
        flush_buffer()

    state.update(
        {
            "updated_at": utc_now(),
            "embedding_complete": True,
            "completed_rows": total_rows,
            "total_rows": total_rows,
            "dimension": shards[0].dimension if shards else None,
        }
    )
    atomic_write_json(state_path, state)
    LOG.info("Embedding phase complete: %d rows", total_rows)
    return shards, state


def load_complete_state(
    args: argparse.Namespace,
    artifact_metadata: dict[str, Any],
) -> tuple[list[Shard], dict[str, Any]]:
    state_path = args.output_dir / STATE_FILE
    if not state_path.is_file():
        raise RuntimeError(f"Missing {state_path}; run the embedding phase first")
    state = read_json(state_path)
    assert_compatible_state(
        state,
        requested_build_identity(args, artifact_metadata),
    )
    if not state.get("embedding_complete"):
        raise RuntimeError("Embedding phase is not complete")
    shards = discover_shards(args.output_dir / SHARDS_DIR, args.cache_dtype)
    total_rows = shards[-1].end if shards else 0
    if total_rows != int(state.get("total_rows", -1)):
        raise RuntimeError(
            f"Shards contain {total_rows} rows, state records {state.get('total_rows')}"
        )
    return shards, state


def rows_from_shards(
    shards: Sequence[Shard],
    row_ids: Sequence[int],
) -> np.ndarray:
    ends = [shard.end for shard in shards]
    arrays: dict[int, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for row_id in row_ids:
        shard_index = bisect_right(ends, row_id)
        if shard_index >= len(shards):
            raise IndexError(f"Row id outside shards: {row_id}")
        shard = shards[shard_index]
        if not (shard.start <= row_id < shard.end):
            raise RuntimeError(f"Cannot map row {row_id} to {shard.path}")
        if shard_index not in arrays:
            arrays[shard_index] = np.load(
                shard.path,
                mmap_mode="r",
                allow_pickle=False,
            )
        rows.append(
            np.asarray(
                arrays[shard_index][row_id - shard.start],
                dtype=np.float32,
            )
        )
    return np.vstack(rows)


def verify_index(
    index: Any,
    shards: Sequence[Shard],
    expected_rows: int,
    expected_dimension: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    if int(index.ntotal) != expected_rows:
        raise RuntimeError(
            f"FAISS ntotal={index.ntotal}, expected {expected_rows}"
        )
    if int(index.d) != expected_dimension:
        raise RuntimeError(
            f"FAISS dimension={index.d}, expected {expected_dimension}"
        )
    if expected_rows == 0:
        raise RuntimeError("Cannot verify an empty index")

    count = min(sample_size, expected_rows)
    row_ids = random.Random(seed).sample(range(expected_rows), count)
    reconstructed = np.vstack(
        [index.reconstruct(row_id) for row_id in row_ids]
    ).astype(np.float32, copy=False)
    expected = rows_from_shards(shards, row_ids)
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("FAISS reconstructed vectors contain NaN or Inf")
    norms = np.linalg.norm(reconstructed, axis=1)
    if np.any(norms <= np.finfo(np.float32).tiny):
        raise RuntimeError("FAISS contains a zero vector")
    max_shard_error = float(np.max(np.abs(reconstructed - expected)))
    if max_shard_error > 1e-6:
        raise RuntimeError(
            f"FAISS/shard mismatch: max absolute error={max_shard_error:.6g}"
        )

    return {
        "verified_at": utc_now(),
        "sample_size": count,
        "max_shard_error": max_shard_error,
        "sample_norms": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "median": float(np.median(norms)),
            "max": float(norms.max()),
        },
        "checks": [
            "ntotal",
            "dimension",
            "random_reconstruct",
            "finite_vectors",
            "nonzero_vectors",
            "shard_index_equality",
        ],
        "normalization_expected": False,
    }


def build_flat_index(
    args: argparse.Namespace,
    artifact_metadata: dict[str, Any],
    shards: Sequence[Shard],
    state: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not shards:
        raise RuntimeError("No embedding shards found")
    faiss = require_faiss()
    dimension = shards[0].dimension
    total_rows = shards[-1].end
    index_path = args.output_dir / args.index_name
    temporary = index_path.with_name(index_path.name + ".tmp")

    if index_path.exists():
        LOG.info("Index already exists; verifying %s", index_path)
        index = faiss.read_index(str(index_path))
    else:
        LOG.info(
            "Building raw IndexFlatIP: rows=%d, dimension=%d",
            total_rows,
            dimension,
        )
        index = faiss.IndexFlatIP(dimension)
        for shard in shards:
            LOG.info("Adding %s", shard.path.name)
            cached = np.load(shard.path, mmap_mode="r", allow_pickle=False)
            vectors = np.array(
                cached,
                dtype=np.float32,
                order="C",
                copy=True,
            )
            if not np.isfinite(vectors).all():
                raise RuntimeError(f"Shard contains NaN or Inf: {shard.path}")
            # Deliberately do not normalize: Q-RAG policy uses raw dot products.
            index.add(vectors)
            if int(index.ntotal) != shard.end:
                raise RuntimeError(
                    f"After {shard.path}, ntotal={index.ntotal}, "
                    f"expected {shard.end}"
                )
        verification = verify_index(
            index,
            shards,
            total_rows,
            dimension,
            args.verify_samples,
            args.seed,
        )
        LOG.info("Writing temporary index %s", temporary)
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)
        LOG.info("Index written: %s", index_path)

    verification = verify_index(
        index,
        shards,
        total_rows,
        dimension,
        args.verify_samples,
        args.seed,
    )
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "purpose": "trained Q-RAG action-tower retrieval",
        "corpus": {
            **state["build_identity"]["corpus"],
            "source_path": str(args.corpus.resolve()),
            "rows": total_rows,
            "contents_field": "contents",
            "faiss_id_mapping": "zero-based JSONL row number",
        },
        "encoder": {
            **state["build_identity"]["encoder"],
            "artifact_directory": ACTION_ARTIFACT_DIR,
            "query_encoder_required": (
                "checkpoint['critic'] keys with prefix 'state_embed.'"
            ),
        },
        "embedding_cache": {
            **state["build_identity"]["shards"],
            "directory": SHARDS_DIR,
            "retained": True,
            "shard_count": len(shards),
        },
        "index": {
            "file": index_path.name,
            "type": "faiss.IndexFlatIP",
            "metric": "raw_inner_product",
            "normalization": "none",
            "dimension": dimension,
            "ntotal": total_rows,
            "id_map": "implicit contiguous row ids; no IndexIDMap wrapper",
            "size_bytes": index_path.stat().st_size,
        },
        "action_artifact": artifact_metadata,
        "verification": verification,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", None),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        },
    }
    atomic_write_json(args.output_dir / MANIFEST_FILE, manifest)
    return index_path, manifest


def verify_existing_index(
    args: argparse.Namespace,
    artifact_metadata: dict[str, Any],
) -> dict[str, Any]:
    shards, state = load_complete_state(args, artifact_metadata)
    manifest_path = args.output_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing {manifest_path}")
    manifest = read_json(manifest_path)
    index_path = args.output_dir / manifest["index"]["file"]
    faiss = require_faiss()
    index = faiss.read_index(str(index_path))
    verification = verify_index(
        index,
        shards,
        int(state["total_rows"]),
        int(state["dimension"]),
        args.verify_samples,
        args.seed,
    )
    LOG.info("Verification successful: %s", json.dumps(verification))
    return verification


def resolve_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.run_dir = args.run_dir.expanduser().resolve()
    args.corpus = args.corpus.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.qrag_repo = infer_qrag_repo(args.run_dir, args.qrag_repo)

    if not args.run_dir.is_dir():
        parser.error(f"Run directory not found: {args.run_dir}")
    if not args.corpus.is_file():
        parser.error(f"Corpus not found: {args.corpus}")
    config_path = args.run_dir / "config.yaml"
    if not config_path.is_file():
        parser.error(f"Run config not found: {config_path}")
    checkpoint_path(args.run_dir, args.checkpoint)

    config = load_run_config(config_path)
    run_metadata = validate_run_config(config)
    configured_max_length = run_metadata["configured_max_length"]
    if args.max_length is None:
        args.max_length = configured_max_length
    elif args.max_length != configured_max_length:
        LOG.warning(
            "max_length=%d differs from training max_action_length=%d",
            args.max_length,
            configured_max_length,
        )
    if args.index_name is None:
        args.index_name = (
            f"qrag-action-{args.checkpoint}.raw.flatip.faiss"
        )

    try:
        devices = parse_devices(args.device)
    except ValueError as error:
        parser.error(str(error))
    if len(set(devices)) != len(devices):
        parser.error("--device contains duplicates")
    if (
        args.batch_size <= 0
        or args.shard_size <= 0
        or args.max_length <= 0
        or args.multi_process_chunk_size <= 0
    ):
        parser.error(
            "batch-size, shard-size, max-length, and "
            "multi-process-chunk-size must be positive"
        )
    if args.truncation_samples_per_shard < 0:
        parser.error("--truncation-samples-per-shard cannot be negative")
    if args.verify_samples <= 0:
        parser.error("--verify-samples must be positive")
    if args.output_dir == args.run_dir or args.output_dir == args.corpus.parent:
        parser.error(
            "--output-dir must be a dedicated new directory, not the run or corpus directory"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a raw FlatIP Wiki-18 index from a trained Q-RAG action tower"
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        choices=("best", "last"),
        default="best",
    )
    parser.add_argument("--qrag-repo", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tar-member", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--text-format",
        choices=("qrag", "raw"),
        default="raw",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="One device or a comma-separated list, e.g. cuda:0,cuda:1",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float32", "float16"),
        default="float32",
    )
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--multi-process-chunk-size", type=int, default=1_000)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument(
        "--truncation-samples-per-shard",
        type=int,
        default=2_048,
    )
    parser.add_argument(
        "--phase",
        choices=("all", "extract", "embed", "index", "verify"),
        default="all",
    )
    parser.add_argument("--index-name", default=None)
    parser.add_argument("--verify-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-id", action="store_true")
    parser.add_argument("--allow-id-row-mismatch", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    resolve_args(args, parser)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.model_dtype != "float32":
        LOG.warning(
            "Inference uses %s while the checkpoint was trained in float32; "
            "run the planned FP32 parity smoke before the full build",
            args.model_dtype,
        )

    extract_action_artifact(args)
    artifact_metadata = action_artifact_metadata(args.output_dir)
    if args.phase == "extract":
        return 0

    if args.phase == "verify":
        verify_existing_index(args, artifact_metadata)
        return 0

    manifest_path = args.output_dir / MANIFEST_FILE
    if args.phase == "all" and manifest_path.exists():
        verify_existing_index(args, artifact_metadata)
        LOG.info("A complete verified index already exists")
        return 0

    if args.phase in {"all", "embed"}:
        shards, state = encode_corpus(args, artifact_metadata)
    else:
        shards, state = load_complete_state(args, artifact_metadata)
    if args.phase == "embed":
        return 0

    build_flat_index(
        args,
        artifact_metadata,
        shards,
        state,
    )
    LOG.info("Done. Manifest: %s", args.output_dir / MANIFEST_FILE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted; completed action artifact and shards are restartable")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Build failed: %s", error)
        raise
