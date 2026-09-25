#!/usr/bin/env python3
"""Build a reproducible GTE/FAISS index for the Search-R1 Wiki-18 corpus.

The builder deliberately has two phases:

1. Stream ``wiki-18.jsonl[.gz]`` and write restartable embedding shards.
2. Assemble the shards into a normalized FAISS ``IndexFlatIP`` index.

FAISS row ids are corpus row numbers.  Keep the original JSONL file unchanged:
the retrieval service must use the same file and row order when resolving ids.

This is a *static first-stage GTE index*.  It is not an index of a fine-tuned
Q-RAG action tower.  If the Q-RAG action/document encoder is updated, use this
index only to generate candidates and rerank them with Q-RAG, or rebuild the
index from the updated action encoder.

Example:

    python src/build_index_wiki_gte.py \
      --corpus /data/wiki-18.jsonl.gz \
      --output-dir /data/wiki18-gte \
      --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
      --model-dtype float16 \
      --max-length 256 \
      --batch-size 256 \
      --shard-size 100000

Set ``--max-length`` to the exact value used by the matching query/action
encoder. The builder samples untruncated token lengths and records the observed
truncation rate in ``build-state.json`` and ``manifest.json``.

The default cache dtype is float32.  This preserves the exact normalized GTE
vectors but temporarily needs about another 65 GB on top of the 65 GB Flat
index.  ``--cache-dtype float16`` halves cache size at the cost of a very small
direction quantization.  Shards are retained by default for reproducibility
and restartability; remove them only with ``--delete-shards-after-build``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gzip
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import random
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Sequence

import numpy as np


DEFAULT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_INDEX_NAME = "gte-multilingual-base.flatip.faiss"
STATE_FILE = "build-state.json"
MANIFEST_FILE = "manifest.json"
SHARDS_DIR = "embedding-shards"
SHARD_PATTERN = re.compile(r"^part-(\d{12})-(\d{12})\.npy$")
LOG = logging.getLogger("wiki-gte-index")


@dataclass(frozen=True)
class Shard:
    path: Path
    start: int
    end: int
    rows: int
    dimension: int
    dtype: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Ожидался JSON-объект в {path}")
    return value


def detect_corpus_container(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(512)
    if header[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as stream:
            payload_header = stream.read(512)
        if (
            len(payload_header) >= 262
            and payload_header[257:262] == b"ustar"
        ):
            return "gzip-tar"
        return "gzip"
    if len(header) >= 262 and header[257:262] == b"ustar":
        return "tar"
    return "plain"


@contextmanager
def open_corpus(
    path: Path,
    tar_member: str | None = None,
) -> Iterator[BinaryIO]:
    """Open by file signature, not suffix: downloaded files are often renamed."""
    container = detect_corpus_container(path)
    if container in {"tar", "gzip-tar"}:
        if not tar_member:
            raise ValueError(
                f"{path} содержит tar-контейнер. Передайте путь JSONL внутри "
                "архива через --tar-member; имена можно посмотреть командой "
                f"`tar -tf {path}`."
            )
        mode = "r|gz" if container == "gzip-tar" else "r|"
        with tarfile.open(path, mode=mode) as archive:
            for member in archive:
                if member.name != tar_member:
                    continue
                if not member.isfile():
                    raise ValueError(
                        f"Tar member {tar_member!r} не является обычным файлом"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Не удалось открыть tar member {tar_member!r}")
                with extracted:
                    yield extracted
                return
        raise ValueError(f"Tar member {tar_member!r} не найден в {path}")

    compressed = container == "gzip"
    if compressed and path.suffix != ".gz":
        LOG.warning("Корпус является gzip по magic bytes, несмотря на имя %s", path)
    elif not compressed and path.suffix == ".gz":
        LOG.warning("Корпус имеет суффикс .gz, но не gzip magic bytes: %s", path)
    if compressed:
        with gzip.open(path, "rb") as stream:
            yield stream
        return
    with path.open("rb") as stream:
        yield stream


def iter_corpus(
    path: Path,
    require_id: bool = True,
    validate_id_matches_row: bool = True,
    tar_member: str | None = None,
) -> Iterator[tuple[int, str]]:
    """Yield ``(zero_based_row_id, contents)`` without materializing the corpus."""
    row_id = 0
    with open_corpus(path, tar_member=tar_member) as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                raise ValueError(
                    f"Пустая строка {line_number} в {path}; она нарушает mapping "
                    "FAISS row id -> corpus row"
                )
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                context_start = max(0, error.start - 8)
                context_end = min(len(raw_line), error.end + 8)
                context = raw_line[context_start:context_end].hex(" ")
                raise ValueError(
                    f"Строка {line_number} в {path} не является UTF-8: "
                    f"байт {error.start}, контекст hex={context}. "
                    "Проверьте тип файла командой `file` и целостность дампа."
                ) from error
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Некорректный JSON в строке {line_number}: {error}"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(f"Строка {line_number}: ожидался JSON-объект")
            if require_id and "id" not in item:
                raise ValueError(f"Строка {line_number}: отсутствует поле 'id'")
            if validate_id_matches_row and "id" in item:
                try:
                    corpus_id = int(item["id"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Строка {line_number}: id={item['id']!r} не является целым числом"
                    ) from error
                if corpus_id != row_id:
                    raise ValueError(
                        f"Строка {line_number}: corpus id={corpus_id}, "
                        f"но zero-based row id={row_id}"
                    )
            contents = item.get("contents")
            if not isinstance(contents, str) or not contents.strip():
                raise ValueError(
                    f"Строка {line_number}: поле 'contents' должно быть непустой строкой"
                )
            yield row_id, contents
            row_id += 1


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def cached_sha256(path_string: str, size_bytes: int, mtime_ns: int) -> str:
    # size and mtime only invalidate the per-process cache; they are not part of
    # the persisted identity. A plain `touch` therefore does not break resume.
    del size_bytes, mtime_ns
    path = Path(path_string)
    LOG.info("Вычисляю SHA-256 исходного файла: %s", path)
    return sha256_file(path)


def corpus_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "sha256": cached_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns),
    }


def requested_build_identity(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "corpus": {
            **corpus_identity(args.corpus),
            "tar_member": args.tar_member,
            "require_id": not args.allow_missing_id,
            "validate_id_matches_row": (
                not args.allow_missing_id and not args.allow_id_row_mismatch
            ),
        },
        "encoder": {
            "api": "sentence_transformers.SentenceTransformer.encode",
            "model": args.model,
            "requested_revision": args.revision,
            "max_length": args.max_length,
            "normalization": "l2",
            "model_dtype": args.model_dtype,
        },
        "shards": {
            "rows_per_shard": args.shard_size,
            "cache_dtype": args.cache_dtype,
        },
    }


def assert_compatible_state(
    state: dict[str, Any], expected_identity: dict[str, Any]
) -> None:
    actual = state.get("build_identity")
    # Backward compatibility: tar support was added after the first version of
    # build-state.json. A missing key and explicit null both mean that the
    # corpus was read directly rather than through a tar member.
    if isinstance(actual, dict):
        actual_corpus = actual.get("corpus")
        if isinstance(actual_corpus, dict):
            actual_corpus.setdefault("tar_member", None)
    if actual != expected_identity:
        raise RuntimeError(
            "Параметры существующей сборки не совпадают с текущими. "
            "Используйте прежние параметры или новый --output-dir.\n"
            f"Существующие: {json.dumps(actual, ensure_ascii=False, sort_keys=True)}\n"
            f"Текущие: {json.dumps(expected_identity, ensure_ascii=False, sort_keys=True)}"
        )


def discover_shards(shards_dir: Path, expected_dtype: str | None = None) -> list[Shard]:
    if not shards_dir.exists():
        return []

    shards: list[Shard] = []
    for path in sorted(shards_dir.iterdir()):
        match = SHARD_PATTERN.match(path.name)
        if not match:
            if path.name.endswith(".tmp"):
                continue
            raise RuntimeError(f"Неожиданный файл в каталоге shards: {path}")
        start, end = map(int, match.groups())
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            raise RuntimeError(f"Shard {path} должен быть двумерным, shape={array.shape}")
        rows, dimension = array.shape
        if end - start != rows:
            raise RuntimeError(
                f"Имя и shape shard не совпадают: {path}, rows={rows}"
            )
        dtype = np.dtype(array.dtype).name
        if expected_dtype is not None and dtype != expected_dtype:
            raise RuntimeError(
                f"Shard {path} имеет dtype={dtype}, ожидался {expected_dtype}"
            )
        shards.append(Shard(path, start, end, rows, dimension, dtype))

    expected_start = 0
    dimensions: set[int] = set()
    for shard in shards:
        if shard.start != expected_start:
            raise RuntimeError(
                f"Непоследовательные shards: ожидался start={expected_start}, "
                f"получен {shard.start} в {shard.path}"
            )
        expected_start = shard.end
        dimensions.add(shard.dimension)
    if len(dimensions) > 1:
        raise RuntimeError(f"Shards имеют разные размерности: {sorted(dimensions)}")
    return shards


def save_shard(
    shards_dir: Path,
    start: int,
    embeddings: np.ndarray,
    cache_dtype: str,
) -> Shard:
    end = start + len(embeddings)
    path = shards_dir / f"part-{start:012d}-{end:012d}.npy"
    temporary = path.with_name(path.name + ".tmp")
    shards_dir.mkdir(parents=True, exist_ok=True)

    array = np.asarray(embeddings, dtype=np.dtype(cache_dtype), order="C")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return Shard(path, start, end, len(array), array.shape[1], array.dtype.name)


def parse_devices(value: str) -> list[str]:
    devices = [device.strip() for device in value.split(",")]
    if not devices or any(not device for device in devices):
        raise ValueError(
            "--device должен быть одним устройством или списком через запятую, "
            "например cuda:0,cuda:1"
        )
    return devices


def load_sentence_transformer(args: argparse.Namespace) -> tuple[Any, str | None]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Для embedding-фазы установите torch и sentence-transformers>=3.0.0"
        ) from error

    devices = parse_devices(args.device)
    LOG.info("Загружаю %s revision=%s на %s", args.model, args.revision, devices[0])
    model = SentenceTransformer(
        args.model,
        revision=args.revision,
        device=devices[0],
        trust_remote_code=args.trust_remote_code,
    )
    model.max_seq_length = args.max_length
    if args.model_dtype == "float16":
        model.half()
    elif args.model_dtype == "bfloat16":
        model.bfloat16()
    elif args.model_dtype != "float32":
        raise ValueError(f"Неизвестный model dtype: {args.model_dtype}")

    dimension = model.get_sentence_embedding_dimension()
    if dimension != 768:
        raise RuntimeError(
            f"Ожидалась размерность GTE 768, модель вернула {dimension}. "
            "Проверьте checkpoint и truncate_dim."
        )

    resolved_revision: str | None = None
    try:
        transformer = model[0]
        resolved_revision = transformer.auto_model.config._commit_hash
    except (AttributeError, IndexError, TypeError):
        pass

    LOG.info(
        "Модель готова: dim=%d, max_length=%d, resolved_revision=%s",
        dimension,
        model.max_seq_length,
        resolved_revision or "unknown",
    )
    return model, resolved_revision


def encode_texts(
    model: Any,
    texts: Sequence[str],
    batch_size: int,
    pool: dict[str, Any] | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    if pool is None:
        embeddings = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    else:
        # Current SentenceTransformers exposes multiprocessing through encode().
        # Keep the older API as a compatibility path for ST 3.x/4.x.
        encode_parameters = inspect.signature(model.encode).parameters
        if "pool" in encode_parameters:
            embeddings = model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
                pool=pool,
                chunk_size=chunk_size,
            )
        elif hasattr(model, "encode_multi_process"):
            embeddings = model.encode_multi_process(
                list(texts),
                pool,
                batch_size=batch_size,
                chunk_size=chunk_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            raise RuntimeError(
                "Установленная sentence-transformers не поддерживает multi-process encode"
            )
    # Explicit float32 conversion also makes bfloat16 inference portable to NumPy.
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().float().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
    if embeddings.ndim != 2 or embeddings.shape[1] != 768:
        raise RuntimeError(f"Неожиданный shape embeddings: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Модель вернула NaN или Inf")

    # normalize_embeddings=True runs in the model dtype. fp16/bfloat16 can
    # therefore have visible norm drift after conversion to float32. Normalize
    # once more in float32 so cached vectors have a stable invariant.
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float32).tiny):
        raise RuntimeError("Модель вернула вектор с нулевой или некорректной нормой")
    max_input_error = float(np.max(np.abs(norms - 1.0)))
    if max_input_error > 1e-3:
        LOG.debug(
            "Перенормирую embeddings в float32: max |input norm - 1|=%.6g",
            max_input_error,
        )
    embeddings /= norms[:, None]

    normalized_norms = np.linalg.norm(embeddings, axis=1)
    max_error = float(np.max(np.abs(normalized_norms - 1.0)))
    if max_error > 1e-3:
        raise RuntimeError(
            "Нарушена float32 L2-нормализация после перенормировки: "
            f"max |norm - 1| = {max_error:.6g}"
        )
    return embeddings


def estimate_truncation(
    model: Any,
    texts: Sequence[str],
    max_length: int,
    sample_size: int,
) -> dict[str, int] | None:
    """Estimate how many passages would be truncated by the encoder."""
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or sample_size <= 0 or not texts:
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
    try:
        input_ids = tokenized["input_ids"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("Tokenizer не вернул поле input_ids") from error
    lengths = [len(ids) for ids in input_ids]
    return {
        "sampled_passages": len(lengths),
        "would_truncate": sum(length > max_length for length in lengths),
        "max_observed_tokens": max(lengths, default=0),
    }


def merge_truncation_estimate(
    current: dict[str, Any] | None,
    update: dict[str, int] | None,
) -> dict[str, Any] | None:
    if update is None:
        return current
    merged = dict(current or {})
    merged["sampled_passages"] = int(merged.get("sampled_passages", 0)) + update[
        "sampled_passages"
    ]
    merged["would_truncate"] = int(merged.get("would_truncate", 0)) + update[
        "would_truncate"
    ]
    merged["max_observed_tokens"] = max(
        int(merged.get("max_observed_tokens", 0)), update["max_observed_tokens"]
    )
    sampled = merged["sampled_passages"]
    merged["estimated_truncation_rate"] = (
        merged["would_truncate"] / sampled if sampled else 0.0
    )
    return merged


def encode_corpus(args: argparse.Namespace) -> tuple[list[Shard], dict[str, Any]]:
    output_dir: Path = args.output_dir
    shards_dir = output_dir / SHARDS_DIR
    state_path = output_dir / STATE_FILE
    build_identity = requested_build_identity(args)

    output_dir.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = read_json(state_path)
        assert_compatible_state(state, build_identity)
    else:
        orphaned_shards = list(shards_dir.glob("part-*.npy"))
        if orphaned_shards:
            raise RuntimeError(
                f"В {shards_dir} есть shards, но отсутствует {state_path}. "
                "Нельзя безопасно определить, какой моделью они созданы; "
                "используйте новый --output-dir."
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
    state_completed_rows = int(state.get("completed_rows", 0))
    if state_completed_rows > completed_rows:
        raise RuntimeError(
            f"build-state сообщает {state_completed_rows} строк, "
            f"а shards содержат {completed_rows}"
        )
    if state_completed_rows < completed_rows:
        # A crash can happen after the atomic shard rename but before state update.
        LOG.warning(
            "Восстанавливаю build-state по проверенным shards: %d -> %d строк",
            state_completed_rows,
            completed_rows,
        )
        state.update({"updated_at": utc_now(), "completed_rows": completed_rows})
        atomic_write_json(state_path, state)

    if state.get("embedding_complete"):
        if int(state.get("total_rows", -1)) != completed_rows:
            raise RuntimeError("Неконсистентный завершённый build-state")
        LOG.info("Embedding-фаза уже завершена: %d строк", completed_rows)
        return shards, state

    if completed_rows:
        LOG.info("Продолжаю после %d уже закодированных строк", completed_rows)

    model: Any | None = None
    pool: dict[str, Any] | None = None
    devices = parse_devices(args.device)
    resolved_revision: str | None = state.get("resolved_model_revision")
    buffer: list[str] = []
    buffer_start = completed_rows
    total_rows = 0

    def flush_buffer() -> None:
        nonlocal model, pool, resolved_revision, buffer, buffer_start, shards, state
        if not buffer:
            return
        if model is None:
            model, loaded_revision = load_sentence_transformer(args)
            recorded_revision = state.get("resolved_model_revision")
            if (
                recorded_revision
                and loaded_revision
                and recorded_revision != loaded_revision
            ):
                raise RuntimeError(
                    "Модель изменилась между запусками: "
                    f"раньше {recorded_revision}, сейчас {loaded_revision}. "
                    "Используйте точный --revision или новый --output-dir."
                )
            resolved_revision = loaded_revision or recorded_revision
            if len(devices) > 1:
                LOG.info("Запускаю reusable multi-process pool на %s", devices)
                pool = model.start_multi_process_pool(target_devices=devices)
        truncation = estimate_truncation(
            model,
            buffer,
            args.max_length,
            args.truncation_samples_per_shard,
        )
        state["truncation_estimate"] = merge_truncation_estimate(
            state.get("truncation_estimate"), truncation
        )
        if state.get("truncation_estimate"):
            estimate = state["truncation_estimate"]
            LOG.info(
                "Оценка усечения при max_length=%d: %.3f%% (%d/%d), max=%d токенов",
                args.max_length,
                100.0 * estimate["estimated_truncation_rate"],
                estimate["would_truncate"],
                estimate["sampled_passages"],
                estimate["max_observed_tokens"],
            )
        LOG.info(
            "Кодирую строки [%d, %d), batch_size=%d",
            buffer_start,
            buffer_start + len(buffer),
            args.batch_size,
        )
        embeddings = encode_texts(
            model,
            buffer,
            args.batch_size,
            pool=pool,
            chunk_size=args.multi_process_chunk_size,
        )
        shard = save_shard(shards_dir, buffer_start, embeddings, args.cache_dtype)
        shards.append(shard)
        state.update(
            {
                "updated_at": utc_now(),
                "completed_rows": shard.end,
                "dimension": shard.dimension,
                "resolved_model_revision": resolved_revision,
            }
        )
        atomic_write_json(state_path, state)
        LOG.info("Сохранён %s", shard.path)
        buffer = []
        buffer_start = shard.end

    try:
        for row_id, contents in iter_corpus(
            args.corpus,
            require_id=not args.allow_missing_id,
            validate_id_matches_row=(
                not args.allow_missing_id and not args.allow_id_row_mismatch
            ),
            tar_member=args.tar_member,
        ):
            total_rows = row_id + 1
            if row_id < completed_rows:
                continue
            if row_id != buffer_start + len(buffer):
                raise RuntimeError(
                    f"Нарушен порядок строк: row_id={row_id}, "
                    f"ожидался {buffer_start + len(buffer)}"
                )
            buffer.append(contents)
            if len(buffer) >= args.shard_size:
                flush_buffer()

        if total_rows < completed_rows:
            raise RuntimeError(
                f"Корпус содержит {total_rows} строк, "
                f"но shards уже содержат {completed_rows}"
            )
        flush_buffer()
    finally:
        if pool is not None and model is not None:
            LOG.info("Останавливаю multi-process pool")
            model.stop_multi_process_pool(pool)

    state.update(
        {
            "updated_at": utc_now(),
            "embedding_complete": True,
            "completed_rows": total_rows,
            "total_rows": total_rows,
            "dimension": shards[0].dimension if shards else None,
            "resolved_model_revision": resolved_revision,
        }
    )
    atomic_write_json(state_path, state)
    LOG.info("Embedding-фаза завершена: %d строк", total_rows)
    return shards, state


def require_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "Для сборки индекса установите faiss-cpu или faiss-gpu"
        ) from error
    return faiss


def load_complete_state(args: argparse.Namespace) -> tuple[list[Shard], dict[str, Any]]:
    state_path = args.output_dir / STATE_FILE
    if not state_path.exists():
        raise RuntimeError(f"Не найден {state_path}; сначала запустите embedding-фазу")
    state = read_json(state_path)
    assert_compatible_state(state, requested_build_identity(args))
    if not state.get("embedding_complete"):
        raise RuntimeError("Embedding-фаза ещё не завершена")
    shards = discover_shards(args.output_dir / SHARDS_DIR, args.cache_dtype)
    total_rows = shards[-1].end if shards else 0
    if total_rows != int(state.get("total_rows", -1)):
        raise RuntimeError(
            f"Shards содержат {total_rows} строк, build-state — {state.get('total_rows')}"
        )
    return shards, state


def verify_index(
    index: Any,
    expected_rows: int,
    expected_dimension: int,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    if int(index.ntotal) != expected_rows:
        raise RuntimeError(
            f"FAISS ntotal={index.ntotal}, ожидалось {expected_rows}"
        )
    if int(index.d) != expected_dimension:
        raise RuntimeError(f"FAISS d={index.d}, ожидалось {expected_dimension}")
    if expected_rows == 0:
        raise RuntimeError("Нельзя проверить пустой индекс")

    count = min(sample_size, expected_rows)
    row_ids = random.Random(seed).sample(range(expected_rows), count)
    reconstructed = np.vstack([index.reconstruct(row_id) for row_id in row_ids]).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("В реконструированных FAISS-векторах есть NaN или Inf")
    norms = np.linalg.norm(reconstructed, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if max_norm_error > 2e-3:
        raise RuntimeError(
            f"FAISS-векторы не нормализованы: max |norm - 1|={max_norm_error:.6g}"
        )

    return {
        "verified_at": utc_now(),
        "sample_size": count,
        "max_norm_error": max_norm_error,
        "checks": [
            "ntotal",
            "dimension",
            "random_reconstruct",
            "finite_vectors",
            "l2_norms",
        ],
        "full_corpus_search_skipped": True,
        "full_corpus_search_reason": (
            "Integrity verification must not run exhaustive search over all vectors"
        ),
    }


def build_flat_index(
    args: argparse.Namespace,
    shards: Sequence[Shard],
    state: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not shards:
        raise RuntimeError("Нет embedding-shards")
    faiss = require_faiss()
    dimension = shards[0].dimension
    total_rows = shards[-1].end
    index_path = args.output_dir / args.index_name
    temporary = index_path.with_name(index_path.name + ".tmp")

    if index_path.exists():
        LOG.info("Индекс уже существует, проверяю: %s", index_path)
        index = faiss.read_index(str(index_path))
        verification = verify_index(
            index, total_rows, dimension, args.verify_samples, args.seed
        )
    else:
        LOG.info("Собираю IndexFlatIP: rows=%d, dim=%d", total_rows, dimension)
        index = faiss.IndexFlatIP(dimension)
        for shard in shards:
            LOG.info("Добавляю %s", shard.path.name)
            cached = np.load(shard.path, mmap_mode="r", allow_pickle=False)
            # FAISS normalization is in-place; copy even a float32 read-only memmap.
            vectors = np.array(cached, dtype=np.float32, order="C", copy=True)
            # float16 cache slightly perturbs norms; normalize again before FAISS.
            faiss.normalize_L2(vectors)
            index.add(vectors)
            if int(index.ntotal) != shard.end:
                raise RuntimeError(
                    f"После {shard.path} ntotal={index.ntotal}, ожидалось {shard.end}"
                )

        verification = verify_index(
            index, total_rows, dimension, args.verify_samples, args.seed
        )
        LOG.info("Записываю временный индекс %s", temporary)
        faiss.write_index(index, str(temporary))
        os.replace(temporary, index_path)
        LOG.info("Индекс записан: %s", index_path)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "purpose": "static first-stage GTE candidate retrieval",
        "qrag_compatibility": (
            "Not a fine-tuned Q-RAG action index. Keep the action tower frozen, "
            "use this index only for candidate generation, or rebuild after fine-tuning."
        ),
        "corpus": {
            **state["build_identity"]["corpus"],
            "source_path": str(args.corpus.resolve()),
            "rows": total_rows,
            "faiss_id_mapping": "zero-based JSONL row number",
            "contents_field": "contents",
        },
        "encoder": {
            **state["build_identity"]["encoder"],
            "resolved_revision": state.get("resolved_model_revision"),
            "dimension": dimension,
            "pooling": "model-defined SentenceTransformer pooling (GTE: CLS)",
            "query_encoding_must_match": True,
            "truncation_estimate": state.get("truncation_estimate"),
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
            "metric": "inner_product_on_l2_normalized_vectors",
            "dimension": dimension,
            "ntotal": total_rows,
            "id_map": "implicit contiguous row ids; no IndexIDMap wrapper",
            "size_bytes": index_path.stat().st_size,
        },
        "verification": verification,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "faiss": getattr(faiss, "__version__", None),
            "sentence_transformers": package_version("sentence-transformers"),
            "torch": package_version("torch"),
        },
    }
    atomic_write_json(args.output_dir / MANIFEST_FILE, manifest)
    return index_path, manifest


def verify_existing_index(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.output_dir / MANIFEST_FILE
    if not manifest_path.exists():
        raise RuntimeError(f"Не найден {manifest_path}")
    state_path = args.output_dir / STATE_FILE
    if not state_path.exists():
        raise RuntimeError(f"Не найден {state_path}")
    state = read_json(state_path)
    assert_compatible_state(state, requested_build_identity(args))
    manifest = read_json(manifest_path)
    index_path = args.output_dir / manifest["index"]["file"]
    faiss = require_faiss()
    index = faiss.read_index(str(index_path))
    verification = verify_index(
        index,
        int(manifest["index"]["ntotal"]),
        int(manifest["index"]["dimension"]),
        args.verify_samples,
        args.seed,
    )
    LOG.info("Проверка успешна: %s", json.dumps(verification, ensure_ascii=False))
    return verification


def delete_embedding_shards(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    shards_dir = args.output_dir / SHARDS_DIR
    if not shards_dir.exists():
        return
    LOG.warning("Удаляю embedding cache после успешной сборки: %s", shards_dir)
    shutil.rmtree(shards_dir)
    manifest["embedding_cache"]["retained"] = False
    manifest["embedding_cache"]["deleted_at"] = utc_now()
    atomic_write_json(args.output_dir / MANIFEST_FILE, manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Потоковая сборка нормализованного GTE FlatIP-индекса для Wiki-18"
    )
    parser.add_argument("--corpus", type=Path, required=True, help="wiki-18.jsonl[.gz]")
    parser.add_argument(
        "--tar-member",
        default=None,
        help="Путь JSONL внутри tar или gzip-compressed tar контейнера",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        default="main",
        help="Лучше передать точный commit SHA модели для воспроизводимости",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Одно устройство или список через запятую: cuda:0,cuda:1,...",
    )
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--multi-process-chunk-size",
        type=int,
        default=1_000,
        help="Число текстов в задании worker-процесса при multi-GPU",
    )
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument(
        "--truncation-samples-per-shard",
        type=int,
        default=2_048,
        help="Сколько пассажей каждого shard токенизировать без усечения для оценки",
    )
    parser.add_argument(
        "--cache-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="float32 — точнее; float16 — примерно вдвое меньше временного места",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "embed", "index", "verify"),
        default="all",
    )
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--verify-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing-id",
        action="store_true",
        help="Не требовать поле id; для PeterJinGo/wiki-18-corpus не нужно",
    )
    parser.add_argument(
        "--allow-id-row-mismatch",
        action="store_true",
        help="Не требовать int(item['id']) == zero-based row id",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Требуется текущей реализации Alibaba GTE",
    )
    parser.add_argument(
        "--delete-shards-after-build",
        action="store_true",
        help="Освободить место только после записи и проверки итогового индекса",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = parser.parse_args(argv)

    args.corpus = args.corpus.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.corpus.is_file():
        parser.error(f"Корпус не найден: {args.corpus}")
    try:
        parse_devices(args.device)
    except ValueError as error:
        parser.error(str(error))
    if (
        args.batch_size <= 0
        or args.shard_size <= 0
        or args.max_length <= 0
        or args.multi_process_chunk_size <= 0
    ):
        parser.error(
            "batch-size, shard-size, max-length и multi-process-chunk-size "
            "должны быть положительными"
        )
    if args.truncation_samples_per_shard < 0:
        parser.error("truncation-samples-per-shard не может быть отрицательным")
    if args.verify_samples <= 0:
        parser.error("verify-samples должен быть положительным")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.revision == "main":
        LOG.warning(
            "revision=main изменяемый; для финального эксперимента передайте commit SHA"
        )

    if args.phase == "verify":
        verify_existing_index(args)
        return 0

    # A default rerun after successful cleanup should verify and exit instead of
    # trying to recreate deleted temporary shards.
    manifest_path = args.output_dir / MANIFEST_FILE
    if args.phase == "all" and manifest_path.exists():
        state_path = args.output_dir / STATE_FILE
        if not state_path.exists():
            raise RuntimeError(f"Есть {manifest_path}, но отсутствует {state_path}")
        state = read_json(state_path)
        assert_compatible_state(state, requested_build_identity(args))
        verify_existing_index(args)
        LOG.info("Готовый индекс уже существует; повторная сборка не требуется")
        return 0

    if args.phase in {"all", "embed"}:
        shards, state = encode_corpus(args)
    else:
        shards, state = load_complete_state(args)

    if args.phase == "embed":
        return 0

    _, manifest = build_flat_index(args, shards, state)
    if args.delete_shards_after_build:
        delete_embedding_shards(args, manifest)
    LOG.info("Готово. Manifest: %s", args.output_dir / MANIFEST_FILE)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Прервано пользователем; готовые shards сохранены для продолжения")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Сборка завершилась с ошибкой: %s", error)
        raise
