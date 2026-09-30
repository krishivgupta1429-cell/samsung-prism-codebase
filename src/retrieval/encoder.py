from __future__ import annotations

import os
import importlib.metadata
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

from .config import (
    file_digest,
    fingerprint,
    load_config,
    positive_int,
    repo_path,
    source_fingerprint,
)
from .onnx import patch_position_ids


def select_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name, device):
    if dtype_name is None:
        return None, None
    import torch

    actual = (
        "float16"
        if device.split(":")[0] == "mps" and dtype_name == "bfloat16"
        else dtype_name
    )
    return getattr(torch, actual), actual


class PrePostPipelineEncoder(AbsEncoder):
    """Configured encoder shared by evaluation, profiling, and CPU ONNX inference."""

    def __init__(self, config_path, *, device=None, cache_embeddings=None):
        self.config = load_config(
            config_path, device=device, cache_embeddings=cache_embeddings
        )
        config = self.config
        self.model_name = config["model_name"]
        self.query_prefix = config["query_prefix"]
        self.document_prefix = config["document_prefix"]
        self.default_batch_size = config["batch_size"]
        self.cache_embeddings = config["cache_embeddings"]
        self.device = config.get("device") or select_device()
        dtype, self.dtype_used = resolve_dtype(config.get("dtype"), self.device)
        model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        source = self.model_name
        artifact_hashes = None
        if config["backend"] == "onnx":
            directory = repo_path(config["onnx_model_dir"])
            graph = directory / config["onnx_file_name"]
            if not graph.is_file():
                raise FileNotFoundError(
                    f"Missing ONNX artifact: {graph}. Run experiments/export_quantized_onnx.py first."
                )
            source = str(directory)
            # Includes tokenizer, pooling, external tensor data, and the graph.
            artifact_hashes = {
                str(p.relative_to(directory)): file_digest(p)
                for p in sorted(directory.rglob("*"))
                if p.is_file()
            }
            model_kwargs.update(
                file_name=config["onnx_file_name"],
                provider=config["provider"],
                export=False,
            )
            if config.get("onnx_intra_op_threads"):
                import onnxruntime

                session_options = onnxruntime.SessionOptions()
                session_options.intra_op_num_threads = config["onnx_intra_op_threads"]
                model_kwargs["session_options"] = session_options
        elif config.get("code_revision"):
            model_kwargs["code_revision"] = config["code_revision"]

        from sentence_transformers import SentenceTransformer

        code_kwargs = (
            {"code_revision": config["code_revision"]}
            if config.get("code_revision")
            else None
        )
        self.model = SentenceTransformer(
            source,
            device=self.device,
            backend=config["backend"],
            trust_remote_code=config["trust_remote_code"],
            revision=config.get("revision") if config["backend"] == "torch" else None,
            model_kwargs=model_kwargs or None,
            config_kwargs=code_kwargs,
            tokenizer_kwargs=code_kwargs,
        )
        if config["backend"] == "onnx" and config["onnx_position_ids"]:
            patch_position_ids(self.model)
        if config.get("max_seq_length") is not None:
            self.model.max_seq_length = config["max_seq_length"]
        self.max_seq_length = self.model.max_seq_length
        get_dimension = getattr(
            self.model,
            "get_embedding_dimension",
            self.model.get_sentence_embedding_dimension,
        )
        self.embedding_dimension = get_dimension()
        if not self.embedding_dimension:
            raise ValueError("Model must report its sentence embedding dimension")
        self.embedding_dimension = int(self.embedding_dimension)
        model_config = getattr(
            getattr(self.model[0], "auto_model", None), "config", None
        )
        self.revision = getattr(model_config, "_commit_hash", None) or config.get(
            "revision"
        )
        remote_module = type(self.model[0].auto_model).__module__
        remote_revisions = re.findall(
            r"(?:^|\.)([a-fA-F0-9]{40})(?:\.|$)", remote_module
        )
        code_revision = (
            remote_revisions[0] if remote_revisions else config.get("code_revision")
        )
        if self.dtype_used is None and config["backend"] == "torch":
            self.dtype_used = str(self.model.dtype).removeprefix("torch.")
        if self.cache_embeddings and config["backend"] == "torch":
            if not self.revision or not re.fullmatch(r"[a-fA-F0-9]{40}", self.revision):
                raise ValueError(
                    "Embedding caching requires a resolved immutable model revision"
                )
            if config["trust_remote_code"] and not re.fullmatch(
                r"[a-fA-F0-9]{40}", config.get("code_revision", "")
            ):
                raise ValueError(
                    "Caching remote-code models also requires an immutable code_revision"
                )
        identity = {
            k: v
            for k, v in config.items()
            if k not in {"cache_dir", "cache_embeddings", "notes", "batch_size"}
        }
        libraries = {}
        for name in (
            "torch",
            "transformers",
            "sentence-transformers",
            "numpy",
            "onnxruntime",
        ):
            try:
                libraries[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        identity.update(
            revision=self.revision,
            code_revision=code_revision,
            device=self.device,
            dtype=self.dtype_used,
            max_seq_length=self.max_seq_length,
            embedding_dimension=self.embedding_dimension,
            artifacts=artifact_hashes,
            libraries=libraries,
            source_fingerprint=source_fingerprint(),
        )
        self.pipeline_fingerprint = fingerprint(identity)
        self.resolved_config = {
            **config,
            **identity,
            "pipeline_fingerprint": self.pipeline_fingerprint,
        }
        self.cache_dir = Path(config["cache_dir"]) / self.pipeline_fingerprint
        self.encode_seconds = defaultdict(float)
        # The instance is supplied directly; do not advertise an incompatible loader.
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={
                "name": self.model_name,
                "revision": self.revision or self.pipeline_fingerprint,
                "loader": None,
                "embed_dim": self.embedding_dimension,
                "experiment_kwargs": {"pipeline": self.pipeline_fingerprint},
            }
        )

    def preprocess_query(self, text):
        return text

    def preprocess_document(self, text):
        return text

    def postprocess(self, embeddings):
        return embeddings

    def _cache_path(self, text, options=None):
        return self.cache_dir / f"{fingerprint([text, options or {}])}.npy"

    def _validate_embeddings(self, embeddings, count, dimension):
        array = np.asarray(embeddings)
        if (
            array.shape != (count, dimension)
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                f"Expected finite floating embeddings with shape {(count, dimension)}, got {array.shape} / {array.dtype}"
            )
        return array

    def _encode_texts(self, texts, batch_size, **options):
        dimension = options.get("truncate_dim") or self.embedding_dimension
        if not texts:
            return np.empty((0, dimension), dtype=np.float32)

        def encode(batch):
            return self._validate_embeddings(
                self.model.encode(
                    list(batch),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    prompt="",
                    **options,
                ),
                len(batch),
                dimension,
            )

        if not self.cache_embeddings:
            return encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        by_text = {}
        pending = []
        for text in dict.fromkeys(texts):
            try:
                vector = np.load(self._cache_path(text, options), allow_pickle=False)
                by_text[text] = self._validate_embeddings(
                    vector[None, :], 1, dimension
                )[0]
            except (OSError, ValueError, EOFError, IndexError):
                pending.append(text)
        if pending:
            for text, vector in zip(pending, encode(pending), strict=True):
                by_text[text] = vector
                fd, temporary = tempfile.mkstemp(dir=self.cache_dir, prefix=".tmp-")
                try:
                    with os.fdopen(fd, "wb") as stream:
                        np.save(stream, vector, allow_pickle=False)
                    os.replace(temporary, self._cache_path(text, options))
                finally:
                    Path(temporary).unlink(missing_ok=True)
        return np.stack([by_text[text] for text in texts])

    def encode_texts(
        self,
        texts,
        *,
        kind,
        batch_size=None,
        normalize_embeddings=False,
        truncate_dim=None,
    ):
        if kind not in ("query", "document"):
            raise ValueError("kind must be query or document")
        if not isinstance(normalize_embeddings, bool):
            raise ValueError("normalize_embeddings must be a boolean")
        if truncate_dim is not None:
            truncate_dim = positive_int(truncate_dim)
            if truncate_dim > self.embedding_dimension:
                raise ValueError("truncate_dim exceeds the model embedding dimension")
        batch_size = positive_int(
            batch_size if batch_size is not None else self.default_batch_size
        )
        options = {"normalize_embeddings": normalize_embeddings}
        if truncate_dim is not None:
            options["truncate_dim"] = truncate_dim
        start = time.perf_counter()
        preprocess = (
            self.preprocess_query if kind == "query" else self.preprocess_document
        )
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        prepared = [prefix + preprocess(text) for text in texts]
        output = self.postprocess(self._encode_texts(prepared, batch_size, **options))
        output = self._validate_embeddings(
            output, len(prepared), truncate_dim or self.embedding_dimension
        )
        self.encode_seconds[kind] += time.perf_counter() - start
        return output

    def encode(
        self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs
    ):
        if prompt_type not in (PromptType.query, PromptType.document):
            raise ValueError(
                "Retrieval encoding requires an explicit query or document prompt_type"
            )
        # MTEB owns the loader; these options do not change embedding values.
        for key in ("num_workers", "show_progress_bar"):
            kwargs.pop(key, None)
        if kwargs.pop("precision", "float32") != "float32":
            raise ValueError(
                "Integer embedding quantization is unsupported; use an ONNX model for weight quantization"
            )
        unknown = set(kwargs) - {"batch_size", "normalize_embeddings", "truncate_dim"}
        if unknown:
            raise ValueError(f"Unsupported encoding options: {sorted(unknown)}")
        chunks = [
            self.encode_texts(batch["text"], kind=prompt_type.value, **kwargs)
            for batch in inputs
        ]
        if not chunks:
            return np.empty(
                (0, kwargs.get("truncate_dim") or self.embedding_dimension),
                dtype=np.float32,
            )
        return np.concatenate(chunks, axis=0)
