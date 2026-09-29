from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs


def select_device() -> str:
    """Pick the best available device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name: str | None, device: str) -> tuple["torch.dtype | None", str | None]:
    """Map a config dtype name to a torch dtype, avoiding bf16 on MPS.

    PyTorch's MPS backend has incomplete/unstable bfloat16 support, so a
    bfloat16 request is downgraded to float16 on that device. Returns
    (torch_dtype_or_None, actual_dtype_name_used_or_None).
    """
    if dtype_name is None:
        return None, None

    import torch

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_name]
    actual_name = dtype_name

    if device == "mps" and dtype is torch.bfloat16:
        dtype = torch.float16
        actual_name = "float16"

    return dtype, actual_name


class PrePostPipelineEncoder(AbsEncoder):
    """SentenceTransformer-backed encoder driven by a JSON config.

    Every query and document is routed through `preprocess_query` /
    `preprocess_document` before encoding and through `postprocess` after
    encoding. All three are no-ops for now; later phases fill them in.
    """

    def __init__(self, config_path: str) -> None:
        with open(config_path) as f:
            config = json.load(f)

        self.config = config
        self.model_name: str = config["model_name"]
        self.query_prefix: str = config.get("query_prefix", "")
        self.document_prefix: str = config.get("document_prefix", "")
        self.max_seq_length = config.get("max_seq_length")
        self.default_batch_size: int = config.get("batch_size", 32)
        self.trust_remote_code: bool = config.get("trust_remote_code", False)
        self.cache_embeddings: bool = config.get("cache_embeddings", False)
        self.cache_dir = Path(config.get("cache_dir", ".cache/embeddings"))

        self.device: str = config.get("device") or select_device()
        torch_dtype, self.dtype_used = resolve_dtype(config.get("dtype"), self.device)

        from sentence_transformers import SentenceTransformer

        model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else None

        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=model_kwargs,
        )
        if self.max_seq_length is not None:
            self.model.max_seq_length = self.max_seq_length

        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites={"name": self.model_name, "loader": type(self)}
        )

    def preprocess_query(self, text: str) -> str:
        return text

    def preprocess_document(self, text: str) -> str:
        return text

    def postprocess(self, embeddings: Array) -> Array:
        return embeddings

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _encode_texts(self, texts: Sequence[str], batch_size: int) -> Array:
        if not self.cache_embeddings:
            return np.asarray(
                self.model.encode(
                    list(texts),
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        embeddings: list[Any] = [None] * len(texts)
        pending_texts: list[str] = []
        pending_indices: list[int] = []

        for i, text in enumerate(texts):
            path = self._cache_path(text)
            if path.exists():
                embeddings[i] = np.load(path)
            else:
                pending_texts.append(text)
                pending_indices.append(i)

        if pending_texts:
            new_embeddings = self.model.encode(
                pending_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for idx, embedding in zip(pending_indices, new_embeddings, strict=True):
                embeddings[idx] = embedding
                np.save(self._cache_path(texts[idx]), embedding)

        return np.stack(embeddings)

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        texts = [text for batch in inputs for text in batch["text"]]

        if prompt_type == PromptType.query:
            prefix = self.query_prefix
            processed = [self.preprocess_query(text) for text in texts]
        else:
            prefix = self.document_prefix
            processed = [self.preprocess_document(text) for text in texts]

        prefixed = [prefix + text for text in processed]

        batch_size = kwargs.get("batch_size", self.default_batch_size)
        embeddings = self._encode_texts(prefixed, batch_size=batch_size)

        return self.postprocess(embeddings)
