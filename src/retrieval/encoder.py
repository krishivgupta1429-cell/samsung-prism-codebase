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

    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import Array, BatchedInput, EncodeKwargs


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

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name,
            device="cpu",
            trust_remote_code=self.trust_remote_code,
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
