"""Local model doubles; unit tests never import or download the ML stack."""

import importlib
import json
import sys
import types
from enum import Enum
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def encoder_factory(monkeypatch, tmp_path):
    class PromptType(Enum):
        query = "query"
        document = "document"

    class Meta:
        @classmethod
        def create_empty(cls, overwrites):
            return SimpleNamespace(**overwrites)

    class FakeModel:
        instances = []

        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs
            self.max_seq_length = 256
            self.dtype = "torch.float32"
            self.calls = []
            self.auto_model = SimpleNamespace(
                config=SimpleNamespace(_commit_hash="a" * 40), forward=lambda **kw: kw
            )
            self.instances.append(self)

        def __getitem__(self, index):
            return SimpleNamespace(
                auto_model=self.auto_model,
                tokenizer=SimpleNamespace(save_pretrained=lambda path: None),
            )

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            rows = np.array(
                [
                    [len(t), self.max_seq_length, sum(map(ord, t)) % 97 + 1]
                    for t in texts
                ],
                dtype=np.float32,
            )
            rows = rows[:, : kwargs.get("truncate_dim", 3)]
            if kwargs.get("normalize_embeddings"):
                rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            return rows

    modules = {
        "mteb": {},
        "mteb.models": {},
        "mteb.models.abs_encoder": {"AbsEncoder": object},
        "mteb.models.model_meta": {"ModelMeta": Meta},
        "mteb.types": {"PromptType": PromptType},
        "sentence_transformers": {"SentenceTransformer": FakeModel},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "src.retrieval.encoder", raising=False)
    encoder = importlib.import_module("src.retrieval.encoder")

    def factory(**overrides):
        config = {
            "model_name": "test/model",
            "device": "cpu",
            "max_seq_length": 256,
            "cache_dir": str(tmp_path / "cache"),
            **overrides,
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        return encoder.PrePostPipelineEncoder(str(path))

    factory.module = encoder
    factory.model_class = FakeModel
    factory.prompt_type = PromptType
    yield factory
    # Do not leak a module inheriting our fake AbsEncoder into integration tests.
    sys.modules.pop("src.retrieval.encoder", None)
