from types import SimpleNamespace

import numpy as np

from experiments.export_quantized_onnx import choose_preset
from src.retrieval.onnx import patch_position_ids


class Tensor(np.ndarray):
    def long(self):
        return self.astype(np.int64)

    def masked_fill_(self, mask, value):
        self[mask] = value


def test_position_ids_left_right_padding_explicit_ids_and_idempotence():
    auto = SimpleNamespace(forward=lambda **kwargs: kwargs)
    model = [SimpleNamespace(auto_model=auto)]
    patch_position_ids(model)
    original = auto.forward
    patch_position_ids(model)
    assert auto.forward is original
    mask = np.array([[0, 0, 1, 1], [1, 1, 1, 0]]).view(Tensor)
    result = auto.forward(attention_mask=mask)
    np.testing.assert_array_equal(result["position_ids"], [[1, 1, 0, 1], [0, 1, 2, 1]])
    explicit = np.array([[7, 8, 9, 10]])
    assert auto.forward(position_ids=explicit)["position_ids"] is explicit


def test_architecture_preset(monkeypatch):
    monkeypatch.setattr(
        "experiments.export_quantized_onnx.platform.machine", lambda: "x86_64"
    )
    assert choose_preset() == "avx2"
    monkeypatch.setattr(
        "experiments.export_quantized_onnx.platform.machine", lambda: "aarch64"
    )
    assert choose_preset() == "arm64"
    assert choose_preset("avx512_vnni") == "avx512_vnni"


def test_export_uses_stable_filename_and_preserves_existing_output(
    encoder_factory, monkeypatch, tmp_path
):
    import json
    import sys
    import types
    from pathlib import Path
    import pytest
    from experiments.export_quantized_onnx import export_and_quantize

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda: SimpleNamespace(
        model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40)
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "modules.json").write_text("[]")
    hub.snapshot_download = lambda *args, **kwargs: str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    backend = types.ModuleType("sentence_transformers.backend")

    def export(model, preset, destination, file_suffix=None):
        # Mirrors the real preset-dependent default that broke AVX2 exports.
        suffix = file_suffix or "quint8_avx2"
        path = Path(destination) / "onnx" / f"model_{suffix}.onnx"
        path.parent.mkdir()
        path.write_bytes(b"graph")

    backend.export_dynamic_quantized_onnx_model = export
    monkeypatch.setitem(sys.modules, "sentence_transformers.backend", backend)
    config = tmp_path / "export.json"
    output = tmp_path / "published"
    config.write_text(
        json.dumps(
            {
                "model_name": "test/model",
                "backend": "onnx",
                "onnx_model_dir": str(output),
                "onnx_file_name": "onnx/model_quantized.onnx",
            }
        )
    )
    export_and_quantize(config, preset="avx2")
    assert (output / "onnx/model_quantized.onnx").read_bytes() == b"graph"
    serving = json.loads((output / "serving_config.json").read_text())
    assert serving["revision"] == "a" * 40 and serving["onnx_model_dir"] == str(output)
    with pytest.raises(FileExistsError):
        export_and_quantize(config, preset="avx2")
    assert not list(tmp_path.glob(".onnx-export-*"))
