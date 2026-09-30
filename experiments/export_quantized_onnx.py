"""Export jina-code-embeddings-0.5b to a dynamically-quantized INT8 ONNX model
for fast CPU query-time serving.

This is a profiling/serving artifact, not part of the official APPS benchmark
pipeline (that remains the full-precision model evaluated on the full corpus
via experiments/run_eval.py). See docs/progress.md for the full writeup:
unquantized vs. quantized single-query CPU latency, and a scoped accuracy
sanity check.

Requires a SEPARATE virtual environment from the project's main .venv —
`sentence-transformers[onnx]` pulls in `optimum` and `onnxruntime`, which are
not part of requirements.txt and were deliberately kept out of the main
environment. Set up an isolated venv before running this:

    python3.11 -m venv .venv-onnx-test
    .venv-onnx-test/bin/pip install "optimum[onnxruntime]" sentence-transformers \
        transformers mteb torch
    .venv-onnx-test/bin/python experiments/export_quantized_onnx.py

Output layout (self-contained, loadable without touching the shared HF
cache): <output_dir>/onnx/model_qint8_arm64.onnx plus the tokenizer/config
files copied alongside it, so the directory can be loaded directly via:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(
        output_dir,
        backend="onnx",
        model_kwargs={"file_name": "onnx/model_qint8_arm64.onnx", "provider": "CPUExecutionProvider"},
    )
    # See patch_position_ids() below — required at inference time too.

The exported ONNX graph requires `position_ids` as an explicit input (Qwen2
uses RoPE), which sentence-transformers' tokenizer does not produce and the
original PyTorch model computed internally when absent. That fallback isn't
preserved by tracing, so callers must patch it back in (see
patch_position_ids), or every encode() call raises:
    ValueError: Input position_ids is required by model but not provided.
"""

import shutil
from pathlib import Path

MODEL_ID = "jinaai/jina-code-embeddings-0.5b"
OUTPUT_DIR = Path(".cache/onnx_models/jina-code-int8")
QUANTIZATION_PRESET = "arm64"  # use "avx2"/"avx512"/"avx512_vnni" on x86 instead


def export_and_quantize(output_dir: Path = OUTPUT_DIR) -> None:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.backend import export_dynamic_quantized_onnx_model

    print("Loading + exporting to base ONNX...")
    model = SentenceTransformer(MODEL_ID, backend="onnx", model_kwargs={"export": True})

    print(f"Quantizing to INT8 ({QUANTIZATION_PRESET})...")
    export_dynamic_quantized_onnx_model(model, QUANTIZATION_PRESET, str(output_dir))

    _copy_companion_files(output_dir)
    print(f"Done. Quantized model ready at: {output_dir}")


def _copy_companion_files(output_dir: Path) -> None:
    """Copy tokenizer/config files (everything but the safetensors weights)
    from the model's HF cache snapshot alongside the exported ONNX file, so
    output_dir is self-contained and loadable on its own.
    export_dynamic_quantized_onnx_model only saves the raw .onnx weight file
    into output_dir by design (it's meant for a push_to_hub workflow where
    the rest of the repo already exists).
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(snapshot_download(MODEL_ID))
    for item in snapshot_dir.iterdir():
        if item.name == "model.safetensors":
            continue
        destination = output_dir / item.name
        if destination.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, destination, symlinks=False)
        else:
            shutil.copy(item, destination, follow_symlinks=True)


def patch_position_ids(sentence_transformer_model) -> None:
    """Monkeypatch the ONNX model's forward to compute `position_ids` from
    `attention_mask` when absent, mirroring what the original PyTorch/Qwen2
    model did internally. Call this once after loading the ONNX model and
    before calling .encode().
    """
    auto_model = sentence_transformer_model[0].auto_model
    original_forward = auto_model.forward

    def patched_forward(input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None and attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs
        )

    auto_model.forward = patched_forward


if __name__ == "__main__":
    export_and_quantize()
