"""Export a revision-pinned Jina ONNX artifact and smoke-test its serving loader.

Use the separate environment described in requirements-onnx.txt. Existing output
folders are never overwritten. Select another --output-dir for a new export.
"""

import argparse
import gc
import platform
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json, environment
from src.retrieval.config import file_digest, load_config, repo_path
from src.retrieval.onnx import (
    patch_position_ids as patch_position_ids,
)  # legacy public import


def choose_preset(requested="auto"):
    if requested != "auto":
        if requested not in ("arm64", "avx2", "avx512", "avx512_vnni"):
            raise ValueError(f"Unknown quantization preset: {requested}")
        return requested
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "avx2"
    raise ValueError(f"Choose a quantization preset explicitly for {machine}")


def export_and_quantize(
    config_path="configs/jina_code_embeddings_0_5b_int8_onnx.json",
    *,
    output_dir=None,
    preset=None,
):
    from huggingface_hub import HfApi, snapshot_download
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.backend import export_dynamic_quantized_onnx_model
    from src.retrieval.encoder import PrePostPipelineEncoder

    config = load_config(config_path)
    if config["backend"] != "onnx":
        raise ValueError("Export requires an ONNX config")
    output = repo_path(output_dir or config["onnx_model_dir"])
    if output.exists():
        raise FileExistsError(f"{output} already exists; select a new --output-dir")
    preset = choose_preset(preset or config.get("quantization_preset", "auto"))
    revision = (
        HfApi().model_info(config["model_name"], revision=config.get("revision")).sha
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".onnx-export-", dir=output.parent))
    try:
        model_kwargs = {"export": True, "provider": "CPUExecutionProvider"}
        if config.get("onnx_intra_op_threads"):
            import onnxruntime

            session_options = onnxruntime.SessionOptions()
            session_options.intra_op_num_threads = config["onnx_intra_op_threads"]
            model_kwargs["session_options"] = session_options
        model = SentenceTransformer(
            config["model_name"],
            revision=revision,
            backend="onnx",
            device="cpu",
            model_kwargs=model_kwargs,
        )
        print(f"Quantizing revision {revision} with preset {preset}", flush=True)
        # AVX2 and ARM64 presets use different signedness; do not guess a qint8 suffix.
        export_dynamic_quantized_onnx_model(
            model, preset, str(staging), file_suffix="quantized"
        )
        graph = staging / "onnx" / "model_quantized.onnx"
        if not graph.is_file():
            raise FileNotFoundError(f"Exporter did not produce expected graph: {graph}")
        target = staging / config["onnx_file_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target != graph:
            graph.rename(target)
        snapshot = Path(
            snapshot_download(
                config["model_name"],
                revision=revision,
                allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"],
            )
        )
        for source in snapshot.rglob("*"):
            if source.is_file() and source.suffix in (
                ".json",
                ".txt",
                ".model",
                ".tiktoken",
                ".jinja",
            ):
                destination = staging / source.relative_to(snapshot)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        # Persist the resolved tokenizer configuration, even if absent on the Hub.
        model[0].tokenizer.save_pretrained(str(staging))
        # Release the full-precision ORT session before loading the quantized probe.
        del model
        gc.collect()
        serving = {
            **config,
            "onnx_model_dir": str(staging),
            "revision": revision,
            "quantization_preset": preset,
            "cache_embeddings": False,
        }
        with tempfile.TemporaryDirectory(prefix="prism-export-check-") as temporary:
            probe_config = Path(temporary) / "config.json"
            atomic_json(probe_config, serving)
            probe = PrePostPipelineEncoder(str(probe_config))
            probe.encode_texts(["Sort a list."], kind="query", batch_size=1)
            probe.encode_texts(
                [
                    "Sort a list.",
                    "Find the longest increasing subsequence in a list of integers.",
                ],
                kind="query",
                batch_size=2,
            )
        serving["onnx_model_dir"] = str(output)
        atomic_json(staging / "serving_config.json", serving)
        atomic_json(
            staging / "export_manifest.json",
            {
                "model": config["model_name"],
                "revision": revision,
                "quantization_preset": preset,
                "environment": environment(),
                "smoke_test": "single and mixed-length batch produced finite embeddings",
                "files": {
                    str(p.relative_to(staging)): file_digest(p)
                    for p in sorted(staging.rglob("*"))
                    if p.is_file()
                },
            },
        )
        staging.rename(output)
        return output
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/jina_code_embeddings_0_5b_int8_onnx.json"
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--preset", choices=["auto", "arm64", "avx2", "avx512", "avx512_vnni"]
    )
    args = parser.parse_args(argv)
    output = export_and_quantize(
        args.config, output_dir=args.output_dir, preset=args.preset
    )
    print(f"Exported and smoke-tested: {output}; use {output / 'serving_config.json'}")


if __name__ == "__main__":
    main()
