"""Validated, repository-relative encoder configuration."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIELDS = {
    "model_name",
    "revision",
    "code_revision",
    "backend",
    "device",
    "dtype",
    "query_prefix",
    "document_prefix",
    "max_seq_length",
    "batch_size",
    "trust_remote_code",
    "cache_embeddings",
    "cache_dir",
    "notes",
    "onnx_model_dir",
    "onnx_file_name",
    "provider",
    "quantization_preset",
    "onnx_position_ids",
    "onnx_intra_op_threads",
}


def repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def positive_int(value):
    if isinstance(value, bool):
        raise ValueError("Expected a positive integer")
    result = int(value)
    if result <= 0 or str(result) != str(value):
        raise ValueError("Expected a positive integer")
    return result


def load_config(path, **overrides):
    with repo_path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    config.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(config) - FIELDS
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    if (
        not isinstance(config.get("model_name"), str)
        or not config["model_name"].strip()
    ):
        raise ValueError("model_name must be a nonempty string")
    for key in ("query_prefix", "document_prefix"):
        config.setdefault(key, "")
        if not isinstance(config[key], str):
            raise ValueError(f"{key} must be a string")
    for key in ("cache_embeddings", "trust_remote_code", "onnx_position_ids"):
        config.setdefault(key, False)
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be a boolean")
    config["batch_size"] = positive_int(config.get("batch_size", 32))
    if config.get("max_seq_length") is not None:
        config["max_seq_length"] = positive_int(config["max_seq_length"])
    if config.get("dtype") not in (None, "float32", "float16", "bfloat16"):
        raise ValueError("dtype must be float32, float16, or bfloat16")
    for key in ("revision", "code_revision", "device"):
        if config.get(key) is not None and (
            not isinstance(config[key], str) or not config[key].strip()
        ):
            raise ValueError(f"{key} must be a nonempty string")
    config.setdefault("backend", "torch")
    if config["backend"] not in ("torch", "onnx"):
        raise ValueError("backend must be torch or onnx")
    if config["backend"] == "onnx":
        if "onnx_intra_op_threads" in config:
            config["onnx_intra_op_threads"] = positive_int(
                config["onnx_intra_op_threads"]
            )
        for key in ("onnx_model_dir", "onnx_file_name"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise ValueError(f"ONNX config requires {key}")
        filename = Path(config["onnx_file_name"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("onnx_file_name must be relative to onnx_model_dir")
        config.setdefault("provider", "CPUExecutionProvider")
        if (
            config["provider"] != "CPUExecutionProvider"
            or config.get("device", "cpu") != "cpu"
        ):
            raise ValueError(
                "ONNX serving currently supports CPUExecutionProvider on cpu"
            )
        if config.get("dtype") is not None:
            raise ValueError("ONNX precision is fixed by its artifact; remove dtype")
        config["device"] = "cpu"
    elif (
        any(
            k in config
            for k in (
                "onnx_model_dir",
                "onnx_file_name",
                "provider",
                "quantization_preset",
                "onnx_intra_op_threads",
            )
        )
        or config["onnx_position_ids"]
    ):
        raise ValueError("ONNX fields require backend=onnx")
    config["cache_dir"] = str(repo_path(config.get("cache_dir", ".cache/embeddings")))
    return config


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprint():
    paths = sorted(
        p
        for folder in ("src", "experiments", "scripts")
        for p in (REPO_ROOT / folder).rglob("*.py")
    )
    return fingerprint({str(p.relative_to(REPO_ROOT)): file_digest(p) for p in paths})
