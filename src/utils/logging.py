from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

from .io import ensure_dir


def setup_logger(name: str, log_dir: str | Path = "outputs/logs") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
    }
    try:
        import torch
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            info["torch_cuda_current_device"] = torch.cuda.current_device()
            info["torch_cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


def log_environment(logger: logging.Logger) -> dict[str, Any]:
    info = collect_environment()
    logger.info("environment=%s", json.dumps(info, ensure_ascii=False))
    return info
