from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 468) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def get_hardware_info() -> dict[str, str]:
    """Return basic hardware information for the report."""
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
        }

    return {
        "device": "cpu",
        "gpu": "None",
    }


def save_json(data: dict, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)