import os
import torch
import time
from typing import Any, List

def save_model(model: torch.nn.Module, path: str) -> None:
    if model is None or path is None:
        raise ValueError("model or path is None")
    torch.save(model.load_state_dict(), path)

def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if model is None or path is None:
        raise ValueError("model or path is None")
    model.load_state_dict(torch.load(path))
    return model

def set_device() -> str:
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def timer(func: Any) -> Any:
    def wrapper(*args, **kwargs):
        start: float = time.time()
        end: float = time.time()
        print(f"Function -> {func.__name__} was work -> {end - start:.6f}s")
    return wrapper


def show_all_models(path: str) -> None:
    if path is None:
        raise ValueError("File Path can't be empty")

    all: List[str] = ""
    for model in os.path.dirname(path):
        all += model
    return model
