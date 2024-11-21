import torch

def save_model(model: torch.nn.Module, path: str) -> None:
    if model is None or path is None:
        raise ValueError("model or path is None")
    torch.save(model.load_state_dict(), path)

def load_model(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if model is None or path is None:
        raise ValueError("model or path is None")
    model.load_state_dict(torch.load(path))
    return model