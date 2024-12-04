import torch
import onnx

from gesture_recognition.src.dataset import GestureDataSet
from gesture_recognition.src.utils import set_device
from gesture_recognition.models.model import GestureNN
from gesture_recognition.src.utils import timer

@timer
def export_model(model: GestureNN, model_name: str) -> bool:
    if model is None or model_name is None:
        raise ValueError("Model or model_name can't be null")

    DEVICE: str = set_device()
    model = model.to(DEVICE)
    model_input = GestureDataSet().to(DEVICE)[0].unsqueeze(0)

    torch.onnx.export(model, model_input, f"{model_name}.onnx", opset_version=11)

    onnx_model = onnx.load(f"{model_name}.onnx")
    try:
        onnx.checker.check_model(onnx_model)
        print(f"Model successfully exported to ONNX format: {model_name}.onnx")
        return True
    except onnx.checker.ValidationError as e:
        print(f"Problem with exporting model to ONNX format: {e}")
        return False
