import torch
from model import GestureNN
from torch.optim import Adam
from torch import DataLoader
from torchvision import transforms
from ..src.utils import set_device
from ..src.dataset import GestureDataSet
from ..src.utils import save_model
from typing import Any

device: str = set_device()
learning_rate: float = 1e-3

def train_model(model: GestureNN, path: str, batch_size: int, transform: transforms.Compose, epochs: int) -> None:
    try:
        train_set: GestureDataSet = GestureDataSet(path, transform)
        train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        optimizer: Adam = Adam(model.parameters(), lr=learning_rate)
        criterion: Any = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for labels, images in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")

            if epoch % 100 == 0:
                save_model(model, f"../models/model_{epoch}.pth")
    except Exception as e:
        raise ValueError(f"Error in train_model -> {e}")