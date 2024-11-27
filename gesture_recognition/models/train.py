import torch
from model import GestureNN
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from src.utils import set_device, save_model, timer  
from src.dataset import GestureDataSet  
from typing import Any
import os

device: str = set_device()
learning_rate: float = 1e-3

transform: T.Compose = T.Compose([
    T.Resize((256, 256)),  
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

@timer
def train_model(model: GestureNN, path: str, batch_size: int, epochs: int) -> None:
    try:
        train_set: GestureDataSet = GestureDataSet(path, transform)
        train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        optimizer: Adam = Adam(model.parameters(), lr=learning_rate)
        criterion: Any = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()  
            for images, labels in train_loader: 
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")

            if epoch % 100 == 0:
                os.makedirs("../models", exist_ok=True)  
                save_model(model, f"../models/model_{epoch}.pth")

    except Exception as e:
        raise ValueError(f"Error in train_model -> {e}")

