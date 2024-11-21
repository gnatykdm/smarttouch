import torch
from torch.optim import Adam
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from .model import GestureNN
from ..src.dataset import GestureDataSet
from ..src import utils


learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model: GestureNN, epochs: int, batch_size: int, path: str) -> GestureNN:
    model: Module = model.to(device)
    model.train()

    optimizer: Adam = Adam(model.parameters(), lr=learning_rate)
    criterion: CrossEntropyLoss = CrossEntropyLoss()

    train_dataset: GestureDataSet = GestureDataSet(path)
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(epochs):
            r_loss: float = 0.00

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                output: torch.Tensor = model(images)
                loss: torch.Tensor = criterion(output, labels)

                loss.backward()
                optimizer.step()

                r_loss += loss.item()

            r_loss /= len(train_loader)
            if epoch % 10 == 0:
                print(f"EPOCH: {epoch} | LOSS: {r_loss:4f}")

            if epoch % 100 == 0:
                utils.save_model(model, f"saved/trained_model_{epoch}.pth")

    except Exception as e:
        print(f"Error -> {e}")

    utils.save_model(model, "final_model.pth")
    return model
