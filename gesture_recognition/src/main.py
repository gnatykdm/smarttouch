from transform import transform
from ..models.model import GestureNN
from ..models.train import train_model
from utils import set_device, save_model
from torchvision import transforms

device: str = set_device()
model: GestureNN = GestureNN().to(device)
transform: transforms.Compose = transform()

def main(epochs: int, batch_size: int, root: str) -> None:
    print(f"Start training model with -> {epochs} epochs:")
    train_model(model, root, batch_size, epochs)

    print(f"Saving model: model.pth")
    save_model(model, 'model.pth')

if __name__ == '__main__':
    EPOCHS: int = 10000
    BATCH_SIZE: int = 32
    ROOT: str = '../data'

    main(EPOCHS, BATCH_SIZE, ROOT)
