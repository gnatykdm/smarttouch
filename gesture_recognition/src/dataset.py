import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from typing import List, Tuple
from torchvision.transforms import transforms as T

class GestureDataSet(Dataset):
    def __init__(self, path: str, transform: transforms.Compose):
        self.path = path
        self.transform = transform
        self.images: List[str] = []
        self.labels: List[int] = []

        for label, root_dir in enumerate(os.listdir(path)):
            class_path = os.path.join(path, root_dir)
            if os.path.isdir(class_path):
                for file in os.listdir(class_path):
                    if file.endswith('.jpg') or file.endswith('.png'):
                        self.images.append(os.path.join(class_path, file))
                        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        image = Image.open(self.images[index])
        label = self.labels[index]
        return self.transform(image), label
