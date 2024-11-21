import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class GestureDataSet(Dataset):
    def __init__(self, path: str, transform: transforms.Compose):
        self.path = path
