from torchvision import transforms as T

def transform() -> T.Compose:
    return T.Compose([
        T.Resize(64, 64),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])