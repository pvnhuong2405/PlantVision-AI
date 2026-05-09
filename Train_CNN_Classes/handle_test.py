import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class InferenceImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, img_path


def handle_data(path, img_size=300, batch_size=4, num_workers=0):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    test_dataset = InferenceImageDataset(
        image_dir=path,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,   # ← QUAN TRỌNG
        pin_memory=True
    )

    return test_loader


def main():
    img_loader = handle_data(
        "D:/Project_PlantDisease/Train_CNN_Classes/Dataset_new/test",
        batch_size=4,
        num_workers=0  
    )

    images, paths = next(iter(img_loader))

    print(images.shape)   # [B, 3, 300, 300]
    print(paths)


if __name__ == "__main__":
    main()
