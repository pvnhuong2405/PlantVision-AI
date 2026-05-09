import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np



# IMG_SIZE = 300
BATCH_SIZE = 4
NUM_WORKERS = 2

DATA_DIR = "D:/Train_CNN/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =========================
# AUGMENTATION
# =========================
# train_transform = A.Compose([
#     A.Resize(IMG_SIZE, IMG_SIZE),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Rotate(limit=15, p=0.5),
#     A.RandomBrightnessContrast(p=0.5),
#     A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#     ToTensorV2(),
    
# ])
train_transform = A.Compose([
    A.Resize(300, 300),

    # GEOMETRY (nhẹ)
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),

    # LIGHTING (mạnh)
    A.RandomBrightnessContrast(
        brightness_limit=0.3,
        contrast_limit=0.3,
        p=0.8
    ),
    A.RandomGamma(
        gamma_limit=(80, 120),
        p=0.5
    ),
    A.CLAHE(
        clip_limit=2.0,
        tile_grid_size=(8, 8),
        p=0.3
    ),

    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])



val_transform = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# =========================
# CUSTOM DATASET WRAPPER
# =========================
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, label

# =========================
# DATASET
# =========================
# train_dataset = AlbumentationsDataset(
#     root=f"{DATA_DIR}/train",
#     transform=train_transform
# )

# val_dataset = AlbumentationsDataset(
#     root=f"{DATA_DIR}/valid",
#     transform=val_transform
# )

# # =========================
# # DATALOADER
# # =========================
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     pin_memory=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=NUM_WORKERS,
#     pin_memory=True
# )

# # =========================
# # TEST LOAD
# # =========================
# images, labels = next(iter(train_loader))
# print("Batch shape:", images.shape)
# print("Labels shape:", labels.shape)

def build_dataloader():
    train_dataset = AlbumentationsDataset(
        root=f"{DATA_DIR}/train",
        transform=train_transform
    )

    val_dataset = AlbumentationsDataset(
        root=f"{DATA_DIR}/valid",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, val_loader

def main():
    train_loader, val_loader = build_dataloader()

    images, labels = next(iter(train_loader))
    print("Train batch:", images.shape, labels.shape)

    images, labels = next(iter(val_loader))
    print("Val batch:", images.shape, labels.shape)
if __name__ == "__main__":
    main()