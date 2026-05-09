import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
# Nhận 1 ảnh
# Nhận list nhiều ảnh
# Không phụ thuộc thư mục
# Dùng được cho Streamlit / FastAPI / Chatbot
# -----> hỗ trợ:
                # handle_data("image.jpg")
                # handle_data(["img1.jpg", "img2.jpg"])
                # handle_data(PIL_image)

# ==============================
# DATASET
# ==============================

class InferenceImageDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        images:
            - str (1 ảnh)
            - list[str] (nhiều ảnh)
            - list[PIL.Image]
        """
        if isinstance(images, str):
            self.image_paths = [images]
        elif isinstance(images, list):
            self.image_paths = images
        elif isinstance(images, Image.Image): # Thêm hỗ trợ cho 1 ảnh PIL trực tiếp
            self.image_paths = [images]
        else:
            raise ValueError(f"images phải là str, list hoặc PIL Image. Nhận được: {type(images)}")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_input = self.image_paths[idx]

        # Nếu là đường dẫn
        if isinstance(img_input, str):
            img_path = img_input
            image = Image.open(img_path).convert("RGB")
        else:
            # Nếu là PIL image
            image = img_input
            img_path = "memory_image"

        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, img_path


# ==============================
# HANDLE DATA (Flexible)
# ==============================

def handle_data(images,
                img_size=300,
                batch_size=1,
                num_workers=0):

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

    dataset = InferenceImageDataset(
        images=images,
        transform=transform
    )

    # pin_memory chỉ hoạt động với CUDA, không dùng trên MPS hoặc CPU
    use_pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return loader


