import torch

ckpt_path = "D:/Project_PlantDisease/Train_CNN_Classes/weight/best.pt"

checkpoint = torch.load(ckpt_path, map_location="cpu")

state_dict = checkpoint["model"]

print("Classifier weight shape:",
      state_dict["classifier.weight"].shape)

print("Classifier bias shape:",
      state_dict["classifier.bias"].shape)
