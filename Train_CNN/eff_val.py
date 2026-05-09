import eff_train as tr
import torch


# def validate(model, loader, criterion):
#     model.eval()
#     total_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images = images.to(tr.device)
#             labels = labels.to(tr.device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     acc = correct / total
#     return total_loss / len(loader), acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return total_loss / len(loader), acc
