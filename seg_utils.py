import torch, numpy as np
from typing import Dict, Any
from .metrics import iou_score

def train_segmentation_epoch(model, dataloader, criterion, optimizer, device) -> Dict[str,float]:
    model.train()
    running_loss, total = 0.0, 0
    for images,masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()*images.size(0)
        total += images.size(0)
    return {"loss": running_loss/total}

def evaluate_segmentation(model, dataloader, device, num_classes: int) -> Dict[str,Any]:
    model.eval()
    all_ious = []
    with torch.no_grad():
        for images,masks in dataloader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            masks = masks.numpy()
            for p,t in zip(preds,masks):
                all_ious.append(iou_score(p,t,classes=num_classes))
    all_ious = np.stack(all_ious,0)
    return {"per_class_iou": all_ious.mean(0), "mean_iou": all_ious.mean()}
