import numpy as np
import torch
from typing import Tuple, List

def accuracy(outputs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int] = (1,)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res

def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm

def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, classes=None, eps=1e-7) -> np.ndarray:
    if pred_mask.ndim == 2:
        pred_mask, true_mask = pred_mask[None], true_mask[None]
    if classes is None:
        classes = int(max(pred_mask.max(), true_mask.max())+1)
    ious = np.zeros((classes,), dtype=np.float32)
    for c in range(classes):
        inter = ((pred_mask==c) & (true_mask==c)).sum()
        union = ((pred_mask==c) | (true_mask==c)).sum()
        ious[c] = (inter+eps)/(union+eps)
    return ious

def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps=1e-7) -> float:
    pred, true = pred_mask.astype(bool), true_mask.astype(bool)
    inter = np.logical_and(pred, true).sum()
    return (2.0*inter+eps)/(pred.sum()+true.sum()+eps)
