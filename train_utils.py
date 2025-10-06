import torch, numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
from .metrics import confusion_matrix

def train_one_epoch(model, dataloader: DataLoader, criterion: Callable, optimizer, device, scaler=None, max_grad_norm=None, accumulate_steps=1) -> Dict[str,float]:
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    optimizer.zero_grad()
    for step,(inputs,labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(inputs)
            loss = criterion(outputs, labels)/accumulate_steps
        if scaler: scaler.scale(loss).backward()
        else: loss.backward()
        if (step+1)%accumulate_steps==0:
            if scaler:
                if max_grad_norm: 
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                if max_grad_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()*accumulate_steps
        running_corrects += (outputs.argmax(1)==labels).sum().item()
        total += labels.size(0)
    return {"loss": running_loss/len(dataloader.dataset), "acc": 100.0*running_corrects/total}

def evaluate_classification(model, dataloader: DataLoader, device, num_classes=None) -> Dict[str,Any]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs,labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    all_preds, all_labels = np.concatenate(all_preds), np.concatenate(all_labels)
    results = {"accuracy": (all_preds==all_labels).mean()*100.0}
    if num_classes:
        cm = confusion_matrix(all_preds, all_labels, num_classes)
        results["confusion_matrix"] = cm
        results["per_class_acc"] = cm.diagonal().astype(float)/(cm.sum(1)+1e-12)
    return results
