import matplotlib.pyplot as plt, math, numpy as np, torch
from typing import Optional, List, Any

def imshow_tensor(img_tensor: torch.Tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), title: Optional[str]=None):
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    img = np.clip(img*np.array(std)+np.array(mean),0,1)
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis("off"); plt.show()

def visualize_batch(images: torch.Tensor, labels: Optional[List[Any]]=None, ncols=4):
    B, nrows = images.size(0), math.ceil(images.size(0)/ncols)
    plt.figure(figsize=(ncols*3,nrows*3))
    for i in range(B):
        plt.subplot(nrows,ncols,i+1)
        imshow_tensor(images[i], title=(str(labels[i]) if labels is not None else None))
