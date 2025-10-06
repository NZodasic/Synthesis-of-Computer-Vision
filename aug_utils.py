import torch, numpy as np

def rand_bbox(size, lam):
    W,H = size[2], size[3]
    cut_rat = np.sqrt(1.-lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx-cut_w//2,0,W), np.clip(cy-cut_h//2,0,H)
    bbx2, bby2 = np.clip(cx+cut_w//2,0,W), np.clip(cy+cut_h//2,0,H)
    return bbx1,bby1,bbx2,bby2

def mixup_data(x,y,alpha=1.0,device=None):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1.0
    if device is None: device=x.device
    index = torch.randperm(x.size(0)).to(device)
    return lam*x+(1-lam)*x[index,:], y, y[index], lam

def cutmix_data(x,y,alpha=1.0,device=None):
    lam = np.random.beta(alpha,alpha) if alpha>0 else 1.0
    if device is None: device=x.device
    index = torch.randperm(x.size(0)).to(device)
    bbx1,bby1,bbx2,bby2 = rand_bbox(x.size(),lam)
    x[:,:,bbx1:bbx2,bby1:bby2] = x[index,:,bbx1:bbx2,bby1:bby2]
    lam = 1-((bbx2-bbx1)*(bby2-bby1)/(x.size(-1)*x.size(-2)))
    return x, y, y[index], lam
