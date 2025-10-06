import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional

def save_checkpoint(state: Dict[str,Any], filename="checkpoint.pth"): torch.save(state, filename)

def load_checkpoint(model, optimizer: Optional[torch.optim.Optimizer]=None, filename="checkpoint.pth", map_location=None):
    ckpt = torch.load(filename, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optim_state" in ckpt: optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt

@dataclass
class EarlyStopping:
    patience:int=7; min_delta:float=0.0; mode:str="max"
    best_score:Optional[float]=None; num_bad_epochs:int=0; is_converged:bool=False
    def step(self, current: float)->bool:
        if self.best_score is None: self.best_score=current; return False
        improvement = (current-self.best_score) if self.mode=="max" else (self.best_score-current)
        if improvement>self.min_delta: self.best_score=current; self.num_bad_epochs=0; return False
        self.num_bad_epochs+=1
        if self.num_bad_epochs>=self.patience: self.is_converged=True; return True
        return False
