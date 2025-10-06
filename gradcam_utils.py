import torch, torch.nn.functional as F, numpy as np
import torch.nn as nn

class SimpleGradCAM:
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations, self.gradients = None, None
        self._register_hooks()

    def _get_target_module(self):
        module = self.model
        for attr in self.target_layer.split('.'):
            module = getattr(module, attr)
        return module

    def _register_hooks(self):
        target = self._get_target_module()
        def f_hook(_,__,out): self.activations = out.detach()
        def b_hook(_,__,gout): self.gradients = gout[0].detach()
        target.register_forward_hook(f_hook)
        target.register_backward_hook(b_hook)

    def generate_cam(self, input_tensor: torch.Tensor, target_class=None):
        input_tensor = input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)
        if target_class is None: target_class = outputs.argmax(1).item()
        loss = outputs[0,target_class]
        self.model.zero_grad(); loss.backward(retain_graph=True)
        grads, acts = self.gradients[0], self.activations[0]
        weights = grads.mean((1,2),keepdim=True)
        cam = F.relu((weights*acts).sum(0)).cpu().numpy()
        cam = (cam-cam.min())/(cam.max()-cam.min()+1e-9)
        return np.uint8(cam*255)
