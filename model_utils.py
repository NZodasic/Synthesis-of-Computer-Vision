import torch.nn as nn
from torchvision import models

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(model: nn.Module, mode='kaiming'):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if mode == 'kaiming':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif mode == 'normal':
                nn.init.normal_(m.weight, 0, 0.01)
            else:
                raise ValueError("Unknown init mode")
            if m.bias is not None: nn.init.constant_(m.bias,0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight,1)
            nn.init.constant_(m.bias,0)

def get_pretrained_resnet(num_classes=1000, version='resnet50', pretrained=True):
    if version=='resnet18': model = models.resnet18(pretrained=pretrained)
    elif version=='resnet34': model = models.resnet34(pretrained=pretrained)
    elif version=='resnet50': model = models.resnet50(pretrained=pretrained)
    else: raise ValueError("Unsupported resnet version")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
