import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        base_model = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # remove FC
        
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)  # flatten