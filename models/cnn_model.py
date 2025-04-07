import torch.nn as nn
from torchvision import models

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        output = self.regressor(features)
        return output.squeeze(1)
