import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Resize
from torchinfo import summary
from peripheral_foveal_vision_model import PeripheralModel, FovealModel, CombinerModel

# weights = ResNet50_Weights.IMAGENET1K_V2
# model = resnet50(weights=weights)
model = PeripheralModel()
# print(model)
print(summary(model))

test_input = torch.randn(3, 3, 420,420)
print(model(test_input).shape)