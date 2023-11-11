from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from foveation_module import FoveationModule

class PeripheralFovealVisionModel(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.foveation_module = FoveationModule()
        self.peripheral_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.foveation_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.combiner_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.current_fixation = None
        self.position_encoding = None

    def forward(self, x):
        context_features = self.peripheral_model(x)
        foveal_patches = self.foveation_module.sample_fovea(x, self.current_fixation)
        foveal_features = self.foveation_model(foveal_patches)
        output, self.current_fixation = self.combiner_model(context_features, foveal_features)
        return output