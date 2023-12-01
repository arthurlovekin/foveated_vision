import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from foveation_module import FoveationModule
from torchvision.transforms import Resize

class PeripheralModel(nn.Module):
    """
    Chops a pre-trained ResNet50 to add a custom bounding box regression head
    along with the parametrization of the next foveal fixation point
    Also makes the input match the feature space shape coming from the fovealModel and the pPeri
    """
    def __init__(self, input_resolution=(480, 360)):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.input_resolution = input_resolution


class FovealModel(nn.Module):
    """
    Chops a pre-trained ResNet50 to remove the classification head
    and also make the input resolution match the shape of the foveal patches
    """
    def __init__(self, input_resolution=(480, 360)):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.input_resolution = input_resolution

class CombinerModel(nn.Module):
    """
    Chops a pre-trained ResNet50 to remove the classification head
    and also make the input higher resolution than the original ResNet50
    """
    def __init__(self, input_resolution=(480, 360)):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.input_resolution = input_resolution

class PeripheralFovealVisionModel(nn.Module):
    def __init__(self):
        super().__init__(self)
        self.foveation_module = FoveationModule()
        # TODO: Adjust the front and back of the resnets so they fit our dataloader images
        self.peripheral_model = PeripheralModel()
        self.foveal_model = FovealModel()
        self.combiner_model = CombinerModel()
        self.position_encoding = None
        self.current_fixation = None
        self.fovea_stack = None

    def forward(self, current_image, fovea_stack):
        background_img = self.downsample_periphery(current_image)
        context_features = self.peripheral_model(background_img)

        foveal_patch = self.foveation_module.sample_fovea(current_image, self.current_fixation)
        foveal_patches = torch.cat([foveal_patch, fovea_stack], dim=0) # Remove the oldest fovea from the stack
        foveal_features = self.foveal_model(foveal_patches)

        output, self.current_fixation = self.combiner_model(context_features, foveal_features)
        return output
    
    def downsample_periphery(self, image, target_resolution=(480, 360)):
        return Resize(image, (target_resolution[0], target_resolution[1]), antialias=True)
    

