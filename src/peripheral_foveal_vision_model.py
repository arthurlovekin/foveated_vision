import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from foveation_module import FoveationModule
from torchvision.transforms import Resize
from torchinfo import summary

class PeripheralModel(nn.Module):
    """
    Chops a pre-trained ResNet50 to add a custom bounding box regression head
    along with the parametrization of the next foveal fixation point
    Also makes the input match the feature space shape coming from the fovealModel and the pPeri
    """
    def __init__(self, input_resolution=(420, 420)):
        super().__init__()
        pretrained_input_shape = (224, 224)
        self.input_resolution = input_resolution
        
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if hasattr(self.pretrained, 'fc'):
            self.pretrained.fc = torch.nn.Identity()
        else:
            print('No fc layer found in pretrained model')
        self.pretrained.fc = nn.Linear(2048, 4)
        # self.pretrined_new = torch.nn.Sequential(*list(self.pretrained.children())[:-1])
        # for param in self.pretrained[-1].parameters():
        #     param.requires_grad = False

        # TODO: chop the classification head off of the pretrained model


    def forward(self, image):
        # image = self.input_adapter(image)
        return self.pretrained(image)



class FovealModel(nn.Module):
    """
    Chops a pre-trained ResNet50 to remove the classification head
    and also make the input resolution match the shape of the foveal patches
    """
    def __init__(self, input_resolution=(480, 360)):
        super().__init__()
        
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.input_resolution = input_resolution

    def forward(self, foveal_patches):
        return self.model(image)

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
        super().__init__()
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

if __name__ == "__main__":
    print("Making model")
    model = PeripheralModel()
    print("Printing model summary")
    print(summary(model))
    # print(summary(model, input_size=(1, 3, 420, 420)))
    # test_input = torch.randn(1, 3, 420,420)
    # print(model(test_input).shape)