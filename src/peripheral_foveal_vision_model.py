import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from foveation_module import FoveationModule
from torchvision.transforms import Resize
from torchinfo import summary

RESNET_DEFAULT_INPUT_SIZE = (224, 224)
# TODO: Add positional encoding


class PeripheralModel(nn.Module):
    """
    Creates a feature vector from a low-resolution background image
    using a pre-trained ResNet50 model. (These features are later combined
    with the foveal features to produce a bounding box and fixation point).
    """

    def __init__(self):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if hasattr(self.pretrained, "fc"):
            self.pretrained.fc = torch.nn.Identity()
        else:
            print("No fc layer found in pretrained model")
        # self.pretrained.fc = nn.Linear(2048, 4)
        # self.pretrined_new = torch.nn.Sequential(*list(self.pretrained.children())[:-1])
        # for param in self.pretrained[-1].parameters():
        #     param.requires_grad = False

    def forward(self, low_res_image):
        # output: (batch, 2048) feature vector
        return self.pretrained(low_res_image)


class FovealModel(nn.Module):
    """
    Creates a feature vector from a single high-resolution foveal patch.
    (These features are later combined
    with the peripheral features to produce a bounding box and fixation point).
    """

    def __init__(self):
        super().__init__()
        self.pretrained = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if hasattr(self.pretrained, "fc"):
            self.pretrained.fc = torch.nn.Identity()
        else:
            print("No fc layer found in pretrained model")

    def forward(self, foveal_patch):
        # output: (batch, 2048) feature vector
        return self.pretrained(foveal_patch)


class CombinerModel(nn.Module):
    """
    Takes buffers of foveal features, peripheral features, and fovea points, and
    combines them to produce a bounding box and fixation point.
    """

    def __init__(self):
        super().__init__()
        self.transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, num_decoder_layers=0)
    
    def make_sequence(self, foveal_features, peripheral_features, fovea_points):
        """
        Creates a sequence of features from the buffers. Transfomer expects
        batch x n x 1 sequence.
        """
        return torch.cat([foveal_features, peripheral_features, fovea_points], dim=1).unsqueeze(2)


    def forward(self, peripheral_features, foveal_features, fovea_points):
        input_sequence = self.make_sequence(foveal_features, peripheral_features, fovea_points)
        return self.transformer_model(input_sequence)


class PeripheralFovealVisionModel(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.foveation_module = FoveationModule(center=(100, 100), width=20, height=20)
        self.peripheral_model = PeripheralModel()
        self.foveal_model = FovealModel()
        self.combiner_model = CombinerModel()
        self.position_encoding = None
        self.current_fixation = None
        self.foveal_feature_buffer = None
        self.peripheral_feature_buffer = None
        self.fovea_point_buffer = None
        self.buffer_len = 3
        self.feature_len = 2048  # For extracted periperal and foveal features
        # Initialize the buffers
        self.reset_buffers()

    def reset_buffers(self):
        # Initialize the buffers with zeros
        self.foveal_feature_buffer = torch.zeros(
            (self.batch_size, self.buffer_len, self.feature_len), dtype=torch.float32
        )
        self.fovea_point_buffer = torch.zeros(
            (self.batch_size, self.buffer_len, 2), dtype=torch.float32
        )
        self.peripheral_feature_buffer = torch.zeros(
            (self.batch_size, self.buffer_len, self.feature_len), dtype=torch.float32
        )

    def forward(self, current_image, fovea_stack):
        # Extract features from the peripheral image
        background_img = self.downsample_periphery(current_image)
        peripheral_feature = self.peripheral_model(background_img)

        # Add the peripheral feature to the buffer
        self.peripheral_feature_buffer = torch.cat(
            [peripheral_feature, self.peripheral_feature_buffer], dim=0
        )
        # Pop the oldest peripheral feature from the buffer
        peripheral_features = self.peripheral_feature_buffer[-self.buffer_len :]

        foveal_patch = self.foveation_module.sample_fovea(
            current_image, self.current_fixation
        )

        # Extract features from the foveal patch
        foveal_feature = self.foveal_model(foveal_patch)

        # Add the foveal feature to the buffer
        self.foveal_feature_buffer = torch.cat(
            [foveal_feature, self.foveal_feature_buffer], dim=0
        )
        # Pop the oldest foveal feature from the buffer
        foveal_features = self.foveal_feature_buffer[-self.buffer_len :]

        # Add the fovea point to the buffer
        self.fovea_point_buffer = torch.cat(
            [self.current_fixation, self.fovea_point_buffer], dim=0
        )
        # Pop the oldest fovea point from the buffer
        fovea_points = self.fovea_point_buffer[-self.buffer_len :]

        # Combine and produce a bounding box + fixation point
        output = self.combiner_model(
            peripheral_features, foveal_features, fovea_points
        )

        # Update the buffers
        self.foveal_feature_buffer = foveal_features
        self.peripheral_feature_buffer = peripheral_features
        self.fovea_point_buffer = fovea_points

        return output

    def downsample_periphery(self, image, target_resolution=RESNET_DEFAULT_INPUT_SIZE):
        return Resize(
            image, (target_resolution[0], target_resolution[1]), antialias=True
        )


if __name__ == "__main__":
    # model = PeripheralModel()
    # print("Peripheral model summary:")
    # print(summary(model))

    model = PeripheralFovealVisionModel()
    print("Model summary:")
    print(summary(model))
    
    batch_size = 1
    test_input = torch.randn(1, 3, 224,224)
    print(f"Output from test input {model(test_input).shape}")

