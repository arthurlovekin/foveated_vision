import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Resize, CenterCrop, Normalize, Compose
from torchinfo import summary
import torchvision.transforms
# from peripheral_foveal_vision_model import PeripheralModel, FovealModel, CombinerModel
"""
the thing is that transformers expect to generate a sequence of the same type (eg. bounding boxes))
However in our implementation we are taking in features from the peripheral and foveal images, plus the previous fixation (or bounding box)
and outputting a bounding box .
We could make the transformer predict both the next image features and the next bounding box, 
and then have a loss between the transformer's predicted features and the actual next features in addition to the loss on the bounding box itself.

"""

weights = ResNet50_Weights.IMAGENET1K_V2
model = resnet50(weights=weights)
# print(model)
# print(summary(model))

test_input = torch.randn(3, 3, 420,420) # (batch, channels, height, width)

print(model(test_input).shape)

# get transforms that match the way the pretrained resnets were trained
resnet50_transforms = weights.transforms
print(f'Resnet50 transforms: {resnet50_transforms}')

# Get the default transforms for the ResNet model
transforms_list = Compose([
            Resize(232),
            CenterCrop(224),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
print("Default Transforms:")
print(transforms_list)
print("\nNormalization Mean:", transforms_list.transforms[-1].mean)
print("Normalization Standard Deviation:", transforms_list.transforms[-1].std)

test_image = torch.randn(3,3,420,420)

def resnet50_transform(image_tensor):
        resize = Resize(size=232,antialias=True)
        crop = CenterCrop(size=224)
        norm = Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return norm(crop(resize(image_tensor)))


# print(resnet50_transforms(test_image))
# print("=--------------")
print(resnet50_transform(test_image).shape)




# def forward2(self, fixation, image, crop_width_frac=None, crop_height_frac=None):
#     """
#         Fixation: tensor of shape (batch, 2) where the last dimension is the x and y coordinate of the center 
#         of the foveal patch, as fractions from 0 to 1 of the image width and height

#         image: tensor of shape (batch, channels, width, height)
#         crop_width_frac: tensor of shape (batch, 1), where each element is a fraction of the image width [0,1]
#         crop_height_frac: tensor of shape (batch, 1), where each element is a fraction of the image height [0,1]
#     """
#     if crop_width_frac is None:
#         crop_width_frac = self.crop_width
#     if crop_height_frac is None:
#         crop_height_frac = self.crop_height
    
#     image_width_px = image.shape[-2]
#     image_height_px = image.shape[-1]
#     crop_width_px = crop_width_frac * image_width_px
#     crop_height_px = crop_width_frac * image_height_px

#     # If the fixation center is too close to the edge of the image, bump it inwards
#     top_left_x = torch.max(torch.zeros_like(fixation[...,0]),fixation[...,0]*image_width_px - crop_width_px / 2.0)
#     top_left_x = torch.min(top_left_x, image_width_px - crop_width_px)
#     top_left_y = torch.max(torch.zeros_like(fixation[...,1]),fixation[...,1]*image_height_px - crop_height_px / 2.0)
#     top_left_y = torch.min(top_left_y, image_height_px - crop_height_px)

#     # Apply a different crop to each image in the batch
#     cropped =  torch.cat([TF.crop(image[i:i+1], int(top), int(left), int(crop_width_px), int(crop_height_px))
#                             for i,(top,left) in enumerate(zip(list(top_left_x), list(top_left_y)))],dim=0)
        

# self.current_fixation = torch.ones((self.batch_size, self.fixation_length), dtype=torch.float32, device=current_image.device)*0.5
# # if next(self.parameters()).is_cuda: # returns a boolean
# #     self.current_fixation = self.current_fixation.cuda()



        # # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=n_inputs, nhead=n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True)
        # # self.transformer_model = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer, num_layers=n_encoder_layers) # (contains multiple TransformerEncoderLayers)
        # self.sequence_dim = buffer_size*n_inputs
        
        # # Temporary: replace transformer with a simple linear model to see if it is a problem
        # self.transformer_model = nn.Sequential(
        #     nn.Linear(n_inputs, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, n_inputs),
        #     nn.ReLU(),
        # )