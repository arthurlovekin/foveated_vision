"""
Methods for sampling points from an image tensor
"""
import torch
import torchvision.transforms.functional as TF
from torch import nn
from torchvision.transforms import Resize


class FoveationModule(nn.Module):
    """
    Given an image tensor and a fixation point, sample points and return the high-resolution foveal patch
    """

    def __init__(
        self,
        crop_width=0.25,
        crop_height=0.25,
        out_width_px=224,
        out_height_px=224,
        n_fixations=1,
        bound_crops=True,
    ):
        """
        Args:
            center: (fraction_x, fraction_y) where fraction_x and fraction_y are floats between 0 and 1
            crop_width: width of the foveal patch (a [0,1] fraction of the image width)
            crop_height: height of the foveal patch (a [0,1] fraction of the image height)
            out_width_px: width of the output image in pixels. This is needed so we can upsample the foveal patch for the foveal model.
            out_height_px: height of the output image in pixels
            n_fixations: number of fixation points (for later experimentation)
        """
        super().__init__()
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.n_fixations = n_fixations
        self.bound_crops = bound_crops
        self.resize = Resize((out_width_px, out_height_px), antialias=True)

    def forward(self, fixation, image, shape=None):
        """
        Args:
            fixation: (fraction_x, fraction_y) where fraction_x and fraction_y are floats between 0 and 1
            image: Tensor of shape [batch, channels, height, width]
            shape: Tensor of shape [..., 2], the target crop size in portion of the image; floats between 0 and 1
        """
        # If shape is not provided, use the default shape of the foveal patch; OK

        print("\n")

        # print("Image shape: ", image.shape)
        # print("Fixation shape: ",fixation.shape)
        # print("Requested shape vector: ", shape)
        image_xshape, image_yshape = image.shape[-2], image.shape[-1]
        crop_width_px = self.crop_width * image_xshape
        crop_height_px = self.crop_height * image_yshape
        if shape is None:
            fix_shape = fixation.shape
            widths = torch.full((fix_shape[0],), crop_width_px)
            heights = torch.full((fix_shape[0],), crop_height_px)
        else:
            widths = shape[..., 0] * image_xshape
            heights = shape[..., 1] * image_yshape
        # Make sure widths and heights are on the right device
        widths = widths.to(image.device)
        heights = heights.to(image.device)
        if self.bound_crops:
            # make sure we don't crop past border
            left_vec_term = torch.max(
                torch.zeros_like(fixation[..., 0]),
                fixation[..., 0] * image_xshape - widths // 2,
            )
            left_vec = torch.min(left_vec_term, image_xshape - widths)

            top_vec_term = torch.max(
                torch.zeros_like(fixation[..., 1]),
                fixation[..., 1] * image_yshape - heights // 2,
            )
            top_vec = torch.min(top_vec_term, image_yshape - heights)
        else:
            left_vec = fixation[..., 0] * image_xshape - widths // 2
            top_vec = fixation[..., 1] * image_yshape - heights // 2
        # Slice so we don't lose the batch dimension, but can apply a different crop to each image in the batch
        cropped = torch.cat(
            [
                TF.crop(image[i : i + 1], int(top), int(left), int(width), int(height))
                for i, (top, left, width, height) in enumerate(
                    zip(list(left_vec), list(top_vec), list(widths), list(heights))
                )
            ],
            dim=0,
        )
        # print("cropped image shape: ",cropped.shape)
        # Upscale cropped patch for when foveal model requires larger inputs than the crop window (in pixels).
        resized = self.resize(cropped)
        # print("Out resized image shape: ", resized.shape)
        return resized


class NeuralFoveationModule(nn.Module):
    """
    Takes the output attentions of the transformer and uses them to generate a sample foveation patch
    """

    def __init__(self, **kwargs):
        super().__init__()
        default_kwargs = {
            "crop_width": 0.25,
            "crop_height": 0.25,
            "out_width_px": 224,
            "out_height_px": 224,
            "n_fixations": 1,
            "bound_crops": False,
        }
        default_kwargs.update(kwargs)
        self.foveation_module = FoveationModule(default_kwargs)
        # TODO: Make the parameters of this model dynamic and also make the architecture better if necessary
        self.transformer_attentions_len = 2048
        self.fixation_model = nn.Sequential(
            nn.Linear(self.transformer_attentions_len, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),  # get into the range [0,1]
        )

    def forward(self, transformer_attentions, image):
        """
        Compute the parametrization of the foveal patch from the transformer attentions,
        then apply it to the image
        """
        fixation = self.fixation_model(transformer_attentions)
        return self.foveation_module(fixation, image)
