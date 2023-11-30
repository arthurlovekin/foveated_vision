"""
Methods for sampling points from an image tensor
"""
from torch import nn
from torch.functional import F
import torchvision.transforms.functional as TF

class FoveationModule(nn.Module):
    """
    Given an image tensor and a fixation point, sample points and return the high-resolution foveal patch 
    """
    def __init__(self, center, width, height, n_fixations=1):
        """
        center: (pixel_x, pixel_y) of the fixation point
        width: width of the foveal patch (in pixels)
        height: height of the foveal patch (in pixels)
        n_fixations: number of fixation points (for later experimentation)
        """
        self.center = (center[0], center[1])
        self.width = width
        self.height = height
        self.n_fixations = n_fixations

    def sample_fovea(self, image):
        """
        Sample points from the fovea region of the image
        Input:
            image: (batch, channels, width, height) image tensor
        Output:
            fovea: (batch, channels, fovea_width, fovea_height) image tensor
        """
        top = min(0,self.center[1] - self.height // 2)
        left = min(0,self.center[0] - self.width // 2)
        return TF.crop(image, top, left, self.height, self.width) 

    




