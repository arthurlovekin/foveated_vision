"""
Methods for sampling points from an image tensor
"""
import torch 
from torch import nn
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize


class FoveationModule(nn.Module):
    """
    Given an image tensor and a fixation point, sample points and return the high-resolution foveal patch 
    """
    def __init__(self, out_width=100, out_height=100, n_fixations=1):
        """
        center: (fraction_x, fraction_y) where fraction_x and fraction_y are floats between 0 and 1
        width: width of the foveal patch (a [0,1] fraction of the image width)
        height: height of the foveal patch (a [0,1] fraction of the image height)
        n_fixations: number of fixation points (for later experimentation)
        """
        super().__init__() 
        self.out_width = out_width
        self.out_height = out_height
        self.n_fixations = n_fixations
        self.resize = Resize((out_width,out_height), antialias=True)

    def forward(self,fixation,image, shape=None): 
        """
        fixation: (fraction_x, fraction_y) where fraction_x and fraction_y are floats between 0 and 1
        shape: Tensor of shape [..., 2], the target crop size in portion of the image; floats between 0 and 1 
        """
        # If shape is not provided, use the default shape of the foveal patch; OK 

        print('\n')

        # print("Image shape: ", image.shape)
        # print("Fixation shape: ",fixation.shape)
        # print("Requested shape vector: ", shape)
        image_xshape, image_yshape = image.shape[-2], image.shape[-1]
        if shape is None: 
            fix_shape = fixation.shape
            widths = torch.full((fix_shape[0],),self.out_width)
            heights = torch.full((fix_shape[0],),self.out_height)
        else: 
            widths = shape[...,0] * image_xshape
            heights = shape[...,1] * image_yshape
        # make sure we don't crop past border 
        left_vec = torch.min(torch.max(torch.zeros_like(fixation[...,0]),fixation[...,0]*image_xshape - widths // 2) , image_xshape - widths)
        top_vec = torch.min(torch.max(torch.zeros_like(fixation[...,0]),fixation[...,1]*image_yshape - heights // 2) , image_yshape -heights)
        
        cropped =  torch.cat([TF.crop(image[i:i+1], int(top), int(left),int(width),int(height)) for i,(top,left,width,height) in enumerate(zip(list(left_vec), list(top_vec),list(widths),list(heights)))],dim=0)
        # print("cropped image shape: ",cropped.shape)
        resized =  self.resize(cropped)
        # print("Out resized image shape: ", resized.shape)
        return resized
        


