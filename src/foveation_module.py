"""
Methods for sampling points from an image tensor
"""
import torch 
from torch import nn
import numpy as np 
import torchvision.transforms.functional as TF


class FoveationLoss:
    def __init__(self,img_size,exp=1): 
        self.img_size = img_size
        self.exp = exp
    
    def __call__(self,fixation, bb): 
        """
        fixation: [batch]x 2 tensor, fixation point 
        bb:       [batch]x 4  tensor [xlow, xhigh, ylow,yhigh] 
        """
        xinside = torch.logical_and(bb[...,0:1] < fixation[...,0:1],  fixation[...,0:1] < bb[...,1:2])
        xdist = torch.where(xinside,0,torch.min(torch.abs(fixation[...,0:1] - bb[...,0:1]),torch.abs(fixation[...,0:1] - bb[...,1:2]),))
        print(xdist)
        yinside = torch.logical_and(bb[...,2:3] < fixation[...,1:2], fixation[...,1:2] < bb[...,3:4])
        ydist = torch.where(yinside,0,torch.min(torch.abs(fixation[...,1:2] - bb[...,2:3]),torch.abs(fixation[...,1:2] - bb[...,3:4]),))
        print(ydist)
        xdist = xdist / self.img_size[0]
        ydist = ydist / self.img_size[1]
        dist = torch.pow(xdist**2  + ydist**2,self.exp/2)
        return dist

# TODO: Make this 
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

    def update_fixation(self, center, width, height):
        self.center = (center[0], center[1])
        self.width = width
        self.height = height

    def forward(self,fixation,image): 
        top = min(0,fixation[1] - self.height // 2)
        left = min(0,fixation[0] - self.width // 2)
        return TF.crop(image, top, left, self.height, self.width) 
    

if __name__ == "__main__": 
    xes = torch.tensor(
        [[0,0],
         [1,0],
         [0,1],
         [1,1],
         [-1,-1],
         [2,0.5],
         [-0.5,2],
         [2,2],
        ])
    bbs = torch.tensor([[-1,1,-1,1]]*8)
    fl = FoveationLoss([2,2])
    losses = fl(xes,bbs)
    expected_losses = torch.tensor([[0,0,0,0,0,0.5,0.5,(2.0**0.5)/2]]).T
    print(torch.allclose(losses, expected_losses))
    assert(torch.allclose(losses, expected_losses))

