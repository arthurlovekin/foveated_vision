import torch
import torchvision
from torch import nn
import logging

# TODO: Could make all loss functions nn.Modules if they contain learnable parameters
# (just swap out the __call__ for a forward method, and add (nn.Module) after the class name)

class FoveationLoss:
    def __init__(self,img_size):
        self.img_size = img_size
    
    def __call__(self,fixation, bb): 
        """
        fixation: [batch]x 2 tensor, fixation point 
        bb:       [batch]x 4  tensor [xlow, xhigh, ylow,yhigh] 
        """
        xinside = torch.logical_and(bb[...,0:1] < fixation[...,0:1],  fixation[...,0:1] < bb[...,1:2])
        xdist = torch.where(xinside,0,torch.min(torch.abs(fixation[...,0:1] - bb[...,0:1]),torch.abs(fixation[...,0:1] - bb[...,1:2]),))
        logging.debug(f"x dist: {xdist}")
        yinside = torch.logical_and(bb[...,2:3] < fixation[...,1:2], fixation[...,1:2] < bb[...,3:4])
        ydist = torch.where(yinside,0,torch.min(torch.abs(fixation[...,1:2] - bb[...,2:3]),torch.abs(fixation[...,1:2] - bb[...,3:4]),))
        logging.debug(f"y dist: {ydist}")
        xdist = xdist / self.img_size[0]
        ydist = ydist / self.img_size[1]
        dist_squared = xdist**2 + ydist**2
        return dist_squared

class IntersectionOverUnionLoss:
    def __init__(self): 
        pass
    
    def __call__(self,box1,box2): 
        """
        bb1: [batch]x 4 tensor [xlow, xhigh, ylow,yhigh] 
        bb2: [batch]x 4 tensor [xlow, xhigh, ylow,yhigh] 
        """
        return torchvision.ops.distance_box_iou_loss(box1, box2, reduction='none')

class PeripheralFovealVisionModelLoss:
    def __init__(self):
        self.iou_loss = IntersectionOverUnionLoss()
        # TODO: Make this independent of the image size?
        self.foveation_loss = FoveationLoss((224,224))
    
    def __call__(self, curr_bbox, next_fixation, true_curr_bbox, true_next_bbox):
        """
        Current version: consists of 2 parts
        1. Intersection over union loss between the current bounding box and the ground-truth current bounding box
        2. Foveation loss between the next fixation and the ground-truth bounding box (from the next timestep)
        """
        loss_iou = self.iou_loss(curr_bbox, true_curr_bbox)
        loss_foveation = self.foveation_loss(next_fixation, true_next_bbox)
        # TODO: Make this a weighted combination
        return loss_iou + loss_foveation


if __name__ == "__main__": 
    fl = FoveationLoss([2,2])
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
    losses = fl(xes,bbs)
    expected_losses = torch.tensor([[0,0,0,0,0,0.5,0.5,(2.0**0.5)/2]]).T
    print(torch.allclose(losses, expected_losses))
    assert(torch.allclose(losses, expected_losses))