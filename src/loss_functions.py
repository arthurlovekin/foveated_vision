import torch
import torchvision
from torch import nn
import logging

# TODO: Could make all loss functions nn.Modules if they contain learnable parameters
# (just swap out the __call__ for a forward method, and add (nn.Module) after the class name)

def center_width_to_corners(boxes):
    """
    Convert fovea fixation parametrization from 
    (batched) [xcenter, ycenter, width, height] to [xlow, xhigh, ylow, yhigh]
    boxes: [batch]x 4 tensor [xcenter, ycenter, width, height] 
    All values should be nonnegative (for this project should be in range 0-1)
    """
    xcenter = boxes[...,0:1]
    ycenter = boxes[...,1:2]
    width = boxes[...,2:3]
    height = boxes[...,3:4]
    xlow = xcenter - width/2
    xhigh = xcenter + width/2
    ylow = ycenter - height/2
    yhigh = ycenter + height/2

    return torch.cat([xlow,xhigh,ylow,yhigh],dim=-1)

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
        # Sum over batch dimension
        return torch.sum(dist_squared,dim=0)
    


class IntersectionOverUnionLoss:
    def __init__(self, mode='default'): 
        self.mode = mode
    
    def __call__(self,box1,box2): 
        """
        bb1: [batch]x 4 tensor [xlow, xhigh, ylow,yhigh] 
        bb2: [batch]x 4 tensor [xlow, xhigh, ylow,yhigh] 
        """
        # the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
        # so sum over the first row only
        if self.mode == 'complete':
            return torch.sum(torchvision.ops.complete_box_iou(box1, box2)[0])
        elif self.mode == 'distance':
            return torchvision.ops.distance_box_iou_loss(box1, box2reduction='sum')
        elif self.mode == 'generalized':
            return torchvision.ops.generalized_box_iou_loss(box1, box2, reduction='sum')
        else:
            return torch.sum(torchvision.ops.box_iou(box1, box2)[0])
        

class PeripheralFovealVisionModelLoss:
    def __init__(self,default_fovea_shape=(None,None)):
        self.iou_loss = IntersectionOverUnionLoss()
        # TODO: Make this independent of the image size?
        self.foveation_loss = FoveationLoss((224,224))
        self.iou_weight = 1.0
        self.foveation_weight = 1.0
        self.scale_weight = 0.0  # Disabled by default
        self.default_width, self.default_height = default_fovea_shape

    def fix_fovea_if_needed(self,fixations):
        if fixations.shape[-1] == 2: 
            return torch.cat([
                    fixations,
                    torch.full_like(fixations[...,0:1],self.default_width),
                    torch.full_like(fixations[...,0:1],self.default_width),
                ],axis=-1)
        else: 
            return fixations

    def __call__(self, curr_bbox, next_fixation, true_curr_bbox, true_next_bbox):
        """
        Current version: consists of 2 parts
        1. Intersection over union loss between the current bounding box and the ground-truth current bounding box
        2. Foveation loss between the next fixation and the ground-truth bounding box (from the next timestep)
        """
        next_fixation = self.fix_fovea_if_needed(next_fixation)
        loss_iou = self.iou_loss(curr_bbox, true_curr_bbox)

        if next_fixation.shape[-1] != 4:
            fixation_widths_heights = torch.ones_like(curr_bbox[...,2:4])*0.25
            fixation_bbox = torch.cat([next_fixation[...,0:2], fixation_widths_heights], dim=-1)
            fovea_corner_parametrization = center_width_to_corners(fixation_bbox)
        else:
            fovea_corner_parametrization = next_fixation
        loss_foveation = self.iou_loss(fovea_corner_parametrization, true_next_bbox)
        
        # loss_foveation = self.foveation_loss(next_fixation, true_next_bbox)
        return self.iou_weight*loss_iou + self.foveation_weight*loss_foveation


if __name__ == "__main__": 
    # fl = FoveationLoss([2,2])
    # xes = torch.tensor(
    #     [[0,0],
    #      [1,0],
    #      [0,1],
    #      [1,1],
    #      [-1,-1],
    #      [2,0.5],
    #      [-0.5,2],
    #      [2,2],
    #     ])
    # bbs = torch.tensor([[-1,1,-1,1]]*8)
    # losses = fl(xes,bbs)
    # expected_losses = torch.tensor([[0,0,0,0,0,0.5,0.5,(2.0**0.5)/2]]).T
    # logging.info(torch.allclose(losses, expected_losses))
    # assert(torch.allclose(losses, expected_losses))
    logging.basicConfig(level=logging.INFO)
    iou_loss = IntersectionOverUnionLoss()
    groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
    test_bbox = torch.tensor([[0.25,0.25,0.5,0.5]])
    loss = iou_loss(groundtruth_bbox, test_bbox)
    logging.info(f"1. Actual IoU Loss: {loss}, expected: about {0.25}")

    groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
    test_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
    loss = iou_loss(groundtruth_bbox, test_bbox)
    logging.info(f"Actual IoU Loss: {loss}, expected: {1.0}")

    groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5],[0.0,0.0,0.5,0.5]])
    test_bbox = torch.tensor([[0.0,0.0,0.5,0.5],[0.0,0.0,0.5,0.5]])
    loss = iou_loss(groundtruth_bbox, test_bbox)
    logging.info(f"Actual IoU Loss: {loss}, expected: {2.0}")
