import logging

import torch
import torchvision
from torch import nn

from utils import center_width_to_corners

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
        # Sum over batch dimension
        return torch.sum(dist_squared,dim=0)
    


class IntersectionOverUnionLoss:
    """
    Converts a variety of IoU functions into loss functions (IoU alone is not what you want to minimize)
    https://learnopencv.com/iou-loss-functions-object-detection/#ciou-complete-iou-loss
    """
    def __init__(self, mode='complete', min_sidelength_threshold=0.01): 
        self.mode = mode
        self.min_sidelength_threshold = min_sidelength_threshold

    
    def __call__(self,box1,box2): 
        """
        box1: tensor of shape [batch, 4] where the 4 values are [x1,y1,x2,y2] (bounding box corners)
        box2: tensor of shape [batch, 4] where the 4 values are [x1,y1,x2,y2] (bounding box corners)
        """
        if self.mode == 'distance':
            loss = torchvision.ops.distance_box_iou_loss(box1, box2, reduction='sum')
        elif self.mode == 'complete':
            loss = torchvision.ops.complete_box_iou_loss(box1, box2, reduction='sum')
        elif self.mode == 'generalized':
            # Seems strictly worse than distance and complete
            loss = torchvision.ops.generalized_box_iou_loss(box1, box2, reduction='sum')
        else:
            # box_iou returns NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
            # so sum over the diagonal, where the ground-truth box is matched to the predicted box
            batch_size = box1.shape[0] if len(box1.shape) > 1 else 1
            loss = 1.0*batch_size - torch.sum(torch.diag(torchvision.ops.box_iou(box1, box2)))
        # warn if loss is nan
        if loss != loss: 
            logging.warning(f"NaN loss, box1: {box1}, box2: {box2}")
        # # warn if one box has no area
        # if torch.abs(box1[...,2:4] - box1[..., 0:2]) <= torch.ones_like(box1[...,2:4])*self.min_sidelength_threshold:
        #     logging.warning(f"Box1 has no area: {box1}")
        # if torch.abs(box2[...,2:4] - box2[..., 0:2]) <= torch.ones_like(box2[...,2:4])*self.min_sidelength_threshold:
        #     logging.warning(f"Box2 has no area: {box2}")
        
        return loss
        

class PeripheralFovealVisionModelLoss:
    def __init__(self,default_fovea_shape=(None,None)):
        self.iou_loss = IntersectionOverUnionLoss(mode='default')
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
        fixation_bbox = self.fix_fovea_if_needed(next_fixation)
        loss_iou = self.iou_loss(curr_bbox, true_curr_bbox)

        # TODO: Just output 4 points directly from the model
        fovea_corner_parametrization = center_width_to_corners(fixation_bbox)
        loss_foveation = self.iou_loss(fovea_corner_parametrization, true_next_bbox)
        logging.debug(f'bbox: predicted{curr_bbox} ')
        logging.debug(f'bbox: actual   {true_curr_bbox} ')
        logging.debug(f'fovea: predicted{fovea_corner_parametrization} ')
        logging.debug(f'fovea: actual   {true_next_bbox} ')
        logging.debug(f"foveation_loss (also an iou loss): {loss_foveation}, loss_iou: {loss_iou}")
        # loss_foveation = self.foveation_loss(next_fixation, true_next_bbox)
        return self.iou_weight*loss_iou + self.foveation_weight*loss_foveation


if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO)

    # fl = FoveationLoss([2,2])
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
    # losses = fl(xes,bbs)
    # logging.debug(losses)
    # expected_losses = torch.tensor([[0,0,0,0,0,0.5,0.5,(2.0**0.5)/2]]).T
    # logging.debug(losses-expected_losses)
    # logging.debug(torch.allclose(losses, expected_losses))
    # # assert(torch.allclose(losses, expected_losses))

    # test iou 
    iou_loss = IntersectionOverUnionLoss()
    fixationbox = torch.cat([
                    xes,
                    torch.full_like(xes[...,0:1],2),
                    torch.full_like(xes[...,0:1],2),
                ],axis=-1)
    pred_boxes = center_width_to_corners(fixationbox)
    logging.debug(pred_boxes,bbs)
    iou_loss(pred_boxes,bbs)
    for i in range(8):
        pred, actual = pred_boxes[i:i+1], bbs[i:i+1]
        logging.debug(pred, actual)
        logging.debug(torchvision.ops.generalized_box_iou(pred,actual))
        logging.debug(torchvision.ops.generalized_box_iou_loss(pred,actual))
        logging.debug(iou_loss(pred,actual))

    for mode in ['distance','complete','generalized','default']:
        iou_loss = IntersectionOverUnionLoss(mode=mode)
        groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
        test_bbox = torch.tensor([[0.25,0.25,0.5,0.5]])
        loss = iou_loss(groundtruth_bbox, test_bbox)
        logging.info(f"Standard example, batch = 1  -- {mode} IoU Loss: {loss}, IoU: {0.25}")

        groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
        test_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
        loss = iou_loss(groundtruth_bbox, test_bbox)
        logging.info(f"Perfect Match, batch = 1     -- {mode} IoU Loss: {loss}, IoU: 1.0")

        # test larger batchsize
        groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5],[0.0,0.0,0.5,0.5]])
        test_bbox = torch.tensor([[0.0,0.0,0.5,0.5],[0.0,0.0,0.5,0.5]])
        loss = iou_loss(groundtruth_bbox, test_bbox)
        logging.info(f"Perfect Match, batch = 2     -- {mode} IoU Loss: {loss}, IoU: 1.0")

        # test when bounding box has size 0
        groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
        test_bbox = torch.tensor([[0.0,0.0,0.0,0.0]])
        loss = iou_loss(groundtruth_bbox, test_bbox)
        logging.info(f"Zero bounding box, batch = 1 -- {mode} IoU Loss: {loss}, IoU: 0.0")

        # test when bounding box has nans
        groundtruth_bbox = torch.tensor([[0.0,0.0,0.5,0.5]])
        test_bbox = torch.tensor([[float('nan'),float('nan'),float('nan'),float('nan')]])
        loss = iou_loss(groundtruth_bbox, test_bbox)
        logging.info(f"NaN bounding box, batch = 1  -- {mode} IoU Loss: {loss}, IoU: 0.0")

        logging.info("------------------")






