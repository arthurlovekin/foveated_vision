import logging

import torch
import torchvision
from torch import nn

from utils import center_width_to_corners

# TODO: Could make all loss functions nn.Modules if they contain learnable parameters
# (just swap out the __call__ for a forward method, and add (nn.Module) after the class name)


class FoveationLoss:
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, fixation, bb):
        """
        fixation: [batch]x 2 tensor, fixation point 
        bb:       [batch]x 4  tensor [xlow, xhigh, ylow,yhigh] 
        """
        xinside = torch.logical_and(
            bb[..., 0:1] < fixation[..., 0:1],  fixation[..., 0:1] < bb[..., 1:2])
        xdist = torch.where(xinside, 0, torch.min(torch.abs(
            fixation[..., 0:1] - bb[..., 0:1]), torch.abs(fixation[..., 0:1] - bb[..., 1:2]),))
        logging.debug(f"x dist: {xdist}")
        yinside = torch.logical_and(
            bb[..., 2:3] < fixation[..., 1:2], fixation[..., 1:2] < bb[..., 3:4])
        ydist = torch.where(yinside, 0, torch.min(torch.abs(
            fixation[..., 1:2] - bb[..., 2:3]), torch.abs(fixation[..., 1:2] - bb[..., 3:4]),))
        logging.debug(f"y dist: {ydist}")
        xdist = xdist / self.img_size[0]
        ydist = ydist / self.img_size[1]
        dist_squared = xdist**2 + ydist**2
        # Sum over batch dimension
        return torch.sum(dist_squared, dim=0)


class IntersectionOverUnionLoss:
    def __init__(self, mode='default'):
        self.mode = mode

    def __call__(self, box1, box2):
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
        elif self.mode == 'neg_iou_generalised':
            return - torch.sum(torchvision.ops.generalized_box_iou(box1, box2,))
        else:
            return torch.sum(torchvision.ops.box_iou(box1, box2)[0])


class PeripheralFovealVisionModelLoss:
    def __init__(self, default_fovea_shape=(None, None)):
        self.iou_loss = IntersectionOverUnionLoss(mode='generalized')
        # TODO: Make this independent of the image size?
        self.foveation_loss = FoveationLoss((224, 224))
        self.iou_weight = 1.0
        self.foveation_weight = 1.0
        self.scale_weight = 0.0  # Disabled by default
        self.default_width, self.default_height = default_fovea_shape

    def fix_fovea_if_needed(self, fixations):
        """
        If fixation is two points, add a default width and height
        so we can use IoU loss on the fixation as well as the bounding box output.
        """
        if fixations.shape[-1] == 2:
            return torch.cat([
                fixations,
                torch.full_like(fixations[..., 0:1], self.default_width),
                torch.full_like(fixations[..., 1:2], self.default_height),
            ], axis=-1)
        else:
            return fixations

    def __call__(self, curr_bbox, next_fixation, true_curr_bbox, true_next_bbox):
        """
        Current version: consists of 2 parts
        1. Intersection over union loss between the current bounding box and the ground-truth current bounding box
        2. Foveation loss between the next fixation and the ground-truth bounding box (from the next timestep)
        """
        fixation_bbox = self.fix_fovea_if_needed(next_fixation)
        print(f'bbox: predicted{curr_bbox} ')
        print(f'bbox: actual   {true_curr_bbox} ')
        loss_iou = self.iou_loss(curr_bbox, true_curr_bbox)
        # logging.debug(f"IoU loss: {loss_iou.tolist()}")
        # TODO: Just output 4 points directly from the model
        # Model currently just outputs fixation center, while the width and height of the fovea are fixed.
        fovea_corner_parametrization = center_width_to_corners(fixation_bbox)
        print(f'fovea: predicted{fovea_corner_parametrization} ')
        print(f'fovea: actual   {true_next_bbox} ')
        loss_foveation = self.iou_loss(
            fovea_corner_parametrization, true_next_bbox)
        print(loss_foveation.tolist())
        # loss_foveation = self.foveation_loss(next_fixation, true_next_bbox)
        return self.iou_weight*loss_iou + self.foveation_weight*loss_foveation


if __name__ == "__main__":
    # fl = FoveationLoss([2,2])
    xes = torch.tensor(
        [[0, 0],
         [1, 0],
         [0, 1],
         [1, 1],
         [-1, -1],
         [2, 0.5],
         [-0.5, 2],
         [2, 2],
         ])
    bbs = torch.tensor([[-1, 1, -1, 1]]*8)
    # losses = fl(xes,bbs)
    # print(losses)
    # expected_losses = torch.tensor([[0,0,0,0,0,0.5,0.5,(2.0**0.5)/2]]).T
    # print(losses-expected_losses)
    # print(torch.allclose(losses, expected_losses))
    # # assert(torch.allclose(losses, expected_losses))

    # test iou
    iou_loss = IntersectionOverUnionLoss()
    fixationbox = torch.cat([
        xes,
        torch.full_like(xes[..., 0:1], 2),
        torch.full_like(xes[..., 0:1], 2),
    ], axis=-1)
    pred_boxes = center_width_to_corners(fixationbox)
    print(pred_boxes, bbs)
    iou_loss(pred_boxes, bbs)
    for i in range(8):
        pred, actual = pred_boxes[i:i+1], bbs[i:i+1]
        print(pred, actual)
        print(torchvision.ops.generalized_box_iou(pred, actual))
        print(torchvision.ops.generalized_box_iou_loss(pred, actual))
        print(iou_loss(pred, actual))

    logging.basicConfig(level=logging.INFO)
    iou_loss = IntersectionOverUnionLoss()
    groundtruth_bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    test_bbox = torch.tensor([[0.25, 0.25, 0.5, 0.5]])
    loss = iou_loss(groundtruth_bbox, test_bbox)
    logging.info(f"1. Actual IoU Loss: {loss}, expected: about {0.25}")

    groundtruth_bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    test_bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5]])
    loss = iou_loss(groundtruth_bbox, test_bbox)
    logging.info(f"Actual IoU Loss: {loss}, expected: {1.0}")

    groundtruth_bbox = torch.tensor(
        [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]])
    test_bbox = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]])
    loss = iou_loss(groundtruth_bbox, test_bbox)
    logging.info(f"Actual IoU Loss: {loss}, expected: {2.0}")
