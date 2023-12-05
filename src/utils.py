import torch
import torchvision
import torchvision.transforms.functional as TF


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


def bbox_to_img_coords(bbox, image):
    """ Convert a bounding box from [0, 1] range to image coordinates
    Also ensure that the bounding box is within the image bounds and
    clip if necessary.
    Args:
        bbox (torch.tensor): (batch, 4) bounding box
        image (torch.tensor): (channels, height, width) image
    """
    # Convert each corner and clip to image bounds
    bbox[:, 0] = torch.clamp(bbox[:, 0] * image.shape[2], min=0, max=image.shape[2])
    bbox[:, 1] = torch.clamp(bbox[:, 1] * image.shape[1], min=0, max=image.shape[1])
    bbox[:, 2] = torch.clamp(bbox[:, 2] * image.shape[2], min=0, max=image.shape[2])
    bbox[:, 3] = torch.clamp(bbox[:, 3] * image.shape[1], min=0, max=image.shape[1])
    return bbox

def make_bbox_grid(images, bboxes):
    """ 
    Create a grid of images with bounding boxes
    Args:
        images (list): List of images, each of shape (batch, channels, height, width)
        bboxes (list): List of bounding boxes, each of shape (batch, 4)
    """
    # For now, just show the first clip in the batch
    batch_ind = 0
    bbox_list = []
    decimate_frequency = 5  # Only show every Nth frame to save space
    for i in range(len(images)):
        if i % decimate_frequency != 0:
            continue
        image = TF.convert_image_dtype(images[i][batch_ind, :, :, :], dtype=torch.uint8)
        # Requires a dimension to possibly display multiple bounding boxes
        bbox = bboxes[i][batch_ind, :].unsqueeze(0)
        # Convert bounding boxes from [0, 1] range to image coordinates
        bbox = bbox_to_img_coords(bbox, image) # Also clips to image dimensions
        bbox_list.append(torchvision.utils.draw_bounding_boxes(image, bbox, colors=["green"], width=5))
    bbox_grid = torchvision.utils.make_grid(bbox_list)
    return bbox_grid


def draw_bboxes(images,bboxs,fixation_bboxs=None):
    
    ...
    
def save_bbox_vid(out_file, image,bbox,fixation_bbox=None):
    ...