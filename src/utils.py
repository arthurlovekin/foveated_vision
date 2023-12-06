import torch
import torchvision
import torchvision.transforms.functional as TF
import logging


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
    print('pre')
    print(bbox)
    bbox = torch.stack(
        [torch.clamp(bbox[:, 0] * image.shape[-2], min=0, max=image.shape[-2]),
         torch.clamp(bbox[:, 1] * image.shape[-2], min=0, max=image.shape[-2]),
         torch.clamp(bbox[:, 2] * image.shape[-1], min=0, max=image.shape[-1]),
         torch.clamp(bbox[:, 3] * image.shape[-1], min=0, max=image.shape[-1])
        ],axis=1) 
    print(bbox)
    return bbox

def make_bbox_grid(images, bboxes, gt_bboxes=[], decimation=5):
    """ 
    Create a grid of images with bounding boxes
    Args:
        images (list): List of images, each of shape (batch, channels, height, width)
        bboxes (list): List of bounding boxes, each of shape (batch, 4). Should be from 0 to 1
        gt_bboxes (list): List of ground truth bounding boxes, each of shape (batch, 4). Should be from 0 to 1
        decimation (int): Decimation factor for the images. Will only show every decimation'th image
    """
    # For now, just show the first clip in the batch
    batch_ind = 0
    bbox_list = []
    one_bad_bbox = False
    for i in range(len(images)):
        if i % decimation != 0:
            continue
        image = TF.convert_image_dtype(images[i][batch_ind, :, :, :], dtype=torch.uint8)
        # Requires a dimension to possibly display multiple bounding boxes
        bbox = bboxes[i][batch_ind, :].unsqueeze(0)
        print('\n\n\n')

        print(bbox)
        # Convert bounding boxes from [0, 1] range to image coordinates
        bbox = bbox_to_img_coords(bbox, image) # Also clips to image dimensions
        bb_to_draw = None
        if len(gt_bboxes) > 0:
            gt_bbox = gt_bboxes[i][batch_ind, :].unsqueeze(0)
            gt_bbox = bbox_to_img_coords(gt_bbox, image)
            # Still draw the ground truth bounding box even if the model output is invalid
            if not one_bad_bbox and bbox_valid(bbox):
                colors = ["red", "green"]
                bb_to_draw = torch.cat([bbox, gt_bbox], dim=0)
            else:
                if not one_bad_bbox:
                    logging.info(f"Predicted bounding box invalid, drawing gt bbox only")
                    one_bad_bbox = True
                colors = ["green"]
                logging.info(f"Ground truth bbox: {gt_bbox}")
                bb_to_draw = gt_bbox
        else:
            bb_to_draw = bbox
            colors = ["red"]
        bbox_list.append(torchvision.utils.draw_bounding_boxes(image, bb_to_draw, colors=colors, width=5))
    bbox_grid = torchvision.utils.make_grid(bbox_list)
    return bbox_grid

def bbox_valid(bbox):
    """
    Check if a bounding box is valid (i.e. has a width and height > 0)
    bbox: [batch]x 4 tensor [xlow, xhigh, ylow,yhigh]
    """
    # Check that for all batch elements, xhigh > xlow and yhigh > ylow
    all_valid = torch.all(bbox[...,0:1] < bbox[...,1:2],dim=-1) & torch.all(bbox[...,2:3] < bbox[...,3:4],dim=-1)
    return all_valid
    
def fix_fovea_if_needed(fixations,default_shape):
    if fixations.shape[-1] == 2: 
        return torch.cat([
                fixations,
                torch.full_like(fixations[...,0:1],default_shape[0]),
                torch.full_like(fixations[...,1:2],default_shape[1]),
            ],axis=-1)
    else: 
        return fixations

def cwh_perc_to_pixel_xyxy(bbox,image_shape,default_fovea_shape=[0.25,0.25]): 
        bbox = fix_fovea_if_needed(bbox,default_shape=default_fovea_shape)
        bbox = center_width_to_corners(bbox)
        # print(bbox)
        bbox[...,0:2] = torch.clamp(bbox[...,0:2] * image_shape[-2], 0, image_shape[-2])
        bbox[...,2:4] = torch.clamp(bbox[...,2:4] * image_shape[-1], 0, image_shape[-1])

        return bbox[:,torch.tensor([0,2,1,3])].int()

def xxyy_perc_to_pixel_xyxy(bbox,image_shape): 
        print(bbox[0:4])
        # print(bbox)
        bbox[...,0:2] = torch.clamp(bbox[...,0:2] * image_shape[-2], 0, image_shape[-2])
        bbox[...,2:4] = torch.clamp(bbox[...,2:4] * image_shape[-1], 0, image_shape[-1])

        return bbox[:,torch.tensor([0,2,1,3])].int()

def draw_bboxes(images,bboxes:list[torch.tensor],names:list=None,norm_by:list[str]=None,default_fovea_shape=[0.25,0.25]):
    out_imgs = []
    if norm_by is None: 
        norm_by = ['default']*len(bboxes)
    images = (images * 256).to(torch.uint8)
    for i,(bbox,norm) in enumerate(zip(bboxes,norm_by)):
        if (type(norm) is str and norm == 'default'): 
            bboxes[i] = cwh_perc_to_pixel_xyxy(bbox,images.shape,default_fovea_shape=default_fovea_shape)
        elif type(norm) is str and norm == 'xyxy': 
            bboxes[i] = xxyy_perc_to_pixel_xyxy(bbox,images.shape)
        elif norm is not  None: 
            bbox = norm(bbox)
        
        print(bboxes[i][0:4])
        print(torch.all(torch.logical_and(bboxes[i][...,0] < bboxes[i][...,2],  bboxes[i][...,1] < bboxes[i][...,3])))
        # print(bbox)
    for imagenum in range(images.shape[0]):
        # print(images)
        out_imgs.append(torchvision.utils.draw_bounding_boxes(
                images[imagenum,...], 
                boxes=torch.stack([b[imagenum,...] for b in bboxes]), 
                labels = names 
            ))
    return torch.stack(out_imgs)    
    
def save_bbox_vid(out_file, images,bbox,fixation_bbox=None):
    video = draw_bboxes(images,bbox,fixation_bbox)
    torchvision.io.write_video(out_file,video)