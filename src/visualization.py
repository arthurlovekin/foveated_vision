import torch 
from utils import * 
import subprocess
from ENV import FFMPEG_PATH
import logging 
torchvision.disable_beta_transforms_warning()
import sys
import os
from tqdm import tqdm
from dataset.vot_dataset import *
from single_frame_model import SingleFrameTrackingModel


def visualize_video(model, video_frames,gt_bounding_box,normby=None, out_dir=None):
    model.eval()
    # Make a list of all the bounding boxes and fixations
    all_bboxes = torch.Tensor()
    with torch.no_grad(): 
        # Evaluate model in autoregressive way
        # Make progress bar
        progress_bar = tqdm(range(1, video_frames.shape[0]), leave=True)
        # Start with the gt bounding box
        prev_image = video_frames[0,...].unsqueeze(0)
        prev_bbox = gt_bounding_box[0, :].unsqueeze(0)
        all_bboxes = prev_bbox  # To make the dimensions match
        curr_bbox = None
        curr_image = None
        for i in progress_bar: 
            # Get the current image
            curr_image = video_frames[i,...].unsqueeze(0)
            bboxes = model(prev_image, prev_bbox, curr_image)
            # Update the previous image and bounding box
            prev_image = curr_image
            # prev_bbox = bboxes  # "Correct" way to do it
            # Get the next bounding box from ground truth
            prev_bbox = gt_bounding_box[i, :].unsqueeze(0)
            # Recover sequence dim 
            # logging.info(f"bboxes.shape: {bboxes.shape}")
            all_bboxes = torch.cat([all_bboxes,bboxes],axis=0)
    # logging.info(f"all_bboxes.shape: {all_bboxes.shape}")
    # logging.info(f"gt_bounding_box.shape: {gt_bounding_box.shape}")
    imgs_with_bboxes = draw_bboxes(video_frames,[all_bboxes,gt_bounding_box],names=['predicted bb','ground truth bb'],norm_by=normby)
    # Make sure the out directory exists
    if out_dir is None:
        out_dir = 'out'
    logging.info(f"Writing {video_frames.shape[0]} images to {out_dir}/bbox_*.png")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(video_frames.shape[0]): 
        torchvision.io.write_png(imgs_with_bboxes[i,...],f'{out_dir}/bbox_{i}.png')
    # this assumes you have ffmpeg on your system and added to your path 
    # to install it you can run the following commands: (in an appropriate directory )
    # this downloads a precompiled binary (linked to from the official ffmpeg project page) unpacks it and adds it to your path
    #       wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
    #       tar xJfv ffmpeg-release-amd64-static.tar.xz
    #       rm ffmpeg-release-amd64-static.tar.xz
    #       chmod +x ffmpeg-6.1-amd64-static/ffmpeg
    #       echo "export PATH=$(pwd)/ffmpeg-6.1-amd64-static:\$PATH" >>~/.bashrc
    #       source ~/.bashrc
    logging.info(f"Writing video to out/bboxes.mp4")
    # Log exact command run
    logging.info(f"Running command: {FFMPEG_PATH} -y -f image2 -i out/bbox_%d.png out/bboxes.mp4")
    subprocess.run([FFMPEG_PATH,'-y','-f','image2','-i',f'{out_dir}/bbox_%d.png', f'{out_dir}/bboxes.mp4'])


def main(model_path,choose_vid=32, out_dir=None): 
    model = SingleFrameTrackingModel(input_size=(512,512))
    model.load_state_dict(torch.load(model_path))
    ds = VotDataset()
    # print(ds[choose_vid][1])
    logging.info(f"Video {choose_vid} has {len(ds[choose_vid][0])} frames")
    visualize_video(model,ds[choose_vid][0], ds[choose_vid][1],normby = ['default','xyxy'], out_dir=out_dir)

if __name__ == "__main__":
    # Get the model path from the command line
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1: 
        path = sys.argv[1]
        video_num = 32
        if len(sys.argv) > 2:
            video_num = int(sys.argv[2])
        # Get output directory from command line
        if len(sys.argv) > 3:
            out_dir = sys.argv[3] 
        logging.info(f"Loading model from {path}")
        main(path, choose_vid=video_num, out_dir=out_dir)
    else:
        main('/home/zeichen/foveated_vision/models/20231205_163401_model_epoch_1_step_100.pth')
