import torch 
from utils import * 
import subprocess
from ENV import FFMPEG_PATH
import logging 
torchvision.disable_beta_transforms_warning()
import sys
import os

from dataset.vot_dataset import *
from peripheral_foveal_vision_model import PeripheralFovealVisionModel


def visualize_video(model, video_frames,gt_bounding_box,normby=None):
    model.eval()
    with torch.no_grad(): 
        bboxes, fixations = model(video_frames)
    
    imgs_with_bboxes = draw_bboxes(video_frames,[bboxes,fixations,gt_bounding_box],names=['predicted bb','fovea','ground truth bb'],norm_by=normby)
    logging.info(f"Writing {video_frames.shape[0]} images to out/bbox_*.png")
    # Make sure the out directory exists
    if not os.path.exists('out'):
        os.mkdir('out')
    for i in range(video_frames.shape[0]): 
        torchvision.io.write_png(imgs_with_bboxes[i,...],f'out/bbox_{i}.png')
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
    subprocess.run([FFMPEG_PATH,'-y','-f','image2','-i','out/bbox_%d.png', 'out/bboxes.mp4'])


def main(model_path,choose_vid=32): 
    model = PeripheralFovealVisionModel()
    # TODO: make this actually work on CPU. Currently it crashes with a CUDA error...
    location = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    model.load_state_dict(torch.load(model_path, map_location=location))
    ds = VotDataset()
    # print(ds[choose_vid][1])
    logging.info(f"Video {choose_vid} has {len(ds[choose_vid][0])} frames")
    visualize_video(model,ds[choose_vid][0], ds[choose_vid][1],normby = ['default','default','xyxy'])

if __name__ == "__main__":
    # Get the model path from the command line
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1: 
        path = sys.argv[1]
        video_num = 32
        if len(sys.argv) > 2:
            video_num = int(sys.argv[2])
        logging.info(f"Loading model from {path}")
        main(path, choose_vid=video_num)
    else:
        main('/home/zeichen/foveated_vision/models/20231205_163401_model_epoch_1_step_100.pth')