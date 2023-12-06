import torch 
from utils import * 
import subprocess
from ENV import FFMPEG_PATH
import logging 
torchvision.disable_beta_transforms_warning()

from dataset.vot_dataset import *
from peripheral_foveal_vision_model import PeripheralFovealVisionModel


def visualize_video(model, video_frames,gt_bounding_box,normby=None):
    model.eval()
    with torch.no_grad(): 
        bboxes, fixations = model(video_frames)
    
    imgs_with_bboxes = draw_bboxes(video_frames,[bboxes,fixations,gt_bounding_box],names=['predicted bb','fovea','ground truth bb'],norm_by=normby)
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

    subprocess.run([FFMPEG_PATH,'-y','-f','image2','-i','out/bbox_%d.png', 'out/bboxes.mp4'])


def main(model_path,choose_vid=32): 
    model = PeripheralFovealVisionModel()
    model.load_state_dict(torch.load(model_path))
    ds = VotDataset()
    # print(ds[choose_vid][1])
    visualize_video(model,ds[choose_vid][0], ds[choose_vid][1],normby = ['default','default','xyxy'])

if __name__ == "__main__":
    main('/home/zeichen/foveated_vision/models/20231205_163401_model_epoch_1_step_100.pth')