from torch.utils.data import Dataset 
from torchvision.io import read_image
import torch
import os 

"""
Dataset Properties:
# Videos are different length temporally, and also different resolution
# Min Resolution: (480, 360)
# Max Resolution: (3840, 2160)
# min duration (sec): 0.4 (~40 frames)
# max duration (sec): 148 (~14800 frames)
# avg duration (sec): 15 (~150 frames)
"""
class GOT10kDataset(Dataset):
    def __init__(self, base_dir) -> None:
        super().__init__()
        self.base_dir = base_dir

        # check if the base directory exists
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError('Base directory not found')
        if not os.path.basename(self.base_dir) in ['train', 'val', 'test']:
            raise ValueError('Dataset must come from one of the folders "/train", "/val", or "/test"')
        
        # find the number of videos in the folder, and number of total frames in all videos
        if not os.path.exists(os.path.join(self.base_dir, 'list.txt')):
            raise FileNotFoundError('list.txt not found in the base directory')
        # Video folders in the form GOT-10k_Val_000180
        # Images are in the form 00000000.jpg
        self.n_videos = 0
        self.n_frames = 0
        self.video_lengths = []
        with open(os.path.join(self.base_dir, 'list.txt'), 'r') as f:
            self.video_filenames = f.readlines()
            print(f"Loading Dataset ...")
            for line in self.video_filenames:
                if len(line.strip()) > 0:
                    self.n_videos += 1
                    # get the number of frames in the video
                    video_dir = os.path.join(self.base_dir, line.strip())
                    if not os.path.exists(video_dir):
                        print(f'Video directory {video_dir} not found')
                    
                    # Each video folder contains 4 annotation files, one meta-info file, and then all of the images
                    self.n_frames += len(os.listdir(video_dir)) - 5
                    # for file in os.listdir(video_dir):
                    #     if file.split('.')[1] == 'jpg':
                    #         self.n_frames += 1

        self.frames_per_sequence = 40 # sampled at 10Hz
        # Not sure how much RAM we have


    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx: int):
        """
        Returns:
            image_sequence: (torch.Tensor) of shape (frames_per_sequence, 3, ?, ?)
            label_sequence: (torch.Tensor) of shape (frames_per_sequence, 4)
                where the 4 labels correspond to bounding box corners (x1, y1, x2, y2)
        """


        #
        pass

"""
Pass a fovea stack to the network, consisting of the current fovea and the previous N foveas

Dataloader should pass sequences of images to the model 
batch the sequences across videos
Playing backward is ok

"Warm-start" the model on a single sequence to generate all of the foveas,
then store them and feed them to the model along with the 

Want the concept of multiple "workers" so we can eventually have multiple fovea
"""
if __name__ == "__main__":  
    base_dir = r'/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data'
    base_dir += r'/got10k/val'
    ds = GOT10kDataset(base_dir)
    print(ds)
    print(len(ds))
    print(ds[4])
