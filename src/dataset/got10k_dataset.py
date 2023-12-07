import os

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize

"""
Dataset Properties:
# Videos are different length temporally, and also different resolution
# Min Resolution: (480, 360)
# Max Resolution: (3840, 2160)
# min duration (sec): 0.4 (~40 frames)
# max duration (sec): 148 (~14800 frames)
# avg duration (sec): 15 (~150 frames)

# 4-second segments: 87% use of frames
# 5-second segments: 83% use of frames
# 6-second segments: 75% use of frames
"""

GOT10K_SAMPLE_RATE_HZ = 10 

class GOT10kDataset(Dataset):
    """ Creates video sample sequences from the GOT-10k dataset 
        A sample consists of a sequence of images of fixed length and resolution, 
        along with a sequence of bounding box labels
    """
    def __init__(self, base_dir, sample_resolution=(420,420), sample_seconds:int=6) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.frames_per_sequence = sample_seconds * GOT10K_SAMPLE_RATE_HZ
        self.video_list_filepath = os.path.join(self.base_dir, 'list.txt')

        self.verify_dataset_validity()

        self.resize = Resize(size=sample_resolution, antialias=True) if sample_resolution is not None else lambda x:x
        # find the number of videos in the folder, and number of total frames in all videos
        # Video folders in the form GOT-10k_Val_000180
        # Images are in the form 00000000.jpg
        self.n_videos = 0
        self.n_frames = 0
        self.n_samples = 0
        self.sample_directories = [] # list of tuples (video_name, index_from_video_start)
        with open(self.video_list_filepath, 'r') as f:
            self.video_filenames = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
            self.n_videos = len(self.video_filenames)
            n_videos2 = len(os.listdir(self.base_dir))-1 # subtract 1 for list.txt
            assert n_videos2 == self.n_videos, f'list.txt has {self.n_videos} lines, but the directory has {n_videos2} videos'
            
        # assign each video an index corresponding to the sample index
        for video_name in self.video_filenames:
            video_dir = os.path.join(self.base_dir, video_name)
            with open(video_dir + '/groundtruth.txt', 'r') as gt_file:
                n_video_frames = sum(1 for line in gt_file.readlines() if len(line) != 0)
                # vvv was triggering because there was 1 additional backup file in the directory
                # n_video_frames2 = len(os.listdir(video_dir)) - 5 # subtract 5 for info files
                # assert n_video_frames == n_video_frames2, f'Directory {video_dir} has {n_video_frames2} video files + 5 info files, which does not match number of ground-truth bounding boxes ({n_video_frames})'
                
                n_video_samples = n_video_frames // self.frames_per_sequence
                self.sample_directories += zip( n_video_samples * [video_name],[i for i in range(n_video_samples)])
                
                self.n_frames += n_video_frames
                self.n_samples += n_video_samples
        
        print(f'Found a total of {self.n_frames} frames in {self.n_videos} videos, which yields {self.n_samples} sequences of length {self.frames_per_sequence}. ({float(self.n_samples*self.frames_per_sequence)/float(self.n_frames)*100.0:.2f}% of frames used)')            

    def verify_dataset_validity(self):
        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f'Base directory {self.base_dir} not found')
        if not os.path.basename(self.base_dir) in ['train', 'val', 'test']:
            raise ValueError('Dataset must come from one of the folders "/train", "/val", or "/test"')
        if not os.path.exists(self.video_list_filepath):
            raise FileNotFoundError(f'list.txt not found in the base directory {self.base_dir}')
        
    def __len__(self) -> int:
        return len(self.sample_directories)

    def __getitem__(self, idx: int):
        """
        Returns:
            image_sequence: (torch.Tensor) of shape (frames_per_sequence, 3, ?, ?)
            label_sequence: (torch.Tensor) of shape (frames_per_sequence, 4)
                where the 4 labels correspond to bounding box corners [xmin, ymin, width, height]
        """
        # ## Method 1: Load entire videos with whatever resolution and length they have
        # video_dir = os.path.join(self.base_dir, self.video_filenames[idx].strip())

        # # Load ground Truth labels
        # # Each line of this file is a sequence of 4 floats, separated by commas: [xmin, ymin, width, height]     
        # with open(os.path.join(video_dir,'groundtruth.txt'),'r') as file: 
        #     ground_truth = torch.tensor([[float(x) for x in line.split(',')] for line in file.readlines() if len(line) != 0])

        # ## Load the entire video file (idx refers to which video to load)
        # first = read_image(os.path.join(video_dir,f'{1:08d}.jpg')) # determine image shape
        # image_sequence = torch.zeros( [ground_truth.shape[0]] + list(first.shape))
        # for i in range(ground_truth.shape[0]):
        #     image_sequence[i] = read_image(os.path.join(video_dir, f'{i+1:08d}.jpg'))

        # return image_sequence, ground_truth
    
        ## Method 2: Load video sequences of fixed length and resolution
        video_name, sample_idx = self.sample_directories[idx]
        video_dir = os.path.join(self.base_dir, video_name)
        start_idx = sample_idx*self.frames_per_sequence
        end_idx = (sample_idx+1)*self.frames_per_sequence
        
        # Load ground Truth labels
        # Each line of this file is a sequence of 4 floats, separated by commas: [xmin, ymin, width, height]
        with open(os.path.join(video_dir,'groundtruth.txt'),'r') as file:
            ground_truth = torch.tensor([[float(x) for x in line.split(',')] for line in file.readlines() if len(line) != 0])
            ground_truth = ground_truth[start_idx:end_idx,:]
        
        # Format the image sequence into the correct resolution and length
        first_img = self.resize(read_image(os.path.join(video_dir,f'{1:08d}.jpg')))
        image_sequence = torch.zeros([self.frames_per_sequence] + list(first_img.shape))
        for i in range(self.frames_per_sequence):
            image_path = os.path.join(video_dir, f'{start_idx + i+1:08d}.jpg') # add 1 because the first image is 00000001.jpg
            image_sequence[i] = self.resize(read_image(image_path)) / 255.0 # Normalize (TODO: should be handled by dataloader transforms)

        return image_sequence, ground_truth



        




"""
The network will Pass the fovea stack (consisting of the current fovea and the previous N foveas) to itself
Within the network we want the concept of multiple "workers" so we can eventually have multiple fovea

Can we pre-process the data so that each clip is already the right length and size of image?

The dataset is responsible for saying how to get a single "sample" 
(we need to decide whether an "sample" is: a whole video or a smaller image sequence)
A sample would be what the model receives at runtime (ie a sequence of images over time)

The dataloader is responsible for batching the items together and shuffling them, 
as well as applying any transforms to the data, so that training is efficient and effective.

batch the sequences across videos

Images are all different resolutions; we need to resize them:
    - resize to the minimum resolution in the dataset (480, 360) -- useful for feeding in as the background information
    - resize to some middle resolution in the dataset (1440, 1080) -- balance between computation cost and information loss

    Might want to flip the vertical videos 90 degrees before resizing

"Warm-start" the model on a single sequence to generate all of the foveas,
then store them and feed them to the model along with the 



Use photometric augmentation transforms like 
ColorJitter
RandomGrayscale
Random Invert
Random Posterize
Random Equalize

RandomHorizontalFlip
RandomVerticalFlip
(use RandomApply to apply all of the above with some probabilities)

TODO: Calculate the mean across all images in the dataset for each color channel (which fits the way resnet was trained)

"""
if __name__ == "__main__":  
    base_dir = r'/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data'
    base_dir += r'/got10k/val'
    # base_dir = r'C:\Users\arthu\OneDrive\Documents\Classwork\EECS542_Adv_computer_vision\got10k\val'

    ds = GOT10kDataset(base_dir)
    print(f"Dataset has {len(ds)} videos")
    assert len(ds) == 180, f'Expected 180 videos, but got {len(ds)}'
    print(f"Shape of 4th sequence: {ds[4][0].shape}")
    print(f"Shape of the 180th sequence (index 179): {ds[179][0].shape}")
    print(f"First few ground truth labels of 4th video: {ds[4][1][:5]}")
    try:
        print(ds[len(ds)])
    except IndexError:
        print(f'trying to get the {len(ds)+1}th sequence raises an IndexError')

