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
        if not os.path.exists(os.path.join(self.base_dir, 'list.txt')):
            raise FileNotFoundError('list.txt not found in the base directory')
        
        # find the number of videos in the folder, and number of total frames in all videos
        # Video folders in the form GOT-10k_Val_000180
        # Images are in the form 00000000.jpg
        self.n_videos = 0
        self.n_frames = 0
        # self.video_lengths = []
        with open(os.path.join(self.base_dir, 'list.txt'), 'r') as f:
            self.video_filenames = f.readlines()
            self.n_videos = len(os.listdir(self.base_dir))-1 # subtract 1 for list.txt
            assert len(self.video_filenames) == self.n_videos, f'list.txt has {len(self.video_filenames)} lines, but the directory has {self.n_videos} videos'
            
            for line in self.video_filenames:
                if len(line.strip()) > 0:
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
        return self.n_videos
        # TODO: Might be n_sequences instead of n_videos

    def __getitem__(self, idx: int):
        """
        Returns:
            image_sequence: (torch.Tensor) of shape (frames_per_sequence, 3, ?, ?)
            label_sequence: (torch.Tensor) of shape (frames_per_sequence, 4)
                where the 4 labels correspond to bounding box corners (x1, y1, x2, y2)
        """
        ## Method 1: Load entire videos with whatever resolution and length they have
        video_dir = os.path.join(self.base_dir, self.video_filenames[idx].strip())

        # Load ground Truth labels
        # Each line of this file is a sequence of 4 floats, separated by commas: x1, y1, x2, y2        
        with open(os.path.join(video_dir,'groundtruth.txt'),'r') as file: 
            ground_truth = torch.tensor([[float(x) for x in line.split(',')] for line in file.readlines() if len(line) != 0])

        ## Load the entire video file (idx refers to which video to load)
        first = read_image(os.path.join(video_dir,f'{1:08d}.jpg')) # determine image shape
        image_sequence = torch.zeros( [ground_truth.shape[0]] + list(first.shape))
        for i in range(ground_truth.shape[0]):
            image_sequence[i] = read_image(os.path.join(video_dir, f'{i+1:08d}.jpg'))

        return image_sequence,ground_truth
    
        ## Method 2: Load video sequences of fixed length and resolution
        # # (Each video would have multiple sequences)
        # # then batching is easier and we don't have to define a collate function
        # # then idx would refer to the sequence (might want to track number of sequences in each video so that we can quickly index to the correct video)
        # image_sequence = torch.zeros((self.frames_per_sequence, 3, 1080, 1920))


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


"""
if __name__ == "__main__":  
    base_dir = r'/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data'
    base_dir += r'/got10k/val'
    # base_dir = r'C:\Users\arthu\OneDrive\Documents\Classwork\EECS542_Adv_computer_vision\got10k\val'

    ds = GOT10kDataset(base_dir)
    print(f"Dataset has {len(ds)} videos")
    print(f"Shape of 4th video: {ds[4][0].shape}")
    print(f"Shape of the 180th video (index 179): {ds[179][0].shape}")
    print(f"First few ground truth labels of 4th video: {ds[4][1][:5]}")
    try:
        print(ds[180])
    except IndexError:
        print('trying to get the 180th video raises an IndexError')

