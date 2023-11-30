from torch.utils.data import Dataset 
from torchvision.io import read_image
import torch
import os 
from tqdm import tqdm

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
        with open(os.path.join(self.base_dir, 'list.txt'), 'r') as f:
            lines = f.readlines()
            print(f"Loading Dataset ...")
            for line in tqdm(lines):
                self.n_videos += 1
                # get the number of frames in the video
                video_dir = os.path.join(self.base_dir, line.strip())
                if not os.path.exists(video_dir):
                    print(f'Video directory {video_dir} not found')

                for file in os.listdir(video_dir):
                    if file.split('.')[1] == 'jpg':
                        self.n_frames += 1

            


    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, index: int):  
        pass

if __name__ == "__main__":  
    base_dir = r'/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data'
    base_dir += r'/got10k/val'
    ds = GOT10kDataset(base_dir)
    print(ds)
    print(len(ds))
    print(ds[4])
