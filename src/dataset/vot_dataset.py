from torch.utils.data import Dataset 
from torchvision.io import read_image
import torch
from os import path as Path 
import os 
from tqdm import tqdm

# /scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/foveated_vision/
# LaSOT dataset 
basedir_ = '/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data/vot/'
directories = {'shortterm': Path.join(basedir_,'shortterm'),
               'longterm':Path.join(basedir_,'longterm')}

class VotDataset(Dataset): 
    def __init__(self, dataset_name='longterm'):
        # get list of video sequences: 
        self.basedir = directories[dataset_name]
        with open(Path.join(self.basedir,'sequences','list.txt'),'r') as seqfile: 
            self.videos = [line.strip() for line in seqfile.readlines() if len(line.strip()) > 0 ] 
        # self.cached_imgs = Path.join(self.basedir, 'img_caches')
        # self.cached_labels = Path.join(self.basedir, 'img_labels')
        # os.makedirs(self.cached_imgs,exist_ok=True)
        # os.makedirs(self.cached_labels,exist_ok=True)
        # for i in tqdm(range(len(self))): 
        #     self.preprocess_image(i,reprocess=False)
        
    def preprocess_image(self,imnum,reprocess:bool=False):
        if (reprocess or 
                (not Path.isfile(Path.join(self.cached_imgs,f'{imnum}.pt')) 
                or not Path.isfile(Path.join(self.cached_labels,f'{imnum}.pt')))):
            img, yval = self.get_vid(imnum)
            torch.save(img,Path.join(self.cached_imgs,f'{imnum}.pt'))
            torch.save(yval,Path.join(self.cached_labels,f'{imnum}.pt'))

    def __len__(self): 
        # wut 
        return len(self.videos)
    
    def get_vid(self, idx):
        viddir = Path.join(self.basedir,'sequences',self.videos[idx],)
        with open(Path.join(viddir,'groundtruth.txt'),'r') as file: 
            groundtruth = torch.tensor([[float(elt) for elt in line.split(',')] for line in file.readlines() if len(line) != 0])
        first = read_image(Path.join(viddir,f'color/{1:08d}.jpg'))
        print(groundtruth.shape)
        images = torch.zeros( [groundtruth.shape[0]] + list(first.shape))
        for i in range(groundtruth.shape[0]): 
            images[i] = read_image(Path.join(viddir,f'color/{i+1:08d}.jpg'))
        print(images.shape)
        return images,groundtruth
    
    def __getitem__(self,idx):
<<<<<<< HEAD:src/dataset/dataloader.py
        return self.get_vid(idx)
    
=======

        imgs =  torch.load(Path.join(self.cached_imgs,f'{idx}.pt'))
        labels =  torch.load(Path.join(self.cached_labels,f'{idx}.pt'))
        return imgs,labels

>>>>>>> 8b202003da8c6810ebcf4357b987742b879e9129:src/dataset/vot_dataset.py
if __name__ == "__main__": 
    ds = VotDataset()
    print(ds)
    print(len(ds))
    print(ds[4])
