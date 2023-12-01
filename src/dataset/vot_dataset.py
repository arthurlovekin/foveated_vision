from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize
import torch
from os import path as Path 

# /scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/foveated_vision/
# LaSOT dataset 
basedir_ = '/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data/vot/'
directories = {'shortterm': Path.join(basedir_,'shortterm'),
               'longterm':Path.join(basedir_,'longterm')}

class VotDataset(Dataset): 
    def __init__(self, dataset_name='longterm',targ_size = None):
        # get list of video sequences: 
        self.basedir = directories[dataset_name]
        if targ_size is None:
            self.resize = lambda x:x
        else: 
            self.resize = Resize(size=targ_size,antialias=True)
        with open(Path.join(self.basedir,'sequences','list.txt'),'r') as seqfile: 
            self.videos = [line.strip() for line in seqfile.readlines() if len(line.strip()) > 0 ] 

        vidlengths = []
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
        return len(self.videos)
    
    def get_vid(self, idx):
        viddir = Path.join(self.basedir,'sequences',self.videos[idx],)
        with open(Path.join(viddir,'groundtruth.txt'),'r') as file: 
            groundtruth = torch.tensor([[float(elt) for elt in line.split(',')] for line in file.readlines() if len(line) != 0])
        first = self.resize(read_image(Path.join(viddir,f'color/{1:08d}.jpg')))
        print(groundtruth.shape)
        images = torch.zeros( [groundtruth.shape[0]] + list(first.shape))
        for i in range(groundtruth.shape[0]): 
            images[i] = self.resize(read_image(Path.join(viddir,f'color/{i+1:08d}.jpg')))
        
        return images,groundtruth
    
    def __getitem__(self,idx):
        return self.get_vid(idx)


def get_dataloader(dataset_name='longterm',targ_size = (250,250),batch_size=3,shuffle=True,**loader_kwargs):
    collate_fn = lambda x: x
    ds = VotDataset(dataset_name=dataset_name,targ_size=targ_size)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,**loader_kwargs,collate_fn=collate_fn)
    
if __name__ == "__main__": 
    ds = VotDataset()
    print(ds)
    print(len(ds))
    print(ds[4])

    dl = get_dataloader()
    print(next(iter(dl)))