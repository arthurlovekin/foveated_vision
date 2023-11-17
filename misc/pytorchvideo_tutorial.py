# pip3 install pytorch-lightning
# pip3 install pytorchvideo
# https://pytorchvideo.org/docs/tutorial_classification
import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import pytorchvideo.models.resnet
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

# TODO: Make another DataModule for the LaSot and VOT datasets

class KineticsDataModule(pytorch_lightning.LightningDataModule):
# dataset_filepath = r"/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data/kinetics"
# classnames_filepath = dataset_filepath + "/kinetics_classnames.json"
# video_filepath = dataset_filepath + "/archery.mp4"
  # Dataset configuration
  _DATA_PATH = r"/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data/kinetics"
  _CLIP_DURATION = 2  # Duration of sampled clip for each video
  _BATCH_SIZE = 8
  _NUM_WORKERS = 8  # Number of parallel processes fetching data

  def train_dataloader(self):
      """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train.csv. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform, decode_audio=False
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

  def val_dataloader(self):
    """
    Create the Kinetics validation partition from the list of video labels
    in {self._DATA_PATH}/val
    """
    val_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(self._DATA_PATH, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
        decode_audio=False,
    )
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=self._BATCH_SIZE,
        num_workers=self._NUM_WORKERS,
    )import pytorchvideo

# pytorchvideo.data.Kinetics clips have the following dictionary format
#   {
#      'video': <video_tensor>,     # Shape: (C, T, H, W)
#      'audio': <audio_tensor>,     # Shape: (S)
#      'label': <action_label>,     # Integer defining class annotation
#      'video_name': <video_path>,  # Video file path stem
#      'video_index': <video_id>,   # index of video used by sampler
#      'clip_index': <clip_id>      # index of clip sampled within video
#   }



