"""
Obtain evaluation metrics for our models:
1. AUC: Area Under the Curve of bbox overlap threshold vs success rate 
    (define "success" as passing a given overlap threshold, then plot the curve of  
    the fraction of the time you achieve success given different overlap thresholds.
    Finally, take the area under this curve)
2. AO: Average Overlap (Mean bbox IoU across all frames and all videos)
https://paperswithcode.com/sota/visual-object-tracking-on-got-10k
https://paperswithcode.com/sota/visual-object-tracking-on-vot201718
"""

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k, ExperimentVOT
import json 
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from peripheral_foveal_vision_model import PeripheralFovealVisionModel
from torchvision.io import read_image
import logging
from torchvision.transforms import PILToTensor, Resize
from utils import bbox_to_img_coords, corners_to_corners_width
# pip3 install got10k
# https://github.com/got-10k/toolkit
class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box

# https://github.com/got-10k/siamfc/blob/master/siamfc.py
class FoveatedVisionTracker(Tracker):
    def __init__(self, model_filepath, targ_size=None):
        super(FoveatedVisionTracker, self).__init__(name='FoveatedVisionTracker',
                                                    is_deterministic=True)
        self.model_filepath = model_filepath
        self.model = PeripheralFovealVisionModel()
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cpu") #
        self.model.to(self.device)
        self.model.eval()
        logging.info(f"Loaded Tracker model. Device: {self.device} Filepath: {self.model_filepath}")
        if targ_size is None:
            self.resize = lambda x:x
        else: 
            self.resize = Resize(size=targ_size,antialias=True)
        self.transform = PILToTensor()

    def transform_PIL_image(self, PIL_image):
        image_tensor = self.resize(self.transform(PIL_image)).float().to(self.device)
        return image_tensor

    def init(self, image, box):
        # TODO: should our model take in the starting bounding box as well?
        self.model.reset()
        with torch.no_grad():
            image_tensor = self.transform_PIL_image(image)
            self.image_shape = image_tensor.shape
            self.model(image_tensor)
    
    def update(self, image):
        with torch.no_grad():
            image_tensor = self.transform_PIL_image(image)
            bbox, next_fixation = self.model(image_tensor)
            bbox_GOT10k = self.transform_bboxes(bbox, image_tensor)
            return bbox_GOT10k

    def transform_bboxes(self, bbox, image):
        """ Convert an Nx4 tensor of 0-1 fractions [xmin, ymin, xmax, ymax]
        into GOT10k standard [xmin, ymin, width, height] (in pixels), 
        detached and on CPU
        """
        img_coords = bbox_to_img_coords(bbox, image)
        bbox_corners_width = corners_to_corners_width(img_coords)
        return bbox_corners_width.detach().cpu()
    
    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)
        
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot'+extension)
        key = 'overall'
        
        # filter performance by tracker_names
        performance = {k:v for k,v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]
        
        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))
        
        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots on GOT-10k')
        ax.grid(True)
        fig.tight_layout()
        
        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)
    


# TODO: do I need a separate Tracker for VOT and GOT10k? 

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # setup tracker
    # model_filepath = r'/home/alovekin/foveated_vision/models/20231206_185942_model_epoch_2_step_3760.pth'
    model_filepath_base = r'./models/'
    model_filepath = model_filepath_base + r'20231206_185942_model_epoch_2_step_3760.pth'
    tracker = FoveatedVisionTracker(model_filepath, targ_size=(224,224))

    # # run experiments on VOT
    # logging.info('Running experiments on VOT')
    # experiment = ExperimentVOT(
    #     root_dir=r'/scratch/eecs542s001f23_class_root/eecs542s001f23_class/shared_data/group_raz/data/vot/longterm/sequences',
    #     subset='val', #note that 'test' ground-truth is withheld
    #     result_dir='results',
    #     report_dir='reports')
    # experiment.run(tracker, visualize=False)

    # run experiments on GOT-10k (validation subset)
    logging.info('Running experiments on GOT-10k (validation subset)...')
    experiment = ExperimentGOT10k(
        root_dir=r'/scratch/engin_root/engin1/shared_data/group_raz/data/got10k',
        subset='val', #note that 'test' ground-truth is withheld
        result_dir='results',
        report_dir='reports')
    # experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])

