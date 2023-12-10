from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
import torch
from peripheral_foveal_vision_model import PeripheralFovealVisionModel
from torchvision.io import read_image
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
    def __init__(self, model_filepath):
        super(FoveatedVisionTracker, self).__init__(name='FoveatedVisionTracker',
                                                    is_deterministic=True)
        self.model_filepath = model_filepath
        self.model = PeripheralFovealVisionModel()
        self.model.load_state_dict(torch.load(self.model_filepath))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def init(self, image, box):
        # TODO: should our model take in the starting bounding box as well?
        self.model(read_image(image))
    
    def update(self, image):
        bbox, next_fixation = self.model(image)
        return bbox


if __name__ == '__main__':
    # setup tracker
    # model_filepath = r'/home/alovekin/foveated_vision/models/20231206_185942_model_epoch_2_step_3760.pth'
    model_filepath_base = r'./models/'
    model_filepath = model_filepath_base + r'20231206_185942_model_epoch_2_step_3760.pth'
    tracker = FoveatedVisionTracker(model_filepath)

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k(
        root_dir=r'/scratch/engin_root/engin1/shared_data/group_raz/data/got10k',
        subset='val', #note that test ground-truth is withheld
        result_dir='results',
        report_dir='reports')
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])