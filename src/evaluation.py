from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
import torch
from peripheral_foveal_vision_model import PeripheralFovealVisionModel

# pip3 install got10k
# https://github.com/got-10k/toolkit
class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box

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
        self.model(image)
    
    def update(self, image):
        bbox, next_fixation = self.model(image)
        return bbox


if __name__ == '__main__':
    # setup tracker
    # model_filepath = r'/home/alovekin/foveated_vision/models/20231206_185942_model_epoch_2_step_3760.pth'
    model_filepath_base = r'./models/'
    model_filepath = model_filepath_base + r'20231206_185942_model_epoch_2_step_3760.pth'
    tracker = FoveatedVisionTracker()

    # run experiments on GOT-10k (validation subset)
    got10k_path = r'/scratch/engin_root/engin1/shared_data/group_raz/data/got10k'
    experiment = ExperimentGOT10k(got10k_path, subset='val')
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])