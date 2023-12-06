from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
# pip3 install got10k
# https://github.com/got-10k/toolkit
class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')
    
    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box

if __name__ == '__main__':
    # setup tracker
    tracker = IdentityTracker()

    # run experiments on GOT-10k (validation subset)
    got10k_path = r'/scratch/engin_root/engin1/shared_data/group_raz/data/got10k'
    experiment = ExperimentGOT10k(got10k_path, subset='val')
    experiment.run(tracker, visualize=True)

    # report performance
    experiment.report([tracker.name])