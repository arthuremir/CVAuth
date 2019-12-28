import os
from pathlib import Path

from cvauth.arcface.config import get_config
from cvauth.arcface.learner import FaceLearner
from cvauth.arcface.mtcnn import MTCNN
from cvauth.arcface.utils import load_facebank, prepare_facebank
from utils.predictor import VisualizationDemo
from utils.setup_detectron import setup_detectron_cfg
from utils.arg_parser import get_args
from utils.detectors import *
from utils.face_utils import *
from utils.hand_utils import *

SAVE_FOLDER = Path('cvauth/arcface/data/facebank')
DIRTY_FLAG = SAVE_FOLDER / ".dirty"

def load_face_rec(conf, args):
    detector = MTCNN()
    learner = FaceLearner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()

    if args.update or os.path.exists(DIRTY_FLAG):
        targets, names = prepare_facebank(conf, learner.model, detector, tta=args.tta)
        print('Facebank is updated!')
        os.remove(DIRTY_FLAG)
    else:
        targets, names = load_facebank(conf)
        print('Facebank is loaded!')

    return detector, learner, targets, names


def load_pose_estim(args):
    cfg = setup_detectron_cfg(args)
    return VisualizationDemo(cfg)


def load_hand_rec(args, device):
    return prepare_hand_localizer(args.trained_model, args.cpu, device)


class Visualizer:

    def __init__(self, if_face, if_pose, if_hand):
        self.if_face = if_face.get()
        self.if_pose = if_pose.get()
        self.if_hand = if_hand.get()

        self.conf = get_config(False)

        self.args = get_args()

        if self.if_face:
            self.detector, self.learner, self.targets, self.names = load_face_rec(self.conf, self.args)
            print('Face detector is loaded!')

        if self.if_pose:
            self.detectron = load_pose_estim(self.args)
            print('Pose estimator is loaded!')

        if self.if_hand:
            self.hand_localizer = load_hand_rec(self.args, self.conf.device)
            print('Hand localizer is loaded!')

    def run(self, frame):

        if self.if_pose:
            _, vis = self.detectron.run_on_image(frame)
            vis = vis.get_image()
        else:
            vis = frame[:, :, ::-1]

        if self.if_face:
            faces = detect_faces(frame, self.detector, self.conf, self.learner, self.targets, self.args.tta)
            if faces is not None:
                vis = visualize_face(vis, faces, self.names, self.args.score)

        if self.if_hand:
            hand_bboxes = detect_hands(frame, self.hand_localizer, self.conf.device, self.args)
            if hand_bboxes is not None:
                vis = visualize_hand(vis, hand_bboxes)

        return vis.get() if isinstance(vis, type(cv2.UMat())) else np.array(vis, dtype=np.uint8)
