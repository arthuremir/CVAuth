import os
from pathlib import Path

from cvauth.arcface.config import get_config
from cvauth.arcface.learner import FaceLearner
from cvauth.arcface.mtcnn import MTCNN
from cvauth.arcface.utils import load_facebank, prepare_facebank
from utils.predictor import VisualizationDemo
from utils.setup_detectron import setup_detectron_cfg
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

    def __init__(self, args, if_face, if_pose, if_hand):
        self.if_face = if_face.get()
        self.if_pose = if_pose.get()
        self.if_hand = if_hand.get()

        self.conf = get_config(False)

        self.args = args

        self.hand_box = [100, 440, 300, 640]
        self.num_frames = 0

        self.bg = None

        if self.if_face:
            self.detector, self.learner, self.targets, self.names = load_face_rec(self.conf, self.args)
            print('Face detector is loaded!')

        if self.if_pose:
            self.detectron = load_pose_estim(self.args)
            print('Pose estimator is loaded!')

        if self.if_hand:
            # self.init_hand()
            pass

    def init_hand(self):
        self.hand_localizer = load_hand_rec(self.args, self.conf.device)
        print('Hand localizer is loaded!')

    def run(self, frame, username):

        face_status = 0

        if self.if_pose:
            _, vis = self.detectron.run_on_image(frame)
            vis = vis.get_image()
        else:
            vis = frame[:, :, ::-1]

        if self.if_face:
            faces = detect_faces(frame, self.detector, self.conf, self.learner, self.targets, self.args.tta)
            if faces is not None:
                bboxes, results, score = faces
                # print(bboxes, results, score)
                if len(results) > 1:
                    print('Too many faces on the image!')
                else:
                    name_predicted = self.names[results[0] + 1]
                    print(name_predicted)
                    if name_predicted == username:
                        face_status = 1
                        vis = visualize_face(vis, bboxes[0], score[0], name_predicted, self.args.score)
                    else:
                        face_status = -1
                        print('Face misclassification!')

        if self.if_hand:

            hand_roi = frame[self.hand_box[0]:self.hand_box[2], self.hand_box[1]:self.hand_box[3]]

            gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if self.num_frames < 30:
                self.bg = run_avg(self.bg, gray, 0.5)
                if self.num_frames == 1:
                    print("[STATUS] please wait! calibrating...")
                elif self.num_frames == 29:
                    print("[STATUS] calibration successfull...")
            else:
                hand = segment(self.bg, gray)

                if hand is not None:
                    thresholded, segmented = hand

                    #thresholded = 255 - thresholded
                    #thresholded /= 255

                    thresholded = np.stack((thresholded,) * 3, axis=-1)
                    #print(thresholded.max())
                    #print(vis.max())
                    vis[self.hand_box[0]:self.hand_box[2], self.hand_box[1]:self.hand_box[3]] = thresholded
                    #print(vis.max())
                    # vis = cv2.drawContours(vis, [segmented + (self.hand_box[1], self.hand_box[0])], -1, (0, 0, 0))

                    '''for size in [5]:  # [3, 5, 7, 9]:
                        for iter in [1]:  # range(1, 3):
                            closing = cv2.morphologyEx(thresholded,
                                                       cv2.MORPH_CLOSE,
                                                       np.ones((size, size), np.uint8),
                                                       iterations=iter)
                            cv2.imshow("{}-size {}-iter".format(size, iter), closing)'''

                    # cv2.imshow("Thresholded", thresholded)
                else:
                    pass
                    # vis[self.hand_box[0]:self.hand_box[2], self.hand_box[1]:self.hand_box[3]] = 255

            vis = cv2.rectangle(vis, (self.hand_box[3], self.hand_box[0]), (self.hand_box[1], self.hand_box[2]),
                                (0, 255, 0), 2)

            self.num_frames += 1

            '''hand_bboxes = detect_hands(frame, self.hand_localizer, self.conf.device, self.args)
            if hand_bboxes is not None:
                vis = visualize_hand(vis, hand_bboxes)'''

        return vis.get() if isinstance(vis, type(cv2.UMat())) else np.array(vis, dtype=np.uint8), face_status
