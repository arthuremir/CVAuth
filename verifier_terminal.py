from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import time
from datetime import datetime

import cv2
import numpy as np
import tqdm
from PIL import Image

from detectron2.config import get_cfg as get_detectron_cfg
from detectron2.utils.logger import setup_logger

from arcface.config import get_config
from arcface.learner import FaceLearner
from arcface.mtcnn import MTCNN
from arcface.mtcnn_pytorch.src.visualization_utils import show_bboxes
from arcface.utils import load_facebank, draw_box_name, prepare_facebank, get_facebank_names

from hand_detection.data import cfg as hand_cfg
from hand_detection.layers.functions.prior_box import PriorBox
from hand_detection.utils.box_utils import decode
from hand_detection.utils.py_cpu_nms import py_cpu_nms as nms
from utils.hand_utils import *
from utils.predictor import VisualizationDemo


def get_args():
    parser = argparse.ArgumentParser(description='Face verification app')
    # parser.add_argument("-s", "--save", help="whether save", action="store_true")

    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54,
                        type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")

    parser.add_argument(
        "--config-file",
        default="detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('-m', '--trained_model', default='hand_detection/weights/Final_HandBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

    # parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')

    return parser.parse_args()


def setup_detectron_cfg(args):
    detectron_cfg = get_detectron_cfg()
    detectron_cfg.merge_from_file(args.config_file)
    detectron_cfg.merge_from_list(args.opts)

    detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    detectron_cfg.freeze()
    return detectron_cfg


def detect_hands(frame, localizer, device):
    img = np.float32(frame)

    resize = 2
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    out = localizer(img)

    prior_box = PriorBox(hand_cfg, out[2], (im_height, im_width), phase='test')
    priors = prior_box.forward()
    priors = priors.to(device)
    loc, conf, _ = out
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, hand_cfg['variance'])

    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)

    keep = nms(dets, args.nms_threshold)  # , force_cpu=args.cpu)
    dets = dets[keep, :]

    dets = dets[:args.keep_top_k, :]

    return dets


def detect_faces(frame):
    image = Image.fromarray(frame[..., ::-1])

    time_detect_init = time.time()
    aligned_faces = detector.align_multi(image, conf.face_limit, conf.min_face_size)
    time_detect = time.time() - time_detect_init

    if aligned_faces is None:
        return None
    bboxes, faces = aligned_faces[0], aligned_faces[1]
    bboxes = bboxes[:, :-1]
    bboxes = bboxes.astype(int)
    bboxes = bboxes + [-1, -1, 1, 1]

    time_classif_init = time.time()
    results, score = learner.infer(conf, faces, targets, args.tta)
    time_classif = time.time() - time_classif_init

    return [bboxes, results, score], [time_detect, time_classif]


if __name__ == '__main__':

    conf = get_config(False)

    facebank_path = conf.facebank_path
    facebank_names = get_facebank_names(facebank_path)

    print('Let\'s start verification process.')
    while True:
        username = input('Please, type your name:\n').lower()
        if username not in facebank_names:
            ans = input('First time? [Y/n]\n').lower()
            if ans == 'y':
                new_user = True
                break
            else:
                print('Try again.')
                continue

        else:
            new_user = False
            break

    mp.set_start_method("spawn", force=True)
    logger = setup_logger()

    args = get_args()

    cfg = setup_detectron_cfg(args)

    detectron_flow = VisualizationDemo(cfg)

    user_path = facebank_path / username

    cap = cv2.VideoCapture(0)
    cap.set(3, conf.window_size[0])
    cap.set(4, conf.window_size[1])

    print('Camera loaded!')

    cv2.namedWindow(conf.window_name)
    cv2.moveWindow(conf.window_name, *conf.window_move)

    detector = MTCNN()

    if new_user:
        os.mkdir(user_path)

        time_init = time.time()
        frames_num = 0

        while cap.isOpened():

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            is_success, frame = cap.read()
            frames_num += 1

            if is_success:
                frame_text = cv2.putText(frame,
                                         'Press t to take a picture,q to quit...',
                                         (10, 100),
                                         cv2.FONT_HERSHEY_PLAIN,
                                         1,
                                         (0, 255, 0),
                                         3,
                                         cv2.LINE_AA)
                cv2.imshow(conf.window_name, frame_text)

                p = Image.fromarray(frame[..., ::-1])
                detect_res = detector.align(p)
                if detect_res is None:
                    cv2.imshow(conf.window_name, frame)
                    continue

                bboxes, aligned_face, landmarks = detect_res

                bboxes = bboxes[:, :-1]
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1, -1, 1, 1]

                # p = Image.fromarray(frame[..., ::-1])
                frame = show_bboxes(Image.fromarray(frame), bboxes, landmarks)
                frame = np.array(frame)
                cv2.imshow(conf.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('t'):
                    warped_face = np.array(aligned_face)[..., ::-1]
                    cv2.imwrite(
                        str(user_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
                        warped_face)
                    print('Image saved!')

        time_fin = time.time() - time_init
        print('Total frames: {0}\nTotal time: {1:.2f}\nFPS: {2:.2f}'.format(frames_num,
                                                                            time_fin,
                                                                            frames_num / time_fin))

    else:
        learner = FaceLearner(conf, True)
        learner.threshold = args.threshold
        if conf.device.type == 'cpu':
            learner.load_state(conf, 'cpu_final.pth', True, True)
        else:
            learner.load_state(conf, 'final.pth', True, True)
        learner.model.eval()
        print('FaceLearner loaded')

        if args.update:
            targets, names = prepare_facebank(conf, learner.model, detector, tta=args.tta)
            print('Facebank updated')
        else:
            targets, names = load_facebank(conf)
            print('Facebank loaded')

        hand_localizer = prepare_hand_localizer(args.trained_model, args.cpu, conf.device)

        time_init = time.time()

        num_frames = 0
        time_detect = 0
        num_detect = 0
        time_classif = 0

        while True:
            _, frame = cap.read()
            _, vis = detectron_flow.run_on_image(frame)
            vis = vis.get_image()
            #vis = np.array(vis)
            #print(type(frame), type(vis))
            #print(vis)
            #cv2.imshow(conf.window_name, vis.get_image())
            #cv2.imshow("SECOND", vis)
        #for frame, vis in tqdm.tqdm(detectron_flow.run_on_video(cap)):
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            num_frames += 1

            dets = detect_hands(frame, hand_localizer, conf.device)

            if dets is not None:
                for i in range(dets.shape[0]):
                    cv2.rectangle(vis, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)
                    vis = cv2.putText(vis,
                                      "hand",
                                      (dets[i][0], dets[i][1]),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      2,
                                      (0, 255, 0),
                                      3,
                                      cv2.LINE_AA)

            faces = detect_faces(frame)

            if faces is not None:
                bboxes, results, score = faces[0]
                time_detect += faces[1][0]
                time_classif += faces[1][1]
                num_detect += 1
                for idx, bbox in enumerate(bboxes):
                    if args.score:
                        vis = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), vis)
                    else:
                        vis = draw_box_name(bbox, names[results[idx] + 1], vis)

            cv2.imshow(conf.window_name, vis)

        time_fin = time.time() - time_init
        print('Total frames: {0}\nTotal time: {1:.2f}\nFPS: {2:.2f}'.format(num_frames,
                                                                            time_fin,
                                                                            num_frames / time_fin))

        if num_detect != 0:
            print('Average face detection time: {0:.2f}s'.format(time_detect / num_detect))
            print('Average face classification time: {0:.2f}s'.format(time_classif / num_detect))

    cap.release()
    cv2.destroyAllWindows()
