import time


import numpy as np
import cv2
import torch
from PIL import Image


from cvauth.hand_detection.data.config import cfg as hand_cfg
from cvauth.hand_detection.layers.functions.prior_box import PriorBox
from cvauth.hand_detection.utils.box_utils import decode
from cvauth.hand_detection.utils.py_cpu_nms import py_cpu_nms as nms


def detect_hands(frame, localizer, device, args):
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

    inds = np.where(scores > args.hand_confidence_threshold)[0]
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


def detect_faces(frame, detector, conf, learner, targets, tta):
    image = Image.fromarray(frame[..., ::-1])

    #time_detect_init = time.time()
    aligned_faces = detector.align_multi(image, conf.face_limit, conf.min_face_size)
    #time_detect = time.time() - time_detect_init

    if aligned_faces is None:
        return None
    bboxes, faces = aligned_faces[0], aligned_faces[1]
    bboxes = bboxes[:, :-1]
    bboxes = bboxes.astype(int)
    bboxes = bboxes + [-1, -1, 1, 1]

    #time_classif_init = time.time()
    results, score = learner.infer(conf, faces, targets, tta)
    #time_classif = time.time() - time_classif_init

    return [bboxes, results, score]#, [time_detect, time_classif]