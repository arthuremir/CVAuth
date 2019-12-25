from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
# from utils.nms_wrapper import nms FIX!!!
import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer
from utils.py_cpu_nms import py_cpu_nms as nms  # FIX!!!

parser = argparse.ArgumentParser(description='HandBoxes')

parser.add_argument('-m', '--trained_model', default='weights/Final_HandBoxes.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--video', default='data/video/hand.avi', type=str, help='dataset')
parser.add_argument('--image', default=None, type=str, help='dataset')
parser.add_argument('--confidence_threshold', default=0.2, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_rn18_pretrained(weights_path, num_classes):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(weights_path))
    # model.cuda()

    return model


def draw_box_name(bbox, name, frame):
    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
    frame = cv2.putText(frame,
                        name,
                        (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA)
    return frame


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cudnn.benchmark = True

    device = torch.device("cpu" if args.cpu else "cuda")

    localizer = FaceBoxes(phase='test', size=None, num_classes=2)
    localizer = load_model(localizer, args.trained_model, args.cpu)
    localizer = localizer.to(device)
    localizer.eval()
    print('Localizer loaded!')

    '''CLASSIFIER_WPATH = "weights/best_model.pth"
    classifier = load_rn18_pretrained(CLASSIFIER_WPATH, 24)
    classifier = classifier.to(device)
    classifier.eval()

    print('Classifier loaded!')'''

    resize = 2

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    if args.image:
        to_show = cv2.imread(args.image, cv2.IMREAD_COLOR)
        img = np.float32(to_show)

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        out = localizer(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, args.nms_threshold, force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        _t['misc'].toc()

        for i in range(dets.shape[0]):
            cv2.rectangle(to_show, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)

        cv2.imshow('image', to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        # videofile = args.video

        cap = cv2.VideoCapture(0)

        assert cap.isOpened(), 'Cannot capture source'

        while cap.isOpened():

            ret, frame = cap.read()
            if ret:
                to_show = frame
                img = np.float32(to_show)

                if resize != 1:
                    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                im_height, im_width, _ = img.shape
                scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                img -= (104, 117, 123)
                img = img.transpose(2, 0, 1)
                img = torch.from_numpy(img).unsqueeze(0)
                img = img.to(device)
                scale = scale.to(device)

                _t['forward_pass'].tic()
                out = localizer(img)  # forward pass
                _t['forward_pass'].toc()
                _t['misc'].tic()
                priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
                priors = priorbox.forward()
                priors = priors.to(device)
                loc, conf, _ = out
                prior_data = priors.data
                boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                boxes = boxes * scale / resize
                boxes = boxes.cpu().numpy()
                scores = conf.data.cpu().numpy()[:, 1]

                # ignore low scores
                inds = np.where(scores > args.confidence_threshold)[0]
                boxes = boxes[inds]
                scores = scores[inds]

                # keep top-K before NMS
                order = scores.argsort()[::-1][:args.top_k]
                boxes = boxes[order]
                scores = scores[order]

                # do NMS
                dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                # keep = py_cpu_nms(dets, args.nms_threshold)
                keep = nms(dets, args.nms_threshold)  # , force_cpu=args.cpu)
                dets = dets[keep, :]

                # keep top-K faster NMS
                dets = dets[:args.keep_top_k, :]
                _t['misc'].toc()

                #print(dets, dets.shape)


                for i in range(dets.shape[0]):
                    cv2.rectangle(to_show, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [0, 0, 255], 3)

                cv2.imshow('image', to_show)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break


            else:
                break