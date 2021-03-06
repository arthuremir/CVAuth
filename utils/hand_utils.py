from PIL import Image

import torch
import torchvision
import cv2
from torchvision import transforms

gestures_dict = ('fist', 'five_fingers', 'four_fingers', 'noise', 'ok', 'one_finger', 'three_fingers', 'two_fingers')


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


def prepare_hand_localizer():
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(512, 8)
    model.load_state_dict(torch.load("gestures/gesture_data/resnet18_gest.pth"))
    model.cuda()
    model.eval()
    return model


def visualize_hand(vis, dets):
    for i in range(dets.shape[0]):
        vis = cv2.rectangle(vis, (dets[i][0], dets[i][1]), (dets[i][2], dets[i][3]), [255, 0, 0], 3)
        vis = cv2.putText(vis,
                          "hand",
                          (dets[i][0], dets[i][1]),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          2,
                          (0, 255, 0),
                          3,
                          cv2.LINE_AA)
    return vis


def run_avg(bg, image, aWeight):
    # global bg
    if bg is None:
        bg = image.copy().astype("float")
        return bg

    cv2.accumulateWeighted(image, bg, aWeight)
    return bg


def segment(bg, image, threshold=30):
    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    return thresholded


def segment_deprecated(bg, image, threshold=30):
    # global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        # print(segmented)
        return thresholded, segmented


def rec_gesture(model, image):
    image = Image.fromarray(image)

    data_transform = transforms.Compose([
        transforms.RandomAffine(25,
                                (0.15, 0.15),
                                (0.7, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    image = data_transform(image)

    # image = np.transpose(image, (2, 0, 1)) / 255

    # image = torch.from_numpy(image)
    # image = image.type(torch.FloatTensor)

    output = model(image[None, :, :].cuda())

    out_num = int(torch.argmax(output))

    label = gestures_dict[out_num]

    return out_num, label
