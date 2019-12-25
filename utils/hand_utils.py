import torch
import cv2

from hand_detection.models.handboxes import HandBoxes


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


def prepare_hand_localizer(trained_model, cpu, device):
    hand_localizer = HandBoxes(phase='test', size=None, num_classes=2)
    hand_localizer = load_model(hand_localizer, trained_model, cpu)
    hand_localizer = hand_localizer.to(device)
    hand_localizer.eval()
    return hand_localizer


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
