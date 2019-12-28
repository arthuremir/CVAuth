import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Face verification app')
    # parser.add_argument("-s", "--save", help="whether save", action="store_true")

    # parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54,  # 1.54
                        type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", action="store_true")

    parser.add_argument(
        "--config-file",
        default="cvauth/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml",
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

    parser.add_argument('-m', '--trained_model', default='cvauth/hand_detection/weights/Final_HandBoxes.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')

    parser.add_argument('--hand_confidence_threshold', default=0.5, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=2, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.2, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')

    #parser.add_argument('--fps', default=True, type=int, help='keep_top_k')

    return parser.parse_args()
