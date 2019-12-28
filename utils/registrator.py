import os
import cv2
from PIL import Image
from datetime import datetime
from pathlib import Path

import numpy as np

from cvauth.arcface.mtcnn import MTCNN
from cvauth.arcface.mtcnn_pytorch.src.visualization_utils import show_bboxes

SAVE_FOLDER = Path('cvauth/arcface/data/facebank')
DIRTY_FLAG = SAVE_FOLDER / ".dirty"


class Registrator:

    def __init__(self, username):
        self.save_path = SAVE_FOLDER / username
        os.mkdir(self.save_path)
        self.mtcnn = MTCNN()
        print('MTCNN loaded!')

    def run(self, frame, capture_flag):
        p = Image.fromarray(frame[..., ::-1])
        detect_res = self.mtcnn.align(p)

        if detect_res is None:
            # print("F")
            return frame
        print("YESS")

        bboxes, aligned_face, landmarks = detect_res

        bboxes = bboxes[:, :-1]
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1, -1, 1, 1]

        frame = show_bboxes(Image.fromarray(frame), bboxes, landmarks)

        # if capture_flag:
        warped_face = np.array(aligned_face)[..., ::-1]
        cv2.imwrite(
            str(self.save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
            warped_face)
        print('Image saved!')
        if not os.path.exists(DIRTY_FLAG):
            open(DIRTY_FLAG, 'a').close()
            print('Dirty flag created')
        return frame.get() if isinstance(frame, type(cv2.UMat())) else np.array(frame, dtype=np.uint8)
