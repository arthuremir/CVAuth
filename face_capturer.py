import cv2
import argparse
from PIL import Image
from datetime import datetime
from pathlib import Path

import numpy as np

from arcface.mtcnn import MTCNN

parser = argparse.ArgumentParser(description='take a picture')
parser.add_argument('--name', '-n', default='unknown', type=str, help='input the name of the recording person')
args = parser.parse_args()

data_path = Path('data')
save_path = data_path / 'facebank' / args.name
if not save_path.exists():
    save_path.mkdir()

cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
mtcnn = MTCNN()

while cap.isOpened():

    isSuccess, frame = cap.read()
    if isSuccess:
        frame_text = cv2.putText(frame,
                                 'Press t to take a picture,q to quit.....',
                                 (10, 100),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 2,
                                 (0, 255, 0),
                                 3,
                                 cv2.LINE_AA)
        cv2.imshow("My Capture", frame_text)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('t'):
        p = Image.fromarray(frame[..., ::-1])
        try:
            warped_face = np.array(mtcnn.align(p))[..., ::-1]
            cv2.imwrite(str(save_path / '{}.jpg'.format(str(datetime.now())[:-7].replace(":", "-").replace(" ", "-"))),
                        warped_face)
        except:
            print('no face captured')

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
