import argparse
from PIL import Image
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

num_frames = 0
bg = None


def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return bg

    cv2.accumulateWeighted(image, bg, aWeight)
    return bg


def segment(image, threshold=30):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)

    hand_bin = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    return hand_bin


def get_image(frame):
    global bg, num_frames
    hand_box = [100, 440, 300, 640]

    hand_bin = frame[hand_box[0]:hand_box[2], hand_box[1]:hand_box[3]]

    gray = cv2.cvtColor(hand_bin, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 30:
        bg = run_avg(gray, 0.5)
        if num_frames == 1:
            print("[STATUS] please wait! calibrating...")
        elif num_frames == 29:
            print("[STATUS] calibrated successfully...")
    else:
        hand_bin = segment(gray)

        hand_bin = np.stack((hand_bin,) * 3, axis=-1)

        frame[hand_box[0]:hand_box[2], hand_box[1]:hand_box[3]] = hand_bin

    vis = cv2.rectangle(frame, (hand_box[3], hand_box[0]), (hand_box[1], hand_box[2]),
                        (0, 255, 0), 2)

    num_frames += 1

    return vis, hand_bin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='take a picture')
    parser.add_argument('--gesture', '-n', default='unknown', type=str, help='input the gesture name')
    args = parser.parse_args()

    data_path = Path('gesture_data')
    save_path = data_path / args.gesture
    assert save_path.exists()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame, img_bin = get_image(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):

            cv2.imwrite(
                str(save_path / '{}.jpg'.format(str(datetime.now()).replace(":", "-").replace(" ", "-"))),
                img_bin)

        elif key == ord('q'):
            break

        cv2.imshow("Capturing gesture", frame)

    cap.release()
    cv2.destroyAllWindows()
