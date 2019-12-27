from __future__ import print_function

import multiprocessing as mp
import os
import tkinter as tk
from datetime import datetime
from tkinter import Tk, Label, StringVar, Entry, Button

import PIL
import numpy as np
from PIL import Image, ImageTk
from cvauth.detectron2.detectron2.utils.logger import setup_logger

from cvauth.arcface.config import get_config
from cvauth.arcface.learner import FaceLearner
from cvauth.arcface.mtcnn import MTCNN
from cvauth.arcface.mtcnn_pytorch.src.visualization_utils import show_bboxes
from cvauth.arcface.utils import load_facebank, prepare_facebank, get_facebank_names
from utils.detectors import detect_hands, detect_faces
from utils.face_utils import *
from utils.hand_utils import *
from utils.predictor import VisualizationDemo
from utils.setup_detectron import setup_detectron_cfg
from utils.arg_parser import get_args

WINDOW_SIZE = (640, 600)
TITLE_INIT = "Verification System"


def show_frame():
    _, frame = cap.read()

    if if_pose.get():
        _, vis = detectron_flow.run_on_image(frame)
        vis = vis.get_image()
    else:
        vis = frame[:, :, ::-1]

    #frame = frame[:, :, ::-1]

    if if_hand.get():
        hand_bboxes = detect_hands(frame, hand_localizer, conf.device, args)
        if hand_bboxes is not None:
            vis = visualize_hand(vis, hand_bboxes)

    if if_face.get():
        faces = detect_faces(frame, detector, conf, learner, targets, args.tta)
        if faces is not None:
            vis = visualize_face(vis, faces, names, args.score)

    vis = vis.get() if isinstance(vis, type(cv2.UMat())) else np.array(vis, dtype=np.uint8)
    img = PIL.Image.fromarray(vis)
    img_tk = ImageTk.PhotoImage(image=img)
    cam_wrapper.img_tk = img_tk
    cam_wrapper.configure(image=img_tk)
    cam_wrapper.after(1, show_frame)

    # TODO add return and releasing cap


def process_name():
    username = message.get()
    if username in facebank_names:
        root.title("Loading...")

        for elem in root.place_slaves():
            elem.destroy()

        root.unbind('<Return>')

        root.title("Loading")

        global cap, detectron_flow, targets, \
            names, hand_localizer, detector, learner

        cap = cv2.VideoCapture(0)

        if if_pose.get():
            cfg = setup_detectron_cfg(args)
            detectron_flow = VisualizationDemo(cfg)
            print('Pose estimator is loaded!')

        if if_face.get():
            detector = MTCNN()
            learner = FaceLearner(conf, True)
            learner.threshold = args.threshold
            if conf.device.type == 'cpu':
                learner.load_state(conf, 'cpu_final.pth', True, True)
            else:
                learner.load_state(conf, 'final.pth', True, True)
            learner.model.eval()
            print('FaceLearner is loaded!')

            if args.update:
                targets, names = prepare_facebank(conf, learner.model, detector, tta=args.tta)
                print('Facebank is updated!')
            else:
                targets, names = load_facebank(conf)
                print('Facebank is loaded!')

        if if_hand.get():
            hand_localizer = prepare_hand_localizer(args.trained_model, args.cpu, conf.device)
            print('Hand localizer is loaded!')

        root.title("Authorization process")

        cam_wrapper.pack()

        show_frame()

    else:
        root.title("Wrong name!")


def register_user():
    username = message.get()
    if username in facebank_names:
        root.title("This name already exists. Try another one!")
    else:
        user_path = facebank_path / username

        os.mkdir(user_path)

        capp = cv2.VideoCapture(0)

        detector = MTCNN()

        while capp.isOpened():

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            is_success, frame = capp.read()

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

        capp.release()
        cv2.destroyAllWindows()
        #TODO use tkinter window


def init_window(title, size):
    root = Tk()
    root.title(title)

    x_shift = int(root.winfo_screenwidth() / 2 - WINDOW_SIZE[0] / 2)
    y_shift = int(root.winfo_screenheight() / 2 - WINDOW_SIZE[1] / 2)

    root.geometry("{}x{}+{}+{}".format(WINDOW_SIZE[0], WINDOW_SIZE[1], x_shift, y_shift))
    root.configure(background='#282a36')

    return root


def init_widgets():
    pass
#TODO create window initializer


if __name__ == "__main__":
    root = init_window(title=TITLE_INIT, size=WINDOW_SIZE)

    cam_wrapper = Label(root)

    message = StringVar()

    message_entry = Entry(textvariable=message, font=('Verdana', 30))
    message_entry.place(relx=.5, rely=.4, anchor="c")
    message_entry.focus()

    if_face = tk.IntVar()
    face_chk = tk.Checkbutton(root,
                              text='Face verification',
                              variable=if_face,
                              bg='#E95420',
                              # fg='#ffffff',
                              activebackground='#E95420',
                              # activeforeground='#15151b',
                              font=('Helvetica', 15))
    face_chk.place(relx=.5, rely=.15, anchor="c")

    if_pose = tk.IntVar()
    pose_chk = tk.Checkbutton(root,
                              text='Pose estimation',
                              variable=if_pose,
                              bg='#E95420',
                              # fg='#ffffff',
                              activebackground='#E95420',
                              # activeforeground='#ffffff',
                              font=('Helvetica', 15),
                              )
    pose_chk.place(relx=.5, rely=.2, anchor="c")

    if_hand = tk.IntVar()
    hand_chk = tk.Checkbutton(root,
                              text='Hand detection',
                              variable=if_hand,
                              bg='#E95420',
                              # fg='#ffffff',
                              activebackground='#E95420',
                              # activeforeground='#ffffff',
                              font=('Helvetica', 15))
    hand_chk.place(relx=.5, rely=.25, anchor="c")

    button_signin = Button(text="Sign in",
                           command=process_name,
                           background="#282a36",
                           foreground="#ffffff",
                           padx="20",
                           pady="8",
                           font="16")
    button_signin.place(relx=.5, rely=.5, anchor="c")

    button_signup = Button(text="Sign up",
                           command=register_user,
                           background="#282a36",
                           foreground="#ffffff",
                           padx="20",
                           pady="8",
                           font="16")
    button_signup.place(relx=.5, rely=.7, anchor="c")


    root.bind('<Escape>', lambda e: root.quit())
    root.bind('<Return>', lambda e: process_name())

    conf = get_config(False)

    facebank_path = conf.facebank_path
    facebank_names = get_facebank_names(facebank_path)

    mp.set_start_method("spawn", force=True)
    logger = setup_logger()

    args = get_args()

    root.mainloop()
