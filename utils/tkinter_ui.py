import time
import tkinter as tk
from tkinter import font
from tkinter import Label, StringVar, Entry, Button, IntVar
from PIL import Image, ImageTk

import cv2

from utils.arg_parser import get_args
from cvauth.arcface.config import get_config
from cvauth.arcface.utils import get_facebank_names

from utils.visualizer import Visualizer
from utils.registrator import Registrator
from utils.fps_plt import add_fps_plot


# TODO fix titles
# TODO disable input field


class MainApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.args = get_args()

        self.bind('<Escape>', lambda e: self.quit())

        window_size = (640, 520)

        self.title_font = font.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        x_shift = int(self.winfo_screenwidth() / 2 - window_size[0] / 2)
        y_shift = int(self.winfo_screenheight() / 2 - window_size[1] / 2)

        self.geometry("{}x{}+{}+{}".format(window_size[0], window_size[1], x_shift, y_shift))

        self.frames = dict()
        self.frames["StartPage"] = StartPage(args=self.args, parent=self.container, controller=self)
        self.frames["StartPage"].grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):

        if page_name in self.frames:
            self.frames[page_name].tkraise()
            if page_name == "AuthPage" or page_name == "RegisterPage":
                self.frames[page_name].init_capture()
        else:
            # TODO refactor this
            frame = globals()[page_name](args=self.args, parent=self.container, controller=self)

            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, args, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.configure(background='#2D303D')
        self.controller.title("CVAuth")

        conf = get_config(False)

        facebank_path = conf.facebank_path
        self.facebank_names = get_facebank_names(facebank_path)

        self.message = StringVar()

        message_entry = Entry(self,
                              textvariable=self.message,
                              bd=0,
                              highlightthickness=0,
                              background="#21232d",
                              fg="white",
                              insertbackground='white',
                              font=('Verdana', 30))
        message_entry.place(relx=.5, rely=.2, anchor="c")
        message_entry.focus()

        global if_face, if_pose, if_hand

        if_face = tk.IntVar()
        face_chk = tk.Checkbutton(self,
                                  text='Face verification',
                                  variable=if_face,
                                  pady=10,
                                  bd=0,
                                  highlightthickness=0,
                                  bg='#E35420',
                                  activebackground='#E95420',
                                  font=('Helvetica', 15))
        # face_chk.place(relx=.38, rely=.35, anchor="c")

        if_pose = tk.IntVar()
        pose_chk = tk.Checkbutton(self,
                                  text='Pose estimation  ',
                                  variable=if_pose,
                                  pady=10,
                                  bd=0,
                                  highlightthickness=0,
                                  bg='#E35420',
                                  activebackground='#E95420',
                                  font=('Helvetica', 15),
                                  )
        pose_chk.place(relx=.38, rely=.55, anchor="c")  # change rely to 0.45

        if_hand = tk.IntVar()
        hand_chk = tk.Checkbutton(self,
                                  text='Hand detection   ',
                                  variable=if_hand,
                                  pady=10,
                                  bd=0,
                                  highlightthickness=0,
                                  bg='#E35420',
                                  activebackground='#E95420',
                                  font=('Helvetica', 15))
        # hand_chk.place(relx=.38, rely=.55, anchor="c")

        button_signin = Button(self,
                               text="Sign in",
                               command=self.process_name,
                               bd=0,
                               highlightthickness=0,
                               background="#21232d",
                               foreground="#ffffff",
                               activebackground="#E95420",
                               padx="20",
                               pady="10",
                               font="16")
        button_signin.place(relx=.71, rely=.35, anchor="c")

        button_signup = Button(self,
                               text="Sign up",
                               command=self.register_user,
                               # command=lambda: self.controller.show_frame("RegisterPage"),
                               bd=0,
                               highlightthickness=0,
                               background="#21232d",
                               foreground="#ffffff",
                               activebackground="#E95420",
                               padx="20",
                               pady="10",
                               font="16")
        button_signup.place(relx=.71, rely=.55, anchor="c")

    def process_name(self):
        global username, true_gesture_id
        username = self.message.get().lower()

        names_list = [name[:-2] for name in self.facebank_names]

        if username in names_list:
            name_idx = names_list.index(username)
            user_info = self.facebank_names[name_idx]
            true_gesture_id = int(user_info[-1])
            username = user_info[:-2]
            # print(username, true_gesture_id)
            self.controller.show_frame("AuthPage")

        else:
            self.controller.title("Wrong name")

    def register_user(self):
        global username
        username = self.message.get().lower()
        if username in self.facebank_names:
            self.controller.title("This user already exists!")
        elif '$' not in username and username.strip():
            self.controller.show_frame("GestureRegPage")
        else:
            self.controller.title("Invalid name")


class AuthPage(tk.Frame):

    def __init__(self, args, parent, controller):
        tk.Frame.__init__(self, parent)
        self.args = args

        self.controller = controller

        self.cam_wrapper = Label(self)
        self.cam_wrapper.pack()
        self.fps_arr = list(range(1, 6))
        self.frame_num = 1
        self.time_last = time.time()

        button_int = Button(self,
                            text="X",
                            command=self.raise_stop_flag,
                            bd=0,
                            highlightthickness=0,
                            background="#C60000",
                            foreground="#ffffff",
                            padx="20",
                            pady="10",
                            font="16")
        button_int.place(relx=0.04, rely=0.961, anchor="c")

        self.fps = IntVar()
        self.fps_label = Label(self, textvariable=self.fps, font=('Verdana', 15))
        self.fps_label.place(relx=0.95, rely=0.961, anchor="c")

        '''self.visualizer = Visualizer(self.args,
                                     if_face=1,
                                     if_pose=1,
                                     if_hand=0)'''

        # ------------
        if_face.set(1)
        if_hand.set(0)
        # ------------

        self.visualizer = Visualizer(self.args,
                                     if_face=if_face,
                                     if_pose=if_pose,
                                     if_hand=if_hand)

        self.init_capture()

    def init_capture(self):
        self.cap = cv2.VideoCapture(0)
        self.stop_flag = 0

        self.pos_face_cnt = 0
        self.neg_face_cnt = 0

        self.face_res = 0

        self.pos_gest_cnt = 0
        self.neg_gest_cnt = 0

        self.gest_res = 0

        self.true_gest_id = true_gesture_id

        self.sleep_time = 0
        self.seconds_left = 5
        self.calibration_status = 0

        self.update_cam()

    def update_cam(self):

        # ----- fps tracking--------------------------
        # ----- every 0.2s output fps of last 5 frames
        if time.time() - self.time_last > 0.2:
            delta = self.fps_arr[self.frame_num - 1] - self.fps_arr[self.frame_num]
            self.fps.set(round(5 / delta))
            self.time_last = time.time()

        self.fps_arr[self.frame_num % 5] = time.time()
        self.frame_num = (self.frame_num + 1) % 5
        # --------------------------------------------

        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)

        vis, face_status, gesture_id, calibration_status = self.visualizer.run(frame, username)

        if calibration_status:
            self.calibration_status = 1
            self.controller.title("You have {} seconds to show the gesture to the area!".format(self.seconds_left))
            self.sleep_time = time.time()
            self.time_from_last = time.time()

        if self.calibration_status and self.seconds_left > 0:
            if time.time() - self.time_from_last > 1:
                self.seconds_left -= 1
                self.time_from_last = time.time()
                self.controller.title("You have {} seconds to show the gesture to the area!".format(self.seconds_left))

        # ------------------------ decision making
        if not self.face_res:

            if face_status == 1:
                self.pos_face_cnt += 1
            elif face_status == -1:
                self.neg_face_cnt += 1

            if self.neg_face_cnt == 3:
                self.raise_stop_flag()
                self.controller.title("Wrong person!")
            elif self.pos_face_cnt == 3:
                self.face_res = 1
                self.visualizer.if_face = 0
                self.visualizer.if_pose = 0
                self.visualizer.if_hand = 1
                self.controller.title("Let's start gesture verification step!")

        elif time.time() - self.sleep_time > 5 and gesture_id != -1:
            if gesture_id == self.true_gest_id:
                self.pos_gest_cnt += 1
            else:
                if gesture_id != 3:
                    self.neg_gest_cnt += 1

            if self.neg_gest_cnt == 5:
                self.raise_stop_flag()
                self.controller.title("Wrong gesture!")
            elif self.pos_gest_cnt == 5:
                self.gest_res = 1
                self.raise_stop_flag()
                self.controller.title("Authorized successfully!")
        # --------------------------

        img = Image.fromarray(vis)
        img_tk = ImageTk.PhotoImage(image=img)
        self.cam_wrapper.imgtk = img_tk
        self.cam_wrapper.configure(image=img_tk)
        if self.stop_flag:
            self.disable_frame()
            return
        self.cam_wrapper.after(1, self.update_cam)

    def raise_stop_flag(self):
        self.stop_flag = 1

    def disable_frame(self):
        self.cap.release()

        self.controller.show_frame("StartPage")


class GestureRegPage(tk.Frame):

    def __init__(self, args, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.controller.title("Please, choose a gesture")

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/fist/2020-01-12-13-48-02.866400.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(0))
        img.place(x=40, y=10)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/three_fingers/2020-01-12-13-44-01.981331.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(6))
        img.place(x=40, y=180)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/five_fingers/2020-01-12-13-45-53.772685.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(1))
        img.place(x=440, y=180)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/four_fingers/2020-01-13-00-28-55.789089.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(2))
        img.place(x=240, y=180)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/one_finger/2020-01-12-13-41-26.204770.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(5))
        img.place(x=240, y=10)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/ok/2020-01-12-13-47-03.916938.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(4))
        img.place(x=240, y=350)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/two_fingers/2020-01-12-13-43-22.381874.jpg")
        render = ImageTk.PhotoImage(load.resize((160, 160)))
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: self.choose_gesture(7))
        img.place(x=440, y=10)

    def choose_gesture(self, gesture_id):
        global true_gesture_id

        gestures_dict = (
            'Fist', 'Five fingers', 'Four fingers', 'noise', 'Ok', 'One finger', 'Three fingers', 'Two fingers')

        true_gesture_id = gesture_id
        # self.controller.args['true_gesture_id'] = gesture_id
        self.controller.title(f"You've chosen \'{gestures_dict[gesture_id]}\' gesture")
        self.controller.show_frame("RegisterPage")
    #


class RegisterPage(tk.Frame):

    def __init__(self, args, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.cam_wrapper = Label(self)
        self.cam_wrapper.pack()

        button_int = Button(self,
                            text="X",
                            command=self.raise_stop_flag,
                            bd=0,
                            highlightthickness=0,
                            background="#C60000",
                            foreground="#ffffff",
                            padx="20",
                            pady="10",
                            font="16")
        button_int.place(relx=0.04, rely=0.961, anchor="c")

        button_cap = Button(self,
                            text="Take a photo",
                            command=self.raise_capture_flag,
                            bd=0,
                            highlightthickness=0,
                            background="green",
                            foreground="#ffffff",
                            padx="20",
                            pady="10",
                            font="16")
        # button_cap.place(relx=0.2, rely=0.961, anchor="c")

        self.registrator = Registrator(username, true_gesture_id)

        self.init_capture()

    def init_capture(self):
        self.cap = cv2.VideoCapture(0)
        self.stop_flag = 0
        self.capture_flag = 0

        self.update_cam()

    def update_cam(self):
        _, frame = self.cap.read()

        vis = self.registrator.run(frame, self.capture_flag)
        self.capture_flag = 0

        img = Image.fromarray(vis[..., ::-1])
        img_tk = ImageTk.PhotoImage(image=img)
        self.cam_wrapper.imgtk = img_tk
        self.cam_wrapper.configure(image=img_tk)
        if self.stop_flag:
            self.disable_frame()
            return
        self.cam_wrapper.after(1, self.update_cam)

    def raise_stop_flag(self):
        self.stop_flag = 1

    def raise_capture_flag(self):
        self.capture_flag = 1

    def disable_frame(self):
        self.cap.release()

        self.controller.show_frame("StartPage")
