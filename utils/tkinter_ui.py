#import gc
import tkinter as tk
from tkinter import font
from tkinter import Label, StringVar, Entry, Button
from PIL import Image, ImageTk

import cv2

from cvauth.arcface.config import get_config
from cvauth.arcface.utils import get_facebank_names

from utils.visualizer import Visualizer
from utils.registrator import Registrator


# TODO fix titles


class MainApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

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
        self.frames["StartPage"] = StartPage(parent=self.container, controller=self)
        self.frames["StartPage"].grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):

        if page_name in self.frames:
            self.frames[page_name].tkraise()
            if page_name == "AuthPage" or page_name == "RegisterPage":
                self.frames[page_name].init_capture()
        else:
            # TODO refactor this
            frame = globals()[page_name](parent=self.container, controller=self)

            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
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
        face_chk.place(relx=.38, rely=.35, anchor="c")

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
        pose_chk.place(relx=.38, rely=.45, anchor="c")

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
        hand_chk.place(relx=.38, rely=.55, anchor="c")

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
        global username
        username = self.message.get().lower()
        if username in self.facebank_names:
            self.controller.show_frame("AuthPage")
        else:
            self.controller.title("Wrong name")

    def register_user(self):
        global username
        username = self.message.get().lower()
        if username in self.facebank_names:
            self.controller.title("This user already exists!")
        elif username.strip():
            self.controller.show_frame("RegisterPage")
        else:
            self.controller.title("Invalid name")


class AuthPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        self.cam_wrapper = Label(self)
        self.cam_wrapper.pack()

        button_int = Button(self,
                            text="X",
                            command=lambda: self.raise_stop_flag(),
                            bd=0,
                            highlightthickness=0,
                            background="#C60000",
                            foreground="#ffffff",
                            padx="20",
                            pady="10",
                            font="16")
        button_int.place(relx=0.04, rely=0.961, anchor="c")

        self.visualizer = Visualizer(if_face=if_face,
                                     if_pose=if_pose,
                                     if_hand=if_hand)

        self.init_capture()

    def init_capture(self):
        self.cap = cv2.VideoCapture(0)
        self.stop_flag = 0

        self.update_cam()

    def update_cam(self):
        _, frame = self.cap.read()
        vis = self.visualizer.run(frame)

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

        # gc.collect()

        self.controller.show_frame("StartPage")


class RegisterPage(tk.Frame):

    def __init__(self, parent, controller):
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
        button_cap.place(relx=0.2, rely=0.961, anchor="c")

        self.registrator = Registrator(username)

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
