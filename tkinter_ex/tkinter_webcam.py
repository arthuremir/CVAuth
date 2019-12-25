from PIL import Image, ImageTk

import cv2
from tkinter import *

width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
cam_wrapper = Label(root)
cam_wrapper.pack()


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    img_tk = ImageTk.PhotoImage(image=img)
    cam_wrapper.imgtk = img_tk
    cam_wrapper.configure(image=img_tk)
    cam_wrapper.after(10, show_frame)


show_frame()
root.mainloop()
