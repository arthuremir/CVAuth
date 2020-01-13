from tkinter import *

# pip install pillow
from PIL import Image, ImageTk


def pressed(num):
    print(f'pressed {num}!')

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/fist/2020-01-12-13-48-02.866400.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(1))
        img.place(x=0, y=0)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/ok/2020-01-12-13-47-03.916938.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(2))
        img.place(x=0, y=200)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/five_fingers/2020-01-12-13-45-53.772685.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(3))
        img.place(x=400, y=200)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/four_fingers/2020-01-13-00-28-55.789089.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(4))
        img.place(x=200, y=200)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/one_finger/2020-01-12-13-41-26.204770.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(5))
        img.place(x=200, y=0)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/three_fingers/2020-01-12-13-44-01.981331.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(6))
        img.place(x=600, y=0)

        load = Image.open(
            "/home/user/Desktop/webcam_app/gestures/gesture_data/two_fingers/2020-01-12-13-43-22.381874.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render)
        img.image = render
        img.bind("<Button-1>", lambda e: pressed(7))
        img.place(x=400, y=0)


root = Tk()
app = Window(root)
root.wm_title("Tkinter window")
root.geometry("200x120")
root.mainloop()
