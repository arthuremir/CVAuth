import tkinter as tk
from tkinter import font as tkfont
from tkinter import Tk, Label, StringVar, Entry, Button


class MainApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        window_size = (640, 480)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        x_shift = int(self.winfo_screenwidth() / 2 - window_size[0] / 2)
        y_shift = int(self.winfo_screenheight() / 2 - window_size[1] / 2)

        self.geometry("{}x{}+{}+{}".format(window_size[0], window_size[1], x_shift, y_shift))

        self.frames = {}

        # alternate ways to create the frames & append to frames dict: comment out one or the other

        for F in (StartPage, AuthFrame):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # self.frames["StartPage"] = StartPage(parent=container, controller=self)
        # self.frames["AuthFrame"] = AuthFrame(parent=container, controller=self)
        # self.frames["StartPage"].grid(row=0, column=0, sticky="nsew")
        # self.frames["AuthFrame"].grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    # alternate version of show_frame: comment out one or the other

    def show_frame(self, page_name):
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[page_name]
        frame.grid()

    # def show_frame(self, page_name):
    # frame = self.frames[page_name]
    # frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        cam_wrapper = Label(self)

        message = StringVar()

        message_entry = Entry(textvariable=message, font=('Verdana', 30))
        message_entry.place(relx=.5, rely=.4, anchor="c")
        # message_entry.focus()

        button_signin = Button(text="Sign in",
                               command=lambda: self.controller.show_frame("AuthFrame"),
                               background="#282a36",
                               foreground="#ffffff",
                               padx="20",
                               pady="8",
                               font="16")
        button_signin.place(relx=.5, rely=.5, anchor="c")



class AuthFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Enter something below; the two buttons clear what you type.")
        label.pack(side="top", fill="x", pady=10)
        self.wentry = tk.Entry(self)
        self.wentry.pack(pady=10)
        self.text = tk.Text(self)
        self.text.pack(pady=10)
        restart_button = tk.Button(self, text="Restart", command=self.restart)
        restart_button.pack()
        refresh_button = tk.Button(self, text="Refresh", command=self.refresh)
        refresh_button.pack()

    def restart(self):
        self.refresh()
        self.controller.show_frame("StartPage")

    def refresh(self):
        self.wentry.delete(0, "end")
        self.text.delete("1.0", "end")
        # set focus to any widget except a Text widget so focus doesn't get stuck in a Text widget when page hides
        self.wentry.focus_set()


if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
