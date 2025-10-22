from tkinter import *
from tkinter import ttk
# from tkinter import tk
import tkinter
from tkinter.ttk import Progressbar
import cv2
import time

from PIL import Image, ImageTk


class Splash_screen:
    def __init__(self, main_window):
        self.Header = tkinter.Frame(main_window, bg="#000080")
        self.Header.config(bg = "white")

        self.logo_Header = tkinter.Frame(self.Header, bg="")
        # self.myCanvas = Canvas(self.Header)

        self.logo_img = Image.open(r"/home/reesh/imge_logo.png")
        self.img1 = ImageTk.PhotoImage(self.logo_img.resize((108, 108), Image.ANTIALIAS))
        self.logo_img = tkinter.Label(self.logo_Header, image=self.img1, bg = "Grey")
        self.logo_img.img = self.img1
        self.logo_img.place(height=100, width=200, x=0, y=0)

        self.caption_Label = tkinter.Label(self.logo_Header, text="Discover+Prepare+Achieve", font=('Helvetica', 10, 'bold'),bg = "Grey",  fg="white")
        self.caption_Label.place(height=57, width=200, x=0, y=100)

        self.Title_label = tkinter.Label(self.Header, text="FR Smart Attendance System", font=('Helvetica', 18, 'bold'), bg="DarkSlateGrey", fg="white")
        self.Title_label.place(height=157, width=588, x=0, y=0)
        # self.create_circle(100, 78, 76, self.myCanvas)

        # self.myCanvas.place(height=157, width=200, x=0, y=0)

        #
        self.logo_Header.place(height=157, width=200, x=588, y=0)
        self.Header.place(height=157, width=788, x=0, y=0)

        self.body = tkinter.Frame(main_window)
        self.pil_img = Image.open(r"/home/reesh/Splash_Background.png")
        self.img = ImageTk.PhotoImage(self.pil_img.resize((788, 313), Image.ANTIALIAS))
        self.lbl_img = tkinter.Label(self.body, image=self.img)
        self.lbl_img.img = self.img
        self.lbl_img.place(relx=0.5, rely=0.5, anchor='center')
        self.body.place(height=313, width=788, x=0, y=157)



    def create_circle(self, x, y, r, canvas):  # center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return canvas.create_oval(x0, y0, x1, y1)


if __name__ == '__main__':
    window = tkinter.Tk()
    window.geometry("788x470+250+150")
    window.wm_overrideredirect(True)
    # window.title("Smart Attendance")
    p1 = tkinter.PhotoImage(file=r'/home/reesh/logo.png')
    p1 = p1.subsample(2, 2)
    window.iconphoto(False, p1)
    obj = Splash_screen(window)
    s = ttk.Style()
    s.theme_use('clam')
    s.configure(style="red.Horizontal.TProgressbar", foreground='red', background='DarkSlateGrey')
    progress = Progressbar(window, orient=tkinter.HORIZONTAL,
                           length=100, mode='determinate',
                           style="red.Horizontal.TProgressbar")

    progress.place(height=20, width=788, x= 0, y=450)

    # progress['maximum']=100
    for i in range(101):
        time.sleep(0.02)
        progress['value'] = i
        progress.update()
    if progress['value'] == 100:
        print("close window")
        window.destroy()

    window.mainloop()