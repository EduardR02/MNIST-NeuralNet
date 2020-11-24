"""https://opensourc.es/blog/tensorflow-mnist/
    if followed this websites tutorial on how to center the images

    Also the gui for drawing is from here:
    https://data-flair.training/blogs/python-deep-learning-project-handwritten-digit-recognition/"""


import math
import numpy as np
import mnist  # Get data set from
import keras.models as m
import keras.layers as l
import win32gui
from PIL import ImageGrab
from keras.utils import to_categorical
import matplotlib.pyplot as plt  # Graph
import cv2
import os.path
from tkinter import *
import tkinter as tk

from scipy import ndimage
from tensorflow import compat

config = compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
compat.v1.Session(config=config)

model_name = "my_model"
how_many = 40
my_image_labels = []
my_images = np.zeros((how_many, 784))


def load_my_stuff():
    global my_images, my_image_labels, how_many
    j = 0
    number = 0
    iteration = 1
    dir_root = "Mnist_Addition/"
    check = False
    while not check:
        if j < how_many:
            img_name = str(number) + "_" + str(iteration)
            filename = dir_root + img_name + ".png"
            if os.path.isfile(filename):
                my_images[j] = resize_my_image(28, filename)
                my_image_labels.append(number)
                iteration += 1
                j += 1
            else:
                if number != 9:
                    number += 1
                    iteration = 1
                else:
                    check = True
        else:
            check = True


def resize_my_image(size, filename=None, img=None):
    if img is None and filename is not None:
        img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(255 - img, (size, size))  # inverts color
    (_, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img = image_center(img)
    img = img.flatten() / 255
    return img


def image_center(img):
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    rows, cols = img.shape
    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0 / cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    cols_padding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rows_padding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rows_padding, cols_padding), 'constant')

    sx, sy = getBestShift(img)
    img = shift(img, sx, sy)

    return img


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty


def shift(img, sx, sy):
    rows, cols = img.shape
    matrix = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(img, matrix, (cols, rows))
    return shifted


def reshape_labels(number):
    a_list = np.zeros(10)
    a_list[number] = 1
    return a_list


def predict_one_digit(img, model):
    img = resize_my_image(28, None, img)
    img = img.reshape(-1, 28, 28, 1)
    prediction = model.predict([img])[0]
    return np.argmax(prediction), max(prediction), img


def train_model():
    train_images = mnist.train_images()  # training data of images
    train_labels = mnist.train_labels()  # training data of the labels
    test_images = mnist.test_images()  # testing data images
    test_labels = mnist.test_labels()  # testing data labels

    train_images = train_images.reshape((-1, 28, 28, 1))
    test_images = test_images.reshape((-1, 28, 28, 1))

    train_images = (train_images / 255)
    test_images = (test_images / 255)

    print(train_images.shape)  # 60,000 rows and 784 cols
    print(test_images.shape)  # 10,000 rows and 784 cols

    model = m.Sequential()
    model.add(l.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(l.MaxPooling2D((2, 2)))
    model.add(l.Conv2D(64, (3, 3), activation='relu'))
    model.add(l.MaxPooling2D((2, 2)))
    model.add(l.Flatten())
    model.add(l.Dense(100, activation='relu', input_dim=784))
    model.add(l.Dense(64, activation="relu"))
    model.add(l.Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    model.fit(
        train_images,
        to_categorical(train_labels),
        epochs=15,
        batch_size=32
    )

    model.evaluate(test_images, to_categorical(test_labels))
    model.save(model_name)


def custom():
    global my_images, my_image_labels, how_many
    model = m.load_model(model_name)
    load_my_stuff()
    my_images = my_images.reshape(-1, 28, 28, 1)
    list1 = np.argmax(model.predict(my_images), axis=-1)
    list2 = my_image_labels
    i = 0
    total_errors = 0

    plt.rcParams.update({'figure.max_open_warning': 50})

    while i < how_many:
        if list1[i] != list2[i]:
            total_errors += 1
            print("Error Nr: ", total_errors, "; ", sep="", end="")
            print("Guessed number: ", list1[i], "; ", sep="", end="")
            print("Correct number: ", list2[i], "; ", sep="", end="")
            print("Index of the occurrence:", i, end=" \n")

            first_image = my_images[i]
            first_image = np.array(first_image, dtype='float')
            pixels = first_image.reshape((28, 28))
            plt.figure(total_errors)
            plt.imshow(pixels, cmap='gray')

            i += 1
        else:
            i += 1

    print("Total errors:", total_errors)
    print("Accuracy:", (1 - total_errors / how_many) * 100, "%")
    plt.show()


class App(tk.Tk):
    model = m.load_model(model_name)

    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        # self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        img = np.asarray(ImageGrab.grab(rect))
        digit, acc, img = predict_one_digit(img, self.model)
        self.label.configure(text=str(digit) + ', ' + str(int(acc * 100)) + '%')
        self.show_img(img, digit)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 12
        self.canvas.create_rectangle(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')

    def show_img(self, img, digit):
        img = np.array(img, dtype='float')
        img = img.reshape((28, 28))
        plt.figure(digit)
        plt.imshow(img, cmap='gray')
        plt.show()


if __name__ == "__main__":
    app = App()
    tk.mainloop()
