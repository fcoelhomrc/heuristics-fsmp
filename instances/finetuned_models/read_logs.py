import os
from glob import glob
import pickle

tag = "validation"
ext = "pkl"


class Tracker:
    def __init__(self, backbone, dataset):
        self.backbone = backbone
        self.dataset = dataset

        self.step = []
        self.epoch = []
        self.loss = []
        self.acc = []

    def __call__(self, step, epoch, loss, acc=None):
        self.step.append(step)
        self.epoch.append(epoch)
        self.loss.append(loss)
        if acc is not None:
            self.acc.append(acc)

    def plot_loss(self, ax, *args, **kwargs):
        ax.plot(self.step, self.loss, label=f"{self.backbone}_{self.dataset}", *args, **kwargs)
        ax.redraw_in_frame()

    def plot_acc(self, ax, *args, **kwargs):
        ax.plot(self.step, self.acc, label=f"{self.backbone}_{self.dataset}", *args, **kwargs)
        ax.redraw_in_frame()


import matplotlib.pyplot as plt

for fname in glob(os.path.join(os.getcwd(), f"*{tag}*{ext}")):
    with open(fname, "rb") as f:
        basename = os.path.basename(fname)
        name = "_".join(basename.split("_")[2:]).split(".")[0]
        print(name)
        data = pickle.load(f)
        plt.figure()
        data.plot_acc(plt.gca())
        plt.show()