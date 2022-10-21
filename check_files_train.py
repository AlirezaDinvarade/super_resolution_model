import matplotlib.pyplot as plt
import tifffile
import numpy as np
from matplotlib.widgets import Button


import os

class Index:
    ind = 0

    def next(self, event):
        self.ind += 1
        if self.ind > len(lr_img)-1:
            self.ind = len(lr_img)-1

        i = self.ind
        xdata1 = tifffile.imread(f"./Dataset/train/LR/{lr_img[i]}").astype(np.uint8)
        xdata2 = tifffile.imread(f"./Dataset/train/HR/{hr_img[i]}").astype(np.uint8)
        img_plot1.set_data(xdata1)
        img_plot2.set_data(xdata2)
        plt.show()
        print(f'IMG : {lr_img[i]} INDEX:{i}\n')


    def previous(self, event):
        self.ind -= 1
        if self.ind < 0:
            self.ind = 0
        i = self.ind
        xdata1 = tifffile.imread(f"./Dataset/train/LR/{lr_img[i]}").astype(np.uint8)
        xdata2 = tifffile.imread(f"./Dataset/train/HR/{hr_img[i]}").astype(np.uint8)
        img_plot1.set_data(xdata1)
        img_plot2.set_data(xdata2)
        plt.show()
        print(f'IMG : {lr_img[i]} INDEX:{i}\n')
        


    def Remove(self, event):
        i = self.ind
        os.remove(f"./Dataset/train/LR/{lr_img[i]}")
        os.remove(f"./Dataset/train/HR/{hr_img[i]}")
        print(f"IMG : {lr_img[i]} and {hr_img[i]} INDEX:{i} deleted\n")
        del lr_img[i]
        del hr_img[i]


lr_img = os.listdir("./Dataset/train/LR")
hr_img = os.listdir("./Dataset/train/HR")


fig1 = plt.figure(figsize=(5,5))
img1 = tifffile.imread(f"./Dataset/train/LR/{lr_img[0]}").astype(np.uint8)
img2 = tifffile.imread(f"./Dataset/train/HR/{hr_img[0]}").astype(np.uint8)
img_plot1 = plt.imshow(img1)
print(f'IMG : {lr_img[0]} INDEX:0\n')
plt.title("sentinel")
# plt.axis('off')


callback = Index()
axprev = plt.axes([0.5, 0.01, 0.15, 0.05])
axnext = plt.axes([0.65, 0.01, 0.15, 0.05])
bndel = plt.axes([0.8, 0.01, 0.15, 0.05])

bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)

bndel= Button(bndel, 'Del')
bndel.on_clicked(callback.Remove)

bprev = Button(axprev, 'Prev')
bprev.on_clicked(callback.previous)


fig2 = plt.figure(figsize=(5,5))
img_plot2 = plt.imshow(img2)
plt.title("High resolution")
# plt.axis('off')

plt.show()