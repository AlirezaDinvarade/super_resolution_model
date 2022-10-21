import tifffile
import numpy as np
import os

lr_img_train = os.listdir("./Dataset/train/LR")
hr_img_train = os.listdir("./Dataset/train/HR")

lr_img_val = os.listdir("./Dataset/validation/LR")
hr_img_val= os.listdir("./Dataset/validation/HR")

lr_img_test = os.listdir("./Dataset/test/LR")
hr_img_test= os.listdir("./Dataset/test/HR")

for lr in lr_img_train: 
    if not lr in hr_img_train:
        print(f"Found an intrupt in train (Lr = {lr}) not in high resolution\ntrying to delete\n")
        # os.remove(f"./Dataset/train/LR/{lr}")

for hr in hr_img_train: 
    if not hr in lr_img_train:
        print(f"Found an intrupt in train (HR = {hr}) not in low resolution\ntrying to delete\n")
        # os.remove(f"./Dataset/train/HR/{hr}")
        
for lr in lr_img_val: 
    if not lr in hr_img_val:
        print(f"Found an intrupt in validation (Lr = {lr}) not in high resolution\ntrying to delete\n")
        os.remove(f"./Dataset/validation/LR/{lr}")

for hr in hr_img_val: 
    if not hr in lr_img_val:
        print(f"Found an intrupt in validation (HR = {hr}) not in low resolution\ntrying to delete\n")
        os.remove(f"./Dataset/validation/HR/{hr}")
        

for lr in lr_img_test: 
    if not lr in hr_img_test:
        print(f"Found an intrupt in test (Lr = {lr}) not in high resolution\ntrying to delete\n")
        # os.remove(f"./Dataset/test/LR/{lr}")

for hr in hr_img_test: 
    if not hr in lr_img_test:
        print(f"Found an intrupt in test (HR = {hr}) not in low resolution\ntrying to delete\n")
        # os.remove(f"./Dataset/test/HR/{hr}")

print("finished!")