from importlib.resources import path
import cv2
import tifffile
from skimage.exposure import match_histograms
import os
import numpy as np
import random
import tqdm
import shutil


def comment(message):
    print(f"{message}\n")


def resize_img(lr_img_path, hr_img_path):
    """Resize images into right shape
    for LR images : (256, 255) and 
    for HR images : (1024, 1024)

    Args:
        lr_img_path (str): low resolution image path
        hr_img_path (str): high resolution image path

    Returns:
        new_lr_img: resized low resolution image
        new_hr_img: resized high resolution image
    """
    lr_img = tifffile.imread(lr_img_path)
    hr_img = tifffile.imread(hr_img_path)
    
    new_lr_img = cv2.resize(src=lr_img, dsize=(256, 256), interpolation=cv2.INTER_NEAREST )
    new_hr_img = cv2.resize(src=hr_img, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
    
    return new_lr_img, new_hr_img


# train validation test split
train_size = 80 # % (percentage)
validation_szie = 10 # % (percentage)
test_size = 10 # % (percentage)

# Check if directory exits
if not os.path.exists('./Dataset'):
    os.mkdir('./Dataset')
    os.mkdir('./Dataset/all')
    os.mkdir('./Dataset/all/LR')
    os.mkdir('./Dataset/all/HR')
    os.mkdir('./Dataset/train')
    os.mkdir('./Dataset/train/HR')
    os.mkdir('./Dataset/train/LR')
    os.mkdir('./Dataset/validation')
    os.mkdir('./Dataset/validation/HR')
    os.mkdir('./Dataset/validation/LR')
    os.mkdir('./Dataset/test')
    os.mkdir('./Dataset/test/HR')
    os.mkdir('./Dataset/test/LR')

if not os.path.exists('./Data_Download'):
    raise Exception("Please provide downloaded data path!")


pols = os.listdir('./Data_Download')
counter = 0
# Gathering all pair images pathes in all polygons
for pol in pols:
    
    path_lr_pol = []
    path_hr_pol = []
    
    for i in range(len(os.listdir(f'./Data_Download/{pol}/LR'))):
        path_lr_pol.append(f'./Data_Download/{pol}/LR/LR_{i}.tif')

    for i in range(len(os.listdir(f'./Data_Download/{pol}/HR'))):
        path_hr_pol.append(f'./Data_Download/{pol}/HR/HR_{i}.tif')

    if len(path_lr_pol) != len(path_hr_pol):
        print("\nNot same images in hr and lr in {pol}")
        raise Exception("Number of LR images doesn't match with number of HR images!")
    else:
         
        for j in tqdm.trange(len(path_lr_pol),
        desc=pol,
        bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
            try:
                lr_img, hr_img = resize_img(path_lr_pol[j], path_hr_pol[j])
                tifffile.imwrite(f'./Dataset/all/LR/{counter}.tif', lr_img)
                tifffile.imwrite(f'./Dataset/all/HR/{counter}.tif', hr_img)
                counter = counter + 1
            except Exception :
                print(f"{path_lr_pol[j]} and {path_hr_pol[j]} are not valid!!!")
                continue
            

comment("\nImages resizing  is finished!")

# Check the number of pair images if they are same
if len(os.listdir('./Dataset/all/LR/')) != len(os.listdir('./Dataset/all/HR')):
    raise Exception("Number of total LR images doesn't match with number of total HR images!")


comment("Image copying is started:")
# writing transformed images in new directory

# Creating random index
random.seed(42)
random_index = np.arange(0, len(os.listdir('./Dataset/all/LR')), dtype=int)
np.random.shuffle(random_index)

path_lr = os.listdir('./Dataset/all/LR')
path_hr = os.listdir('./Dataset/all/HR')

for num, index in enumerate(tqdm.tqdm(list(random_index))):

    if (num+1)/len(path_lr) < 0.8 :
        # lr_img, hr_img = resize_img(path_lr[index], path_hr[index])
        # hist_match_hr_img = hist_match_img(lr_img, hr_img)
        shutil.move(f'./Dataset/all/LR/{index}.tif', f'./Dataset/train/LR/{index}.tif')
        shutil.move(f'./Dataset/all/HR/{index}.tif', f'./Dataset/train/HR/{index}.tif')

    elif (num+1)/len(path_lr) < 0.9 :
        # lr_img, hr_img = resize_img(path_lr[index], path_hr[index])
        # hist_match_hr_img = hist_match_img(lr_img, hr_img)
        shutil.move(f'./Dataset/all/LR/{index}.tif', f'./Dataset/validation/LR/{index}.tif')
        shutil.move(f'./Dataset/all/HR/{index}.tif', f'./Dataset/validation/HR/{index}.tif')


    else:
        # lr_img, hr_img = resize_img(path_lr[index], path_hr[index])
        # hist_match_hr_img = hist_match_img(lr_img, hr_img)
        shutil.move(f'./Dataset/all/LR/{index}.tif', f'./Dataset/test/LR/{index}.tif')
        shutil.move(f'./Dataset/all/HR/{index}.tif', f'./Dataset/test/HR/{index}.tif')


def hist_match_img(lr_img_path, hr_img_path):
    """Match histogram of high resolution image based of low resolution image

    Args:
        lr_img (ndarray): low resolution image
        hr_img (ndarray): high resolution image

    Returns:
        new_hr_img: transformed high resolution image
    """
    lr_img = tifffile.imread(lr_img_path)
    hr_img = tifffile.imread(hr_img_path)
    new_hr_img = match_histograms(hr_img, lr_img, channel_axis=2)
    return new_hr_img, lr_img


dirs = ['train',
        'validation',
        'test'
        ]

# Gathering all pair images pathes in all polygons
for dir in dirs:
    counter = 0
    imgs_lr = os.listdir(f"./Dataset/{dir}/LR")
    imgs_hr = os.listdir(f"./Dataset/{dir}/HR")

    for j in tqdm.trange(len(imgs_hr),
                         desc=dir):
        try:
            hr_img, lr_img = hist_match_img(f"./Dataset/{dir}/LR/{imgs_lr[j]}",
                                            f"./Dataset/{dir}/HR/{imgs_hr[j]}")

            tifffile.imwrite(f'./Dataset2/{dir}/LR/{counter}.tif', lr_img)
            tifffile.imwrite(f'./Dataset2/{dir}/HR/{counter}.tif', hr_img)
            counter = counter + 1
        except Exception:
            print(f"{imgs_lr[j]} and {imgs_hr[j]} are not valid!!!")
            continue

comment("\nHistogram matching is finished!")