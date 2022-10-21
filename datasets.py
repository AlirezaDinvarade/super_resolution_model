import torch
from torch.utils.data import Dataset
import os
from tifffile import imread
from utils import convert_image
import tqdm

class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, path:str=None, type:str=None):
        """
        :param path: # folder path with data files
        :param type: which part of dataset? (train/test/validation)
        """
        
        if path is None:
            raise Exception("Path can't be empty!!!")

        if type is None:
            raise Exception("Type can't be empty!!!")
        elif not type.lower() in ["train", "validation", "test"]:
            raise Exception("Wrong dataset name!!!\nNot included (train/test/validation)")
        
        self.path = path
        self.type = type.lower()
        #==========================================
        if self.type == "train":
            print("\nCreating train dataset... this may take some time.\n")
            self.train_images_path = []

            if len(os.listdir(f"{self.path}/HR")) != len(os.listdir(f"{self.path}/LR")):
                raise Exception("Not same number of (HR/LR) pair images in train images path!!!")

            img_path_lr = f"{self.path}/LR"
            img_path_hr = f"{self.path}/HR"
            lr_img_list = os.listdir(img_path_lr)
            hr_img_list = os.listdir(img_path_hr)

            for i in tqdm.trange(len(lr_img_list)):
                img_lr = imread(f"{self.path}/LR/{lr_img_list[i]}")
                img_hr = imread(f"{self.path}/HR/{hr_img_list[i]}")

                if not (img_lr.shape[0] == 256 or img_lr.shape[1] == 256):
                    print("Doesn't support min size of low resolution dataset!!!\nImage is not saved!")
                    continue
            
                elif not(img_hr.shape[0] == 1024 or img_hr.shape[1] == 1024):
                    print("Doesn't support min size of high resolution dataset!!!\nImage is not saved!")
                    continue
                else:
                    self.train_images_path.append([f"{self.path}/LR/{lr_img_list[i]}", f"{self.path}/HR/{hr_img_list[i]}"])
    
            print(f"There are {len(self.train_images_path)} (HR/LR) pair images in the train images path.\n")

        #==========================================
        elif self.type == "validation":
            print("\nCreating validation dataset... this may take some time.\n")
            self.validation_images_path = []

            if len(os.listdir(f"{self.path}/HR")) != len(os.listdir(f"{self.path}/LR")):
                raise Exception("Not same number of (HR/LR) pair images in validation images path!!!")

            img_path_lr = f"{self.path}/LR"
            img_path_hr = f"{self.path}/HR"
            lr_img_list = os.listdir(img_path_lr)
            hr_img_list = os.listdir(img_path_hr)

            for i in tqdm.trange(len(lr_img_list)):
                img_lr = imread(f"{self.path}/LR/{lr_img_list[i]}")
                img_hr = imread(f"{self.path}/HR/{hr_img_list[i]}")

                if not(img_lr.shape[0] == 256 or img_lr.shape[1] == 256):
                    print("Doesn't support min size of low resolution dataset!!!\nImage is not saved!")
                    continue
            
                elif not(img_hr.shape[0] == 1024 or img_hr.shape[1] == 1024):
                    print("Doesn't support min size of high resolution dataset!!!\nImage is not saved!")
                    continue
                else:
                    self.validation_images_path.append([f"{self.path}/LR/{lr_img_list[i]}", f"{self.path}/HR/{hr_img_list[i]}"])
    
            print(f"There are {len(self.validation_images_path)} (HR/LR) pair images in the validation images path.\n")
            
        #==========================================
        elif self.type == "test":
            print("\nCreating test dataset... this may take some time.\n")
            self.test_images_path = []

            if len(os.listdir(f"{self.path}/HR")) != len(os.listdir(f"{self.path}/LR")):
                raise Exception("Not same number of (HR/LR) pair images in test images path!!!")

            img_path_lr = f"{self.path}/LR"
            img_path_hr = f"{self.path}/HR"
            lr_img_list = os.listdir(img_path_lr)
            hr_img_list = os.listdir(img_path_hr)
            for i in tqdm.trange(len(lr_img_list)):
                img_lr = imread(f"{self.path}/LR/{lr_img_list[i]}")
                img_hr = imread(f"{self.path}/HR/{hr_img_list[i]}")

                if not(img_lr.shape[0] == 256 or img_lr.shape[1] == 256):
                    print("Doesn't support min size of low resolution dataset!!!\nImage is not saved!")
                    continue
            
                elif not(img_hr.shape[0] == 1024 or img_hr.shape[1] == 1024):
                    print("Doesn't support min size of high resolution dataset!!!\nImage is not saved!")
                    continue
                else:
                    self.test_images_path.append([f"{self.path}/LR/{lr_img_list[i]}", f"{self.path}/HR/{hr_img_list[i]}"])
    
            print(f"There are {len(self.test_images_path)} (HR/LR) pair images in the test images path.\n")

        #==========================================
        count = 0
        for _, _, files in os.walk('./Dataset'):
            count += len(files)

        print(f"Main Dataset containing has {int(count/2)} pair of (HR/LR) images in total.\n")


    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        if self.type == "train":
            # Read image
            lr_img = imread(self.train_images_path[i][0])
            hr_img = imread(self.train_images_path[i][1])
            lr_img = convert_image(lr_img, 'tensor')
            hr_img = convert_image(hr_img, 'tensor')
            
            return lr_img, hr_img

        elif self.type == "validation":
            lr_img = imread(self.validation_images_path[i][0])
            hr_img = imread(self.validation_images_path[i][1])
            lr_img = convert_image(lr_img, 'tensor')
            hr_img = convert_image(hr_img, 'tensor')

            return lr_img, hr_img
        
        elif self.type ==  "test":
            lr_img = imread(self.test_images_path[i][0])
            hr_img = imread(self.test_images_path[i][1])
            lr_img = convert_image(lr_img, 'tensor')
            hr_img = convert_image(hr_img, 'tensor')

            return lr_img, hr_img


    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        if self.type == "train":
            return len(self.train_images_path)

        elif self.type == "validation":
            return len(self.validation_images_path)
        
        elif self.type ==  "test":
            return len(self.test_images_path)

        
