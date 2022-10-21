import torch
from torch import nn
from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import tifffile
np.random.seed(42)
torch.manual_seed(42)

class Eval():
    def __init__(self) -> None:

        # Data
        self.eval_data_path = "./Dataset/validation"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Evaluate
        validation_dataset = SRDataset(self.eval_data_path,type="validation")
        self.validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=16,
                                                            shuffle=False,
                                                            num_workers=4,
                                                            pin_memory=True)

        if not os.path.exists('./validation_output'):
            os.mkdir('./validation_output')

    def evaluate(self, epoch=None, model=None, loss_l1=None):

        # Keep track of the PSNRs and the SSIMs across batches
        PSNRs = AverageMeter()
        SSIMs = AverageMeter()
        # loss mean value
        losses = AverageMeter()

        
        # Record dict
        record_json = {
            "epoch": epoch,
            "PSNR_per_image" : [],
            "SSIM_per_image" : [],
            "Loss_per_image" : [],
            "PSNR_Mean" : 0,
            "SSIM_Mean" : 0,
            "Loss_Mean" : 0,
            }
        
        model.eval()

        # Prohibit gradient computation explicitly because I had some problems with memory
        with torch.inference_mode():
            # Batches
            for i, (lr_img, hr_img) in enumerate(tqdm(self.validation_loader,
                                                      desc="Evaluation Process:",
                                                      bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
                # Move to default device
                lr_imgs = lr_img.to(self.device)  # (batch_size (1), 3, w / 4, h / 4)
                hr_imgs = hr_img.to(self.device)  # (batch_size (1), 3, w, h)

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (1, 3, w, h)

                # Calculate loss
                loss = loss_l1(sr_imgs, hr_imgs)
                
                sr_img = convert_image(sr_imgs, target='ndarray')
                hr_img = convert_image(hr_imgs, target='ndarray')
                psnr_list = []
                ssim_list = []
                for j in range(len(hr_img)):
                    psnr_list.append(peak_signal_noise_ratio(hr_img[j].astype(np.uint8),
                                                            sr_img[j].astype(np.uint8), data_range=255))
                
                    ssim_list.append(structural_similarity(hr_img[j].astype(np.uint8),
                                                        sr_img[j].astype(np.uint8), channel_axis=0, data_range=255))

                psnr = sum(psnr_list)/len(psnr_list)
                ssim = sum(ssim_list)/len(ssim_list)
        

                # # Calculate PSNR and SSIM
                # sr_img = convert_image(sr_imgs, target='ndarray').squeeze(0).astype(np.uint8)
                # hr_img = convert_image(hr_imgs, target='ndarray').squeeze(0).astype(np.uint8)
                # psnr = peak_signal_noise_ratio(hr_img, sr_img, data_range=255)
                # ssim = structural_similarity(hr_img, sr_img, channel_axis=0, data_range=255)
                
                # Save to record dict
                record_json["PSNR_per_image"].append(psnr)
                record_json["SSIM_per_image"].append(ssim)
                record_json["Loss_per_image"].append(loss.item())

                # Save to value tracker
                PSNRs.update(psnr)
                SSIMs.update(ssim)
                losses.update(loss.item())
                
                lr_img = convert_image(lr_imgs, target='ndarray')
                # Save result of super resolution images as output tif image
                if i==0 :
                    # Check output folder exits
                    if not os.path.exists(f'./validation_output/epoch-{epoch}'):
                        os.mkdir(f'./validation_output/epoch-{epoch}')

                    for k in range(int(len(sr_img)/4)):
                        tifffile.imwrite(f'./validation_output/epoch-{epoch}/SR_{k}.tif', sr_img[k])
                        tifffile.imwrite(f'./validation_output/epoch-{epoch}/HR_{k}.tif', hr_img[k])
                        tifffile.imwrite(f'./validation_output/epoch-{epoch}/LR_{k}.tif', lr_img[k])
                
                # print(f'\nIter: [{i}]-[{i}/{len(self.validation_loader)-1}] | '
                #   f'Loss : {losses.val:.3f} | ({losses.avg:.3f}) | '
                #   f'PSNR : {PSNRs.val:.3f} | ({PSNRs.avg:.3f}) | '
                #   f'SSIM : {SSIMs.val:.4f} | ({SSIMs.avg:.4f})\n'
                # )
                    
        # Save mean value to record dict
        record_json["PSNR_Mean"] = PSNRs.avg
        record_json["SSIM_Mean"] = SSIMs.avg
        record_json["Loss_Mean"] = losses.avg

        # Saving record data as json
        with open(f"./validation_output/record_epoch-{epoch}.json", "w") as f:
            f.write(json.dumps(record_json, indent=4))

        # Print average PSNR and SSIM finally
        print(f"\n\n_____________EVAL______________Epoch-({epoch})________________________________\n")
        print(f'--->  Loss: {losses.avg} | PSNR : {PSNRs.avg:.3f} | SSIM : {SSIMs.avg:.3f}\n')
        torch.cuda.empty_cache()
        return losses.avg, PSNRs.avg, SSIMs.avg
