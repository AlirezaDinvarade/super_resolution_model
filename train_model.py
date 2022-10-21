import torch.backends.cudnn as cudnn
import torch
from torch import nn
from rlfn import RLFN
from datasets import SRDataset
from utils import *
import tqdm
import os
from eval_model_main import Eval
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.tensorboard import SummaryWriter
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.empty_cache()

# tensor writer
writer = SummaryWriter('./tensorboard_checkpoints', comment="shanbe")

# Data parameters
train_data_path = './Dataset/train'  # folder with train data files
validation_data_path = './Dataset/validation'
scaling_factor = 4  # the scaling factor
eval_per_epoch = 3 # evaluation per 10 epoch

# Learning parameters
checkpoint = './checkpoint' # path to model checkpoint
batch_size = 16  # batch size
workers = 4  # number of workers for loading data in the DataLoader
# print_freq = 1  # print training status once every __ batches
lr = 5e-4  # learning rate
grad_clip = None  # clip if gradients are exploding
# Total number of epochs to train for
epochs = 110

# Check device for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main():
    """
    Train model
    """
    print("Training Started:\n")
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if os.listdir(checkpoint) == []:
        start_epoch = 0
        model = RLFN(upscale=scaling_factor)
        # Move to default device
        model = model.to(device)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoints = os.listdir(checkpoint)
        saved_epochs = []
        index_of_ckp = 0
        for indx in range(len(checkpoints)):
                    saved_epochs.append(int(checkpoints[indx].split(sep="-")[1].split(sep=")")[0]))

        for k, num in enumerate(saved_epochs):
            if num == max(saved_epochs):
                index_of_ckp = k
        
        checkpoint = torch.load(f'./checkpoint/{os.listdir(checkpoint)[index_of_ckp]}')
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        # Move to default device
        model = model.to(device)
        optimizer = checkpoint['optimizer']

    
    
    # Loss initialization
    loss_choice = nn.MSELoss().to(device)
    # loss_mse = nn.MSELoss().to(device)


    # Custom dataloaders
    train_dataset = SRDataset(path=train_data_path, type="train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True) 

    # Create evaluator object for eval model result
    eval_model_run = Eval()
    

    # Epochs
    for epoch in tqdm.trange(start_epoch,
                             epochs,
                             desc="\nTraining Process: ",
                             bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):

        # One epoch's training
        train_loss_value, train_pnsr, train_ssim = train(train_loader=train_loader,
                            model=model,
                            loss=loss_choice,
                            optimizer=optimizer,
                            epoch=epoch)
        
        torch.save({'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer,
                        'loss': train_loss_value,
                        'pnsr': train_pnsr,
                        'ssim': train_ssim,
                        },
                    f'./checkpoint/checkpoint_RLFN_(epoch-{epoch}).pt')

        writer.add_scalar(tag="Train Loss" , scalar_value=train_loss_value ,global_step=epoch) 
        writer.add_scalar(tag="Train PSNR" , scalar_value=train_pnsr ,global_step=epoch)
        writer.add_scalar(tag="Train SSIM" , scalar_value=train_ssim ,global_step=epoch)

        if int(epoch % eval_per_epoch) == 0:

            # eval model
            eval_loss_value, eval_psnr , eval_ssim = eval_model_run.evaluate(epoch, model, loss_choice)

            # Save tensorboard
            writer.add_scalar(tag="Eval Loss" , scalar_value=eval_loss_value, global_step=epoch)
            writer.add_scalar(tag="Eval PSNR" , scalar_value=eval_psnr, global_step=epoch)
            writer.add_scalar(tag="Eval SSIM" , scalar_value=eval_ssim, global_step=epoch)

            writer.add_scalar(tag="Loss Difference" ,
                              scalar_value=train_loss_value - eval_loss_value,
                              global_step=epoch)


            


def train(train_loader, model, loss, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param loss: content loss function
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    losses = AverageMeter() # loss mean value
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Batches
    i=0
    for i, (lr_imgs, hr_imgs) in enumerate(tqdm.tqdm(train_loader,
                                                    desc=f'Epoch: [{epoch}] ',
                                                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        model_loss = loss(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        
        model_loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            sr_img = convert_image(sr_imgs, target='ndarray')
            hr_img = convert_image(hr_imgs, target='ndarray')
            psnr_list = []
            ssim_list = []
            for i in range(len(hr_img)):
                psnr_list.append(peak_signal_noise_ratio(hr_img[i].astype(np.uint8),
                                                        sr_img[i].astype(np.uint8), data_range=255))
            
                ssim_list.append(structural_similarity(hr_img[i].astype(np.uint8),
                                                    sr_img[i].astype(np.uint8), channel_axis=0, data_range=255))

            psnr = sum(psnr_list)/len(psnr_list)
            ssim = sum(ssim_list)/len(ssim_list)
            # Save to value tracker
            PSNRs.update(psnr)
            SSIMs.update(ssim)
            # Keep track of loss
            losses.update(model_loss.item())
        
        
            # Print status
            # if i % print_freq == 0:
            # print(f'\nLoss : {losses.val:.4f} | ({losses.avg:.4f}) | '
            #         f'PSNR : {PSNRs.val:.4f} | ({PSNRs.avg:.4f}) | '
            #         f'SSIM : {SSIMs.val:.4f} | ({SSIMs.avg:.4f})\n'
            #     , end = "\r")

    print(f"\n\n____________TRAIN_______________ Epoch-({epoch})________________________________\n")
    print(f'--->  Loss: {losses.avg} | PSNR : {PSNRs.avg:.3f} | SSIM : {SSIMs.avg:.3f}\n')
    # del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
    torch.cuda.empty_cache()
    return losses.avg, PSNRs.avg, SSIMs.avg


if __name__ == '__main__':
    main()
