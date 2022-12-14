import torchvision.transforms.functional as FT
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_image(img, target:str):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of ndarray, tensor
    :return: converted image
    """
    assert target in {'ndarray', 'tensor'}, f"Cannot convert to target format {target}!"

    # Convert from source to tensor
    if target == 'tensor':
        img = FT.to_tensor(img).type(torch.float)

    # Convert from source to ndarray
    elif target == 'ndarray':
        img = img.detach().cpu().numpy()

    return img


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    :param state: checkpoint contents
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
