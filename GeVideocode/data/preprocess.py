import imageio

imageio.plugins.ffmpeg.download()

import torchvision.transforms.functional as functional
import torchvision.transforms as transforms
import torch
from .statefultransforms import StatefulRandomCrop, StatefulRandomHorizontalFlip

def load_video(filename, startframe):
    """Loads the specified video using ffmpeg.

    Args:
        filename (str): The path to the file to load.
            Should be a format that ffmpeg can handle.

    Returns:
        List[FloatTensor]: the frames of the video as a list of 3D tensors
            (channels, width, height)"""
    try:
        vid = imageio.get_reader(filename,  'ffmpeg')
    except:
        print("Error Happedn at: "+filename)

    frames = []
    for i in range(0, 45):
        image = vid.get_data(startframe+i)
        image = functional.to_tensor(image)
        frames.append(image)
    return frames

def bbc(vidframes, augmentation=True):
    """Preprocesses the specified list of frames by center cropping.
    This will only work correctly on videos that are already centered on the
    mouth region, such as LRITW.

    Args:
        vidframes (List[FloatTensor]):  The frames of the video as a list of
            3D tensors (channels, width, height)

    Returns:
        FloatTensor: The video as a temporal volume, represented as a 5D tensor
            (batch, channel, time, width, height)"""

    temporalvolume = torch.FloatTensor(1,45,240,320)
    
    Resize = transforms.Resize((240,320))

    if(augmentation):
        Resize = transforms.Compose([
            #transforms.ColorJitter(brightness=0.5),
            transforms.Resize((240,320))
        ])
    for i in range(0, 45):
        result = transforms.Compose([
            transforms.ToPILImage(),
            Resize,
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            #transforms.Normalize([0.5,],[0.5,]),
        ])(vidframes[i])

        temporalvolume[0][i] = result

    return temporalvolume
