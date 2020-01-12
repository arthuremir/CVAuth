import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import cv2

img = Image.open('gestures/gesture_data/five_fingers/2020-01-12-13-45-53.772685.jpg')

data_transform = transforms.Compose([
    transforms.RandomAffine(25,
                            (0.15, 0.15),
                            (0.7, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

img_tr = data_transform(img)
# output = model(image[None, :, :].cuda())
#print(image, image.shape)
