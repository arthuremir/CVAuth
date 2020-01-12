import numpy as np
import torch
import torchvision
import cv2


image = np.transpose(np.random.randint(0, 2, size=(5,4,3)), (2, 0, 1))

image = torch.from_numpy(image)
image = image.type(torch.FloatTensor)

#output = model(image[None, :, :].cuda())
print(image, image.shape)