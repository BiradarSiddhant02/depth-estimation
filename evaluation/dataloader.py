from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import random


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class DepthDataset(Dataset):
    def __init__(self, csv, root_dir, transform=None):
        self.traincsv = csv
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.traincsv)

    def __getitem__(self, idx):
        sample = self.traincsv[idx]
        img_name = os.path.join(self.root_dir, sample[0])
        image = Image.open(img_name).convert("RGB")

        # Convert to numpy array
        image_np = np.array(image)

        # Convert to grayscale
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Compute gradient magnitude
        gradient = np.gradient(gray_image.astype(np.float32))
        gradient_magnitude = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)

        # Normalize gradient and apply colormap
        gradient_colored = cv2.applyColorMap(
            np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude)),
            cv2.COLORMAP_JET,
        )
        gradient_colored = cv2.resize(
            gradient_colored, (image_np.shape[1], image_np.shape[0])
        )

        # Multiply gradient by 0.3
        gradient_colored = cv2.addWeighted(
            gradient_colored, 0.5, np.zeros_like(gradient_colored), 0, 0
        )
        image_with_gradient = cv2.add(image_np, gradient_colored)

        # Convert gradient back to PIL Image
        gradient_image = Image.fromarray(image_with_gradient)

        depth_name = os.path.join(self.root_dir, sample[1])
        depth = Image.open(depth_name)

        sample1 = {"image": gradient_image, "depth": depth}

        if self.transform:
            sample1 = self.transform(sample1)

        return sample1


class Augmentation(object):
    def __init__(self, probability):
        from itertools import permutations

        self.probability = probability
        # generate some output like this [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        self.indices = list(permutations(range(3), 3))
        # followed by randomly picking one channel in the list above

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        if not _is_pil_image(image):
            raise TypeError("img should be PIL Image. Got {}".format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError("img should be PIL Image. Got {}".format(type(depth)))

        # flipping the image
        if random.random() < 0.5:
            # random number generated is less than 0.5 then flip image and depth
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        # rearranging the channels
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(
                image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])]
            )

        return {"image": image, "depth": depth}


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        pic = np.array(pic)
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            if pic.ndim == 2:
                pic = pic[..., np.newaxis]

            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)
