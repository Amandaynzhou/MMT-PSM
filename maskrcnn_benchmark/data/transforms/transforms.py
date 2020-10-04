# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import torchvision.transforms.functional as tf
import numpy as np
from PIL import Image
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target = None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size


        if max_size is not None:
            if w == max_size and h == max_size:
                # this is the case for ISBI2015 val (1024,1024)
                return (h,w)
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        # print('size', size)
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        # print('img', image.size)
        image = F.resize(image, size)
        # print('before', target)
        if target is not None:
            target = target.resize(image.size)
        # print('after', target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img,target):
        return tf.adjust_gamma(img, random.uniform(1,
                                                   1 +
                                                   self.gamma)),target

class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, target):

        return tf.adjust_saturation(img,
                                        random.uniform(1 - self.saturation,
                                                       1 + self.saturation)), target

class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, target):
        return tf.adjust_hue(img, random.uniform(-self.hue,
                                                      self.hue)), target

class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, target):
        return tf.adjust_brightness(img,
                                        random.uniform(1 - self.bf,
                                                       1 + self.bf)), target
class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, target):
        return tf.adjust_contrast(img,
                                  random.uniform(1 - self.cf,
                                                 1 + self.cf)), target


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''


    def __init__(self, prob):
        self.prob = prob
        self.erasor = self.get_random_eraser()
    def get_random_eraser(self,  s_l=0.001, s_h=0.004, r_1=0.2,
                          r_2=1 / 0.2, v_l=0, v_h=255, pixel_level=True):
        p = self.prob
        def eraser(input_img):
            p_1 = np.random.rand()

            if p_1 > p:
                return input_img

            img_h, img_w, img_c = input_img.shape


            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            if pixel_level:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            else:
                c = np.random.uniform(v_l, v_h)

            input_img[top:top + h, left:left + w, :] = c

            return input_img

        return eraser

    def __call__(self, img,target):

        num = random.randint(0, 10)
        img = np.array(img)
        # print('before', img.shape)
        for _ in range(num):
            img = self.erasor(img)
        img = Image.fromarray(img, mode="RGB")
        # print('after', img.size)
        return img, target