# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import numpy as np
from skimage import measure
import pdb
# from pap_synthetic import flattenlists
import itertools

from pycocotools import mask as maskUtils
import torch
import cv2
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def flattenlists(polygons):
    polygons_new = []

    for polygon in polygons:
        # print(polygon)
        polygon = np.array(polygon, np.int)
        polygon= list(itertools.chain.from_iterable(polygon.tolist()))
        polygons_new.append(polygon)
    return polygons_new


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def _hflip(tensor):
    return torch.flip(tensor, (3,))

def batch_hfilp(tensor):
    if isinstance(tensor,list):
        t_list = []
        for a in tensor:
            t_list.append(_hflip(a))
        return t_list
    elif isinstance(tensor,tuple):
        t_list = []
        for a in tensor:
            t_list.append(_hflip(a))
        return t_list
    else:
        return _hflip(tensor)

def batch_boxlist_hflip(boxlists):
    flip = []
    for box in boxlists:
        flip.append(box.transpose(0))
    return flip

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """

    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons


def maskToPolygons(binary_mask):
    '''
    :param mask: binary mask
    :return: polygon in array
    '''
    contours = measure.find_contours(binary_mask.astype(np.uint8), 0.5)
    polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        mask = np.asarray(segmentation)
        mask = np.reshape(mask, (int(len(mask) / 2), 1, 2)).astype(np.int32)
        if len(segmentation)>=4:
            polygons.append(mask)
        else:
            pdb.set_trace()
    return polygons

def polys_to_mask(polygons, height, width):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
    rle = maskUtils.frPyObjects(polygons, height, width)
    mask = np.array(maskUtils.decode(rle), dtype=np.float32)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.uint8)
    return mask

def mask_to_bbox(mask):
    """Compute the tight bounding box of a binary mask."""
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)

def poly_to_box(poly):
    """Convert a list of polygons into an array of tight bounding boxes."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    boxes_from_polys = [x0, y0, x1, y1]
    return boxes_from_polys

def maskToList(binary_mask):
    h,w = binary_mask.shape
    polygons = maskToPolygons(binary_mask)
    polygon_list = []
    for polygon in polygons:
        polygon = polygon[:, 0, :]
        polygon = polygon.tolist()
        polygon_list.append(polygon)
    # if len(polygons)>1:
    #     polygon = polygons[0][:, 0, :]
    #     polygon = polygon.tolist()
    #
    #
    #     return False, polygon, (h,w)
    #
    # polygon = polygons[0][:,0,:]
    #
    # polygon = polygon.tolist()
    return True, polygon_list,(h,w)

def rle_maskes_to_boxes2(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""


    decoded_masks = [
        np.array(maskUtils.decode(rle), dtype=np.float32) for rle in masks
    ]

    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return inds.min(), inds.max()

    boxes = np.zeros((len(decoded_masks), 4))
    keep = [True] * len(decoded_masks)
    mask_lists = []
    for i, mask in enumerate(decoded_masks):

        if mask.sum() == 0:
            keep[i] = False
            continue


        # mask_list = binary_mask_to_polygon(mask, 0)
        _,mask_list,_ = maskToList(mask)
        flat_mask = mask.sum(axis=0)
        x0, x1 = get_bounds(flat_mask)
        flat_mask = mask.sum(axis=1)
        y0, y1 = get_bounds(flat_mask)
        boxes[i, :] = (x0, y0, x1, y1)
        mask_list = flattenlists(mask_list)
        mask_lists.append(mask_list)

    return boxes, mask_lists, np.where(keep)[0]





def rle_maskes_to_boxes(masks):
    """Computes the bounding box of each mask in a list of RLE encoded masks."""


    decoded_masks = [
        np.array(maskUtils.decode(rle), dtype=np.float32) for rle in masks
    ]
    # pdb.set_trace()
    def get_bounds(flat_mask):
        inds = np.where(flat_mask > 0)[0]
        return inds.min(), inds.max()

    boxes = np.zeros((len(decoded_masks), 4))
    keep = [True] * len(decoded_masks)
    mask_lists = []
    for i, mask in enumerate(decoded_masks):

        if mask.sum() == 0:
            keep[i] = False
            continue


        mask_list = binary_mask_to_polygon(mask, 0)
        # _,mask_list,_ = maskToList(mask)
        flat_mask = mask.sum(axis=0)
        x0, x1 = get_bounds(flat_mask)
        flat_mask = mask.sum(axis=1)
        y0, y1 = get_bounds(flat_mask)
        boxes[i, :] = (x0, y0, x1, y1)
        # mask_list = flattenlists(mask_list)
        mask_lists.append(mask_list)

    return boxes, mask_lists, np.where(keep)[0]

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def sigmoid_rampdown(gap_time, rampdown_length):
    if rampdown_length == 0:
        return 1.0
    else:
        phase = 1.0 - gap_time/rampdown_length
        return float(np.exp(-12 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        tensor = tensor.permute(1,2,0)
        return tensor


def rles_to_binary_mask(labels, contour = False, type = 1):
    '''

    :param labels:
    :param contour:
    :param type: 1: cyto and nuclei in one mask, cyto-1, nuclei-2
            type 2: cyto has overlap, cyto and nuclei in two mask
    :return:
    '''
    rles = labels['nuclei']
    decoded_masks = [
        np.array(maskUtils.decode(rle), dtype=np.float32) for rle in
        rles
    ]
    if not contour:
        binary_masks1 = sum(decoded_masks)

    else:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        binary_masks1 = np.zeros_like(decoded_masks[0])
        for instance in decoded_masks:
            gradient = cv2.morphologyEx(instance, cv2.MORPH_GRADIENT,
                                        kernel, iterations=2)
            gradient = abs(gradient)
            binary_masks1 = gradient + binary_masks1
        binary_masks1 = 1*(binary_masks1>0)

    rles = labels['cyto']
    decoded_masks = [
        np.array(maskUtils.decode(rle), dtype=np.float32) for rle in
        rles
    ]
    if not contour:
        binary_masks2 = sum(decoded_masks)
    else:
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        binary_masks2 = np.zeros_like(decoded_masks[0])
        for instance in decoded_masks:
            gradient = cv2.morphologyEx(instance, cv2.MORPH_GRADIENT,
                                        kernel, iterations=2)
            gradient = abs(gradient)
            binary_masks2 = gradient + binary_masks2
        binary_masks2 = 1 * (binary_masks2 > 0)
    if not contour:
        if type==1:
            binary_masks2[binary_masks2>1] = 3# overlapping
            binary_masks = binary_masks2.copy()
            binary_masks[binary_masks1>=1] = 2
            binary_masks1 = binary_masks
            # binary_masks1[binary_masks1>=1]=2
            # binary_masks1[binary_masks2>0]=1

        else:
            binary_masks1 = np.stack((binary_masks1,binary_masks2),-1)

    else:
        binary_masks1 = binary_masks1 + binary_masks2
        binary_masks1[binary_masks1>1] = 1
        # binary_masks = ((binary_masks1 + binary_masks2)>0 )*1
    return binary_masks1
