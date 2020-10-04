import os
import scipy.io as sio
import cv2
import numpy as np
from collections import defaultdict
import pdb

def get_nuclei_label(nuclei_mask, ori_img, mode='xyxy', location=None):
    assert mode in ('xyxy', 'xywh')

    # resize the image to MPP=0.2529
    h_ori, w_ori = ori_img.shape[:2]
    nuclei_mask = cv2.resize(nuclei_mask, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)

    # crop prediction inside location

    if location is not None:
        x, y, w, h = location
        # padding if right and down side out of original img
        if (x + w) > w_ori or (y + h) > h_ori:
            mask = np.zeros((y + h, x + w),dtype= nuclei_mask.dtype)
            mask[:h_ori, :w_ori] = nuclei_mask
        else:
            mask = nuclei_mask
        # crop into specific region:
        nuclei_mask = mask[y: y + h, x: x + w]
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(nuclei_mask, connectivity=8)
    nuclei = defaultdict(list)

    for i in range(1, retval):
        nuclei['mask'].append(labels == i)
        if mode == 'xyxy':
            nuclei['bbox'].append(
                [stats[i][0], stats[i][1], stats[i][0] + stats[i][2] - 1, stats[i][1] + stats[i][3] - 1])
        elif mode == 'xywh':
            nuclei['bbox'].append([stats[i][0], stats[i][1], stats[i][2], stats[i][3]])
        # the label of nuclei is 1
        nuclei['label'].append(1)

    return nuclei


def get_cytoplasm_label(LSF5, ori_img, mode='xyxy', location=None):
    assert mode in ('xyxy', 'xywh')

    h_ori, w_ori = ori_img.shape[:2]
    cytoplasm = defaultdict(list)

    for i in range(len(LSF5)):
        cytoplasm_mask = LSF5[i][0]

        # resize the image to MPP=0.2529
        cytoplasm_mask = cv2.resize(cytoplasm_mask, (w_ori, h_ori), interpolation=cv2.INTER_NEAREST)

        # crop prediction inside location
        if location is not None:
            x, y, w, h = location
            # padding if right and down side out of original img
            if (x + w) > w_ori or (y + h) > h_ori:
                mask = np.zeros((y + h, x + w),dtype= cytoplasm_mask.dtype)
                mask[:h_ori, :w_ori] = cytoplasm_mask
            else:
                mask = cytoplasm_mask
            # crop into specific region:
            cytoplasm_mask = mask[y: y + h, x: x + w]

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(cytoplasm_mask, connectivity=8)
        if retval >=3:
            print('split the cell because the box region')
        # assert retval <= 2, retval

        if retval == 1:
            # do not contain a mask
            continue

        cytoplasm['mask'].append(labels == 1)
        if mode == 'xyxy':
            cytoplasm['bbox'].append(
                [stats[1][0], stats[1][1], stats[1][0] + stats[1][2] - 1, stats[1][1] + stats[1][3] - 1])
        elif mode == 'xywh':
            cytoplasm['bbox'].append([stats[1][0], stats[1][1], stats[1][2], stats[1][3]])
        # the label of cytoplasm is 2
        cytoplasm['label'].append(2)

    return cytoplasm


def process_folder(root, folder):
    ori_path = os.path.join(root, folder, '{}.png'.format(folder))
    ori_img = cv2.imread(ori_path)

    #     nuclei_path = os.path.join(root, folder, r"Common\NucleiMask.mat")
    nuclei_path = os.path.join(root, folder, "Common/NucleiMask.mat")
    nuclei_mask = sio.loadmat(nuclei_path)
    nuclei_mask = nuclei_mask['NucleiMaskSet'][0][0]

    #     LSF5_path = os.path.join(root, folder, r"LSF\LSF5\LSF_5_beta_5_kappa_13_chi_3_iterIn_20_iterOut_2.mat")
    LSF5_path = os.path.join(root, folder, "LSF/LSF5/LSF_5_beta_5_kappa_13_chi_3_iterIn_20_iterOut_2.mat")
    LSF5 = sio.loadmat(LSF5_path)
    LSF5 = LSF5['LSF_5'][0][0]

    nuclei = get_nuclei_label(nuclei_mask, ori_img)
    cytoplasm = get_cytoplasm_label(LSF5, ori_img)

    return nuclei, cytoplasm


if __name__ == "__main__":
    print("Need to specify the root path of the output folder.")
    # root = r'C:\Users\jiaqi\data\ISBI\output'
    root = './output'
    folders = os.listdir(root)
    print(folders)

    output = {}
    for folder in folders:
        #     print(folder)
        result = {}
        nuclei, cytoplasm = process_folder(root, folder)
        result['mask'] = nuclei['mask'] + cytoplasm['mask']
        result['bbox'] = nuclei['bbox'] + cytoplasm['bbox']
        result['label'] = nuclei['label'] + cytoplasm['label']

        output[folder] = result
