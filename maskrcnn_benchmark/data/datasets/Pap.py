import torch
import torch.utils.data
import numpy as np
import openslide as ops
import slidingwindow as sw
import itertools
import copy
import os
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from tqdm import tqdm
import os.path
import pdb
import time
from pycocotools import mask as maskUtils
from PIL import Image
import json
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

windowSizeRatio = {
    'train': 1.0,
    'val': 1.0,
    'test': 1.0,
    'gen': 1.0
}
overlapPercent = {
    'train': 0.75,
    'val': 0,
    'test': 0,
    'gen': 0,
}

folds = {
    'train': ('1', '2'),
    'val': ('3'),  # 0:5
    'test': ('3',)  # 5:-1 # *IMPORTANT* train&val should not contain any img used in test
}
CV = 1

def _get_img_list(split, root, folddict, easy=True, hard=True):
    imglist = []
    # TODO: GET YOU OWN DATASET IMAGELIST
    # imglist = ['AGC1_BD1601497']
    return imglist


def _get_sw(w, h, windowSize, overlapPercent):
    # reture: location (x, y, w, h)
    windows = sw.generateForSize(w, h, sw.DimOrder.HeightWidthChannel,
                                 windowSize, overlapPercent,
                                 transforms=[])
    windows = [window.getRect() for window in windows]
    return windows


def _check_if_in_crop_img(mask, location):
    """
    Check if mask has points in the location range
    :para mask (int array) : polygon of mask
    :location (tuple)      : sliding window (x, y, w, h)
    """
    x = (np.greater(mask[:, 0], location[0])) & (
        np.less(mask[:, 0], (location[0] + location[2])))
    y = (np.greater(mask[:, 1], location[1])) & (
        np.less(mask[:, 1], (location[1] + location[3])))
    return np.logical_and(x, y).any()


def _modify_out_of_roi_masks(polygon, location):
    """
    Args:
        polygon (numpy array, np.int) : polygons (n*2)
        location (tuple)              : sliding window (x, y, w, h)
    Returns:
        mask (numpy array)            : mask after cropping
    ISSUE!!
        bbox not tight!!
    """
    #####
    # pdb.set_trace()
    mask = polygon.copy()

    mask[:, 0] = (polygon[:, 0] - location[0])
    mask[:, 1] = (polygon[:, 1] - location[1])
    in_roi_w = \
    np.nonzero((mask[:, 0] >= 0) * (mask[:, 0] <= location[2]))[0]
    in_roi_h = \
    np.nonzero((mask[:, 1] >= 0) * (mask[:, 1] <= location[3]))[0]
    in_roi = list(set(in_roi_h) & set(in_roi_w))
    in_roi_mask = mask[in_roi]
    max_w = max(in_roi_mask[:, 0])
    min_w = min(in_roi_mask[:, 0])
    max_h = max(in_roi_mask[:, 1])
    min_h = min(in_roi_mask[:, 1])
    mask[:, 0] = mask[:, 0].clip(min=min_w, max=max_w)
    mask[:, 1] = mask[:, 1].clip(min=min_h, max=max_h)
    #####
    # mask = polygon.copy()
    # mask[:, 0] = (polygon[:, 0] - location[0]).clip(min=0, max=location[2])
    # mask[:, 1] = (polygon[:, 1] - location[1]).clip(min=0, max=location[3])
    return mask



class PapNucleiDataset(torch.utils.data.Dataset):
    # currently this class is only used in eval/test
    CLASSES = (
        "__background__ ",
        "cytoplasm",
        "nuclei",
    )

    def __init__(
            self, root, annFile, split,
            remove_images_without_annotations=True,
            transforms=None,
            maxWindowSize=1000, foldfile='./split.json', use_gen=True,
            gen_fake=0
            , gen_true=0,tta = False):
        from maskrcnn_benchmark.data.datasets.cell import CELL
        self.root = root
        self.cv = 1
        self.annFile = annFile
        self.cell = CELL(annFile)
        assert split in ('train', 'val', 'test',
                         'gen'), "split error, (train, val, test) available"
        self.split = split
        self.transforms = transforms
        self.maxWS = maxWindowSize
        self.ids = []
        self.image_pool = {}
        # load train/val/test fold
        self.gen_fake = gen_fake
        self.gen_true = gen_true
        val_fold = json.load(open(foldfile, 'r'))
        use_img_list = _get_img_list(split, root, val_fold, easy=True,
                                     hard=True)
        tic = time.time()
        cell = self.cell
        winSize = int(self.maxWS * windowSizeRatio[self.split])
        for img_id, img in cell.imgs.items():
            info = cell.loadImgs(img_id)[0]
            file_name = info['file_name'].split('.')[0]
            if file_name not in use_img_list:
                continue
            w, h = img['width'], img['height']
            windows = _get_sw(w, h, winSize, overlapPercent[split])
            if img_id not in self.image_pool:
                self.image_pool[img_id] = ops.open_slide(
                    os.path.join(root, info['file_name']))
            # filter images without detection annotations and only noise/tiny annotation
            id_sublist = [{
                'id': info['id'],
                'file_name': file_name,
                'img_id': img_id,
                'location': wsize,
            } for wsize in windows]
            if remove_images_without_annotations:
                id_sublist = self._remove_unsuitable_patches(
                    id_sublist,
                    thresh=0.001 * winSize ** 2)
            self.ids.extend(id_sublist)
        start_id = len(self.ids)
        self.start_id = start_id
        self.labels = {}
        self.tmp_train_patches_dir = '../temp_patch' +'_'+str(self.cv)
        self.fine_tune_source = False
        self.store_tmp_train_patches()

        self.category_name_to_contiguous_id = {
            c: i for i, c in enumerate(PapNucleiDataset.CLASSES)
        }
        self.contiguous_id_to_category_name = {
            v: k for k, v in
        self.category_name_to_contiguous_id.items()
        }
        cls = PapNucleiDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.contiguous_category_id_to_json_id = {v: k for k, v in
                                                  self.class_to_ind.items()}

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        print('Done (time={:0.2f}s\n) '.format(time.time() - tic))
        print('PapNuclei Dataset has %d patch images...' % (
            len(self.ids)))
        print('%d gen fake images; %d gen true images'%(len(
            self.gen_fake_image_list), len(self.gen_true_image_list)))

    def source_length(self):
        return (self.start_id - 1)

    def store_tmp_train_patches(self):
        # pdb.set_trace()
        self.patch_dir = {}
        if not (os.path.exists(self.tmp_train_patches_dir ) or\
            len(os.listdir(self.tmp_train_patches_dir)) == 0):
            mkdir(self.tmp_train_patches_dir)
            print('create temp patch files...')

            for idx in tqdm(range(self.start_id)):
                item = self.ids[idx]
                img_id = item['img_id']
                img = self.image_pool[img_id].read_region(
                    (item['location'][0], item['location'][1]), 0,
                    (1000,1000)).convert("RGB")
                img_path = os.path.join(self.tmp_train_patches_dir,
                                      item['file_name']+ '_'+str(item['location'][0])
                                      + '_'+str(item['location'][1])
                                      +'.png')
                img.save(img_path)
                self.patch_dir[idx] = img_path
        else:
            print('already have patch files, reuse it')
            for idx in range(self.start_id):
                item = self.ids[idx]
                img_path = os.path.join(self.tmp_train_patches_dir,
                                        item['file_name'] + '_' + str(
                                            item['location'][0])
                                        + '_' + str(
                                            item['location'][1])
                                        + '.png')
                self.patch_dir[idx] = img_path

    def __getitem__(self, index):
        masks = []
        while len(masks) == 0:
            item = self.ids[index]
            img_id = item['img_id']
            winSize = int(
                self.maxWS * windowSizeRatio[self.split])
            if self.split == 'train':
                img_path = self.patch_dir[index]
                img = Image.open(img_path)
            else:

                img = self.image_pool[img_id].read_region(
                    (item['location'][0], item['location'][1]), 0,
                    (winSize, winSize)).convert("RGB")
            # load and filter gt mask& bbox
            # print(img.size)
            masks, bboxes, labels = self.filter_gt(item)
            if len(masks) == 0:
                print('No mask for index:', index)
                index = np.random.choice(len(self.ids))
        bboxes = torch.as_tensor(bboxes).reshape(-1,4)  # guard against no boxes
        target = BoxList(bboxes, img.size, mode='xyxy')
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)
        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, index
    # cytoplasm
    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        # img_id = self.id_to_img_map[index]
        img_data = self.ids[index]['location']
        return {"width": img_data[2], "height": img_data[3],
                "file_name": self.ids[index]['file_name']}

    def get_img_in_inference(self, img_name):
        """
        haven't fixed
        Note that the key of image_pool now is "image_id" rather than "file_name"
        """
        file_name, w, h = img_name.split('~')
        # pdb.set_trace()
        try:
            # it seems that there is a wierd bug, read the image first time will caused trunct
            img = self.image_pool[file_name].read_region(
                (int(w), int(h)), 0, (
                    self.maxWS, self.maxWS)).convert("RGB")
        except:
            img = self.image_pool[file_name].read_region(
                (int(w), int(h)), 0, (
                    self.maxWS, self.maxWS)).convert("RGB")

        img = np.asarray(img, dtype=np.float32)
        return img

    def get_ground_truth(self, item):
        """
        haven't test
        for inference only, return mask in rle modec
        Args:
            item (img_id, location(x, y, w, h)) : image_id and its sliding window
        """
        ratio = 0.005 if self.split == 'test' else 0.05
        masks, bboxes, labels = self.filter_gt(item, ratio)
        if len(masks) == 0:
            pdb.set_trace()
        # if item['location'][2] == self.maxWS and item['location'][3] == self.maxWS:
        #     pdb.set_trace()
        # assert item['location'][2] == self.maxWS and item['location'][3] == self.maxWS, "Error in bbox size."
        bboxes = torch.as_tensor(bboxes).reshape(-1,
                                                 4)  # guard against no boxes
        target = BoxList(bboxes, (self.maxWS, self.maxWS),
                         mode='xyxy')
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        # masks = SegmentationMask(masks, (img_id['location'][2], img_id['location'][3]))
        # pdb.set_trace()
        rle_masks = self.annToRLE(masks, item)
        target.add_field("masks", rle_masks)
        return target

    def filter_gt(self, item, ratio=0.005):
        """
        Get mask, bbox, label in the given image within the sliding window.
        Args:
            item (img_id, location(x, y, w, h)) : image_id and its sliding window
            ratio (float)                       : the minimal area ratio need preserved (cropped bbox area >= original bbox area * ratio)
        Returns:
            mask (int list)
            bbox
            label
        """
        tic = time.time()
        cell = self.cell
        img_id = item['img_id']
        ann_ids = cell.getAnnIds(imgIds=img_id)
        anns = cell.loadAnns(ann_ids)
        location = item['location']

        mask, bbox, label = [], [], []
        for ann in anns:
            mask_ori = np.array(ann['segmentation'], np.int)
            if mask_ori.shape[0] <= 2:
                # remove the annotation whose mask coordinate number <= 2. (line or point cannot be considered as a mask)
                continue
            if not _check_if_in_crop_img(mask_ori, location):
                continue
            mask_crop = _modify_out_of_roi_masks(mask_ori, location)
            # bbox_crop (xmin, ymin, xmax, ymax)
            bbox_crop = [mask_crop[:, 0].min(), mask_crop[:, 1].min(),
                         mask_crop[:, 0].max(), mask_crop[:, 1].max()]
            # remove bbox of which the majority outside the patch
            bbox_area_ori = ann['bbox'][2] * ann['bbox'][3]
            bbox_area_crop = (bbox_crop[2] - bbox_crop[0]) * (
                    bbox_crop[3] - bbox_crop[1])
            if bbox_area_crop < bbox_area_ori * ratio:
                continue

            # polygons: a list of list of lists of numbers.
            mask_crop = [
                list(itertools.chain.from_iterable(
                    mask_crop.tolist()))]
            mask.append(mask_crop)
            bbox.append(bbox_crop)
            # class +1 because in json file, we begin at class 0 and ignore the background
            # pdb.set_trace()
            label.append(ann['category_id'] + 1)

        return mask, bbox, label

    def _remove_unsuitable_patches(self, id_list, thresh):
        """
        remove the patches with no masks in the location region
        or with only small ratio of masks ( ratio*w*h )
        Args:
            id_list [(img_id, location(x, y, w, h))] : image_id and its sliding window
            thresh                                   : bbox area need to be greater than thresh
        Returns:
            ids : id list satisfying the constrains
        """
        # remove the patches with only small ratio of masks ( ratio*w*h )
        ids = []
        for item in id_list:
            # pdb.set_trace()
            mask, bbox, label = self.filter_gt(item)
            if len(label) == 0:
                continue
            area = sum(
                [(obj[2] - obj[0]) * (obj[3] - obj[1]) for obj in
                 bbox])
            if area > thresh:
                ids.append(item)
        return ids

    def _reset_size(self, anno):
        """
        Haven't test
        May have problem anno['id']
        """
        size = int(self.maxWS * windowSizeRatio[self.split])
        anno['id']['location'][2] = size
        anno['id']['location'][3] = size
        return anno

    def annToRLE(self, masks, item):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        size = int(self.maxWS * windowSizeRatio[self.split])

        h, w = size, size
        RLES = []
        if isinstance(masks, SegmentationMask):
            for segm in masks.polygons:
                rles = maskUtils.frPyObjects(
                    [p.numpy() for p in segm], h, w
                )
                rle = maskUtils.merge(rles)
                RLES.append(rle)
        elif type(masks) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            for polygon in masks:
                rles = maskUtils.frPyObjects(polygon, h, w)
                rle = maskUtils.merge(rles)
                RLES.append(rle)
        else:
            raise NotImplementedError
        return RLES

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(
            self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Annotation File: {}\n'.format(self.annFile)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transforms.__repr__().replace(
                                         '\n',
                                         '\n' + ' ' * len(
                                             tmp)))
        return fmt_str

class PapNucleiSourceDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "cytoplasm",
        "nuclei",
    )

    def __init__(
            self, root, annFile, split,
            remove_images_without_annotations=True,
            transforms=None,
            maxWindowSize=1000, foldfile='./split.json',
            use_gen=False,
            gen_fake=0
            , gen_true=0, ratio =1., tta= False):
        from maskrcnn_benchmark.data.datasets.cell import CELL
        self.root = root
        self.cv = 1
        self.annFile = annFile
        self.cell = CELL(annFile)
        assert split in ('train', 'val', 'test',
                         'gen'), "split error, (train, val, test) available"
        self.split = split
        self.transforms = transforms
        self.maxWS = maxWindowSize
        self.ids = []
        self.image_pool = {}
        # load train/val/test fold
        self.gen_fake = gen_fake
        self.gen_true = gen_true
        val_fold = json.load(open(foldfile, 'r'))
        use_img_list = _get_img_list(split, root, val_fold, easy=True,
                                     hard=True)
        tic = time.time()
        cell = self.cell
        winSize = int(self.maxWS * windowSizeRatio[self.split])
        for img_id, img in  sorted(cell.imgs.items()):
            info = cell.loadImgs(img_id)[0]
            file_name = info['file_name'].split('.')[0]
            if file_name not in use_img_list:
                continue
            w, h = img['width'], img['height']
            windows = _get_sw(w, h, winSize, overlapPercent[split])
            if img_id not in self.image_pool:
                self.image_pool[img_id] = ops.open_slide(
                    os.path.join(root, info['file_name']))
            # filter images without detection annotations and only noise/tiny annotation
            id_sublist = [{
                'id': info['id'],
                'file_name': file_name,
                'img_id': img_id,
                'location': wsize,
            } for wsize in windows]
            if remove_images_without_annotations:
                id_sublist = self._remove_unsuitable_patches(
                    id_sublist,
                    thresh=0.001 * winSize ** 2)
            self.ids.extend(id_sublist)
        if split == 'train':
            _total = len(self.ids)
            self.ids = self.ids[: int(_total*ratio)]
        self.start_id = len(self.ids)
        self.labels = {}
        self.tmp_train_patches_dir =  '../temp_patch' +'_'+str(self.cv)
        self.store_tmp_train_patches()

        self.category_name_to_contiguous_id = {
            c: i for i, c in enumerate(PapNucleiDataset.CLASSES)
        }
        self.contiguous_id_to_category_name = {
            v: k for k, v in
        self.category_name_to_contiguous_id.items()
        }
        cls = PapNucleiDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.contiguous_category_id_to_json_id = {v: k for k, v in
                                                  self.class_to_ind.items()}

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        print('Done (time={:0.2f}s\n) '.format(time.time() - tic))
        print('PapNuclei Dataset has %d source patch images...' % (
            len(self.ids)))

    def store_tmp_train_patches(self):
        # pdb.set_trace()
        self.patch_dir = {}
        if not (os.path.exists(self.tmp_train_patches_dir ) or\
            len(os.listdir(self.tmp_train_patches_dir)) == 0):
            mkdir(self.tmp_train_patches_dir)
            print('create temp patch files...')

            for idx in tqdm(range(self.start_id)):
                item = self.ids[idx]
                img_id = item['img_id']
                img = self.image_pool[img_id].read_region(
                    (item['location'][0], item['location'][1]), 0,
                    (1000,1000)).convert("RGB")
                img_path = os.path.join(self.tmp_train_patches_dir,
                                      item['file_name']+ '_'+str(item['location'][0])
                                      + '_'+str(item['location'][1])
                                      +'.png')
                img.save(img_path)
                self.patch_dir[idx] = img_path
        else:
            print('already have patch files, reuse it')
            for idx in range(self.start_id):
                item = self.ids[idx]
                img_path = os.path.join(self.tmp_train_patches_dir,
                                        item['file_name'] + '_' + str(
                                            item['location'][0])
                                        + '_' + str(
                                            item['location'][1])
                                        + '.png')
                self.patch_dir[idx] = img_path

    def __getitem__(self,index):
        masks = []
        while len(masks) == 0:
            item = self.ids[index]

            img_id = item['img_id']
            winSize = int(
                self.maxWS * windowSizeRatio[self.split])
            if self.split == 'train':
                img_path = self.patch_dir[index]
                img = Image.open(img_path)
            else:

                img = self.image_pool[img_id].read_region(
                    (item['location'][0], item['location'][1]), 0,
                    (winSize, winSize)).convert("RGB")

            masks, bboxes, labels = self.filter_gt(item)
            if len(masks) == 0 or (1 not in set(labels) or 2 not
                                   in set(labels)):
                print('No mask for index:', index)
                index = np.random.choice(len(self.ids))
        bboxes = torch.as_tensor(bboxes).reshape(-1, 4)  # guard against no boxes

        target = BoxList(bboxes, img.size, mode='xyxy')
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)
        target = target.clip_to_image(remove_empty=True)
        img, target = self.transforms(img, target)

        return img, target, index

    # cytoplasm
    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        img_data = self.ids[index]['location']
        return {"width": img_data[2], "height": img_data[3],
                "file_name": self.ids[index]['file_name']}

    def get_img_in_inference(self, img_name):
        """
        haven't fixed
        Note that the key of image_pool now is "image_id" rather than "file_name"
        """
        file_name, w, h = img_name.split('~')
        # pdb.set_trace()
        try:
            # it seems that there is a wierd bug, read the image first time will caused trunct
            img = self.image_pool[file_name].read_region(
                (int(w), int(h)), 0, (
                    self.maxWS, self.maxWS)).convert("RGB")
        except:
            img = self.image_pool[file_name].read_region(
                (int(w), int(h)), 0, (
                    self.maxWS, self.maxWS)).convert("RGB")

        img = np.asarray(img, dtype=np.float32)
        return img

    def get_ground_truth(self, item):
        """
        haven't test
        for inference only, return mask in rle modec
        Args:
            item (img_id, location(x, y, w, h)) : image_id and its sliding window
        """
        ratio = 0.005 if self.split == 'test' else 0.05
        masks, bboxes, labels = self.filter_gt(item, ratio)
        if len(masks) == 0:
            pdb.set_trace()
        # if item['location'][2] == self.maxWS and item['location'][3] == self.maxWS:
        #     pdb.set_trace()
        # assert item['location'][2] == self.maxWS and item['location'][3] == self.maxWS, "Error in bbox size."
        bboxes = torch.as_tensor(bboxes).reshape(-1,
                                                 4)  # guard against no boxes
        target = BoxList(bboxes, (self.maxWS, self.maxWS),
                         mode='xyxy')
        labels = torch.tensor(labels)
        target.add_field("labels", labels)
        # masks = SegmentationMask(masks, (img_id['location'][2], img_id['location'][3]))
        # pdb.set_trace()
        rle_masks = self.annToRLE(masks, item)
        target.add_field("masks", rle_masks)
        return target

    def filter_gt(self, item, ratio=0.005):
        """
        Get mask, bbox, label in the given image within the sliding window.
        Args:
            item (img_id, location(x, y, w, h)) : image_id and its sliding window
            ratio (float)                       : the minimal area ratio need preserved (cropped bbox area >= original bbox area * ratio)
        Returns:
            mask (int list)
            bbox
            label
        """
        tic = time.time()
        cell = self.cell
        img_id = item['img_id']
        ann_ids = cell.getAnnIds(imgIds=img_id)
        anns = cell.loadAnns(ann_ids)
        location = item['location']

        mask, bbox, label = [], [], []
        for ann in anns:
            mask_ori = np.array(ann['segmentation'], np.int)
            if mask_ori.shape[0] <= 2:
                # remove the annotation whose mask coordinate number <= 2. (line or point cannot be considered as a mask)
                continue
            if not _check_if_in_crop_img(mask_ori, location):
                continue
            mask_crop = _modify_out_of_roi_masks(mask_ori, location)
            # bbox_crop (xmin, ymin, xmax, ymax)
            bbox_crop = [mask_crop[:, 0].min(), mask_crop[:, 1].min(),
                         mask_crop[:, 0].max(), mask_crop[:, 1].max()]
            # remove bbox of which the majority outside the patch
            bbox_area_ori = ann['bbox'][2] * ann['bbox'][3]
            bbox_area_crop = (bbox_crop[2] - bbox_crop[0]) * (
                    bbox_crop[3] - bbox_crop[1])
            if bbox_area_crop < bbox_area_ori * ratio:
                continue
            # polygons: a list of list of lists of numbers.
            mask_crop = [
                list(itertools.chain.from_iterable(
                    mask_crop.tolist()))]
            mask.append(mask_crop)
            bbox.append(bbox_crop)
            # class +1 because in json file, we begin at class 0 and ignore the background
            label.append(ann['category_id'] + 1)
        return mask, bbox, label

    def _remove_unsuitable_patches(self, id_list, thresh):
        """
        remove the patches with no masks in the location region
        or with only small ratio of masks ( ratio*w*h )
        Args:
            id_list [(img_id, location(x, y, w, h))] : image_id and its sliding window
            thresh                                   : bbox area need to be greater than thresh
        Returns:
            ids : id list satisfying the constrains
        """
        # remove the patches with only small ratio of masks ( ratio*w*h )
        ids = []
        for item in id_list:
            # pdb.set_trace()
            mask, bbox, label = self.filter_gt(item)
            if len(label) == 0:
                continue
            area = sum(
                [(obj[2] - obj[0]) * (obj[3] - obj[1]) for obj in
                 bbox])
            if area > thresh:
                ids.append(item)
        return ids

    def _reset_size(self, anno):
        """
        Haven't test
        May have problem anno['id']
        """
        size = int(self.maxWS * windowSizeRatio[self.split])
        anno['id']['location'][2] = size
        anno['id']['location'][3] = size
        return anno

    def annToRLE(self, masks, item):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        size = int(self.maxWS * windowSizeRatio[self.split])

        h, w = size, size
        RLES = []
        if isinstance(masks, SegmentationMask):
            for segm in masks.polygons:
                rles = maskUtils.frPyObjects(
                    [p.numpy() for p in segm], h, w
                )
                rle = maskUtils.merge(rles)
                RLES.append(rle)
        elif type(masks) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            for polygon in masks:
                rles = maskUtils.frPyObjects(polygon, h, w)
                rle = maskUtils.merge(rles)
                RLES.append(rle)
        else:
            raise NotImplementedError
        return RLES

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(
            self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Annotation File: {}\n'.format(self.annFile)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transforms.__repr__().replace(
                                         '\n',
                                         '\n' + ' ' * len(
                                             tmp)))
        return fmt_str

class PapNucleiUnlabelDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "cytoplasm",
        "nuclei",
    )

    def __init__(
            self, root, annFile, split,
            transforms=None,
            maxWindowSize=1000,
            aug_k  = 2, gen_fake = 0, gen_true = 0):
        self.root = root
        self.cv = 1
        self.aug_k =aug_k +1
        self.annFile = annFile
        assert split in ('train', 'val', 'test',
                         'gen'), "split error, (train, val, test) available"
        self.split = split
        self.transforms = transforms
        self.maxWS = maxWindowSize
        self.ids = []
        self.image_pool = {}
        tic = time.time()
        imagelist = os.listdir(root)
        imagelist = [f for f in imagelist if '.png' in f]
        for idx,  image in enumerate(imagelist):
            self.ids.append(
                {
                'id': idx,
                'file_name':image,
                'img_id': idx,
                'location': (0,0,1000,1000),
            })
        self.labels = {}

        self.category_name_to_contiguous_id = {
            c: i for i, c in enumerate(PapNucleiDataset.CLASSES)
        }
        self.contiguous_id_to_category_name = {
            v: k for k, v in
        self.category_name_to_contiguous_id.items()
        }
        cls = PapNucleiDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.contiguous_category_id_to_json_id = {v: k for k, v in
                                                  self.class_to_ind.items()}

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        print('Done (time={:0.2f}s\n) '.format(time.time() - tic))
        print('PapNuclei Dataset has %d Unlabelled patch images...' % (
            len(self.ids)))



    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.ids[index][
            'file_name'])
        img = Image.open(img_path)

        img, target = self.transforms[0](img, None)
        # copy image
        aug_imgs =[]
        for k in range(self.aug_k):
            new_img =   copy.deepcopy(img)
            new_img, target = self.transforms[1](new_img, None)
            aug_imgs.append(new_img)
        return  aug_imgs, index

    # cytoplasm
    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):
        # img_id = self.id_to_img_map[index]
        img_data = self.ids[index]['location']
        return {"width": img_data[2], "height": img_data[3],
                "file_name": self.ids[index]['file_name']}

    def _reset_size(self, anno):
        """
        Haven't test
        May have problem anno['id']
        """
        size = int(self.maxWS * windowSizeRatio[self.split])
        anno['id']['location'][2] = size
        anno['id']['location'][3] = size
        return anno

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(
            self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Annotation File: {}\n'.format(self.annFile)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transforms.__repr__().replace(
                                         '\n',
                                         '\n' + ' ' * len(
                                             tmp)))
        return fmt_str


