import cv2
import sys
import os
sys.path.append('..')
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from preprocess.colors import get_colors
from maskrcnn_benchmark.structures.image_list import ImageList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, Polygons
from maskrcnn_benchmark.structures.bounding_box import BoxList
from pycocotools import mask as maskUtils
import openslide as ops
import random
import itertools
from maskrcnn_benchmark.utils.miscellaneous import maskToPolygons
import pdb

def vis_bbox(bboxlist, imagelist, normalize = [102.9801, 115.9465, 122.7717] ):
    if isinstance(imagelist, ImageList):
        images = []
        for i, bbox in enumerate(bboxlist):
            if bbox.mode != 'xyxy':
                bbox = bbox.convert('xyxy')

            image = imagelist.tensors[i].numpy()
            image = np.squeeze(image)
            image = np.transpose(image,(1,2,0))
            image +=normalize

            image = image.copy()
            for j in range(bbox.bbox.shape[0]):
                box_coordinate = bbox.bbox[j].numpy().astype(np.int32)
                color = get_colors(j)
                image = cv2.rectangle(image,tuple(box_coordinate[:2]),tuple(box_coordinate[2:]), color=color.tuple(),thickness=3)
            images.append(image)
    else:
        bbox = bboxlist
        image = imagelist

        if bbox.mode != 'xyxy':
            bbox = bbox.convert('xyxy')
        image = image.copy()

        for j in range(bbox.bbox.shape[0]):
            box_coordinate = bbox.bbox[j].numpy().astype(np.int32)
            color = get_colors(j)
            image = cv2.rectangle(image, tuple(box_coordinate[:2]), tuple(box_coordinate[2:]), color=color.tuple(),
                                  thickness=3)
        images =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return images

def vis_mask(masklist, image, normalize =[102.9801, 115.9465, 122.7717] ):

    if isinstance(masklist, SegmentationMask):
        for i, polygon in enumerate(SegmentationMask):
            poly = polygon[0].polygons
            mask = np.asarray(poly[0])
            mask = np.reshape(mask, (int(len(mask) / 2), 2)).astype(
                np.int32)
            color = get_colors(i)
            image = np.asarray(image)
            cv2.polylines(np.asarray(image), [mask], 1, color.tuple(), 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        for j, mask in enumerate(masklist):
            mask = np.asarray(mask[0])
            mask = np.reshape(mask,(int(len(mask)/2),2)).astype(np.int32)
            color = get_colors(j)
            image = np.asarray(image)
            cv2.polylines(np.asarray(image),[mask], 1, color.tuple(),3)
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image






def vis_predict(dataset, gt, dt, name, show_gt =True):
    # input: list of dicts
    def convert_to_np(x):
        rle = x['segmentation']
        arr = maskUtils.decode(rle)
        return arr
    dt = map(convert_to_np, dt)
    name, w, h = name.split('~')
    # img = dataset._imgpath%name
    img = os.path.join(dataset.root, name + '.png' )
    img = ops.open_slide(img)
    img = img.read_region((int(w),int(h)), 0, (dataset.maxWS, dataset.maxWS)).convert("RGB")
    img = np.asarray(img)
    canvas = np.zeros_like(img, dtype =  np.uint8)


    for idx, d in enumerate(dt):
        if d.shape != (1000, 1000):
            import pdb;
            pdb.set_trace()
        r,g,b = get_colors(idx)
        canvas[:, :, 0] = canvas[:, :, 0] + b * d
        canvas[:, :, 1] = canvas[:, :, 1] + g * d
        canvas[:, :, 2] = canvas[:, :, 2] + r * d

    canvas2 = np.zeros_like(img, dtype =  np.uint8)
    if show_gt:
        gt = map(convert_to_np, gt)

        for idx, ins in enumerate(gt):
            if ins.shape != (1000, 1000):
                import pdb;
                pdb.set_trace()
            r, g, b = get_colors(idx )
            canvas2[:, :, 0] = canvas2[:, :, 0] + b * ins
            canvas2[:, :, 1] = canvas2[:, :, 1] + g * ins
            canvas2[:, :, 2] = canvas2[:, :, 2] + r * ins
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    add_img = cv2.addWeighted(img,0.5, canvas,0.5,0 )
    add_img2 = cv2.addWeighted(img,0.5, canvas2,0.5,0 )
    return add_img,add_img2

# def vis_mask(masklist, imagelist, normalize =[102.9801, 115.9465, 122.7717] ):
#     if isinstance(masklist, SegmentationMask):
#         for i, polygon in enumerate(SegmentationMask):
#             poly = polygon[0].convert('mask')
#
#
#     else:
#         image = imagelist
#         for j, mask in enumerate(masklist):
#             mask = np.asarray(mask[0])
#             mask = np.reshape(mask,(int(len(mask)/2),2)).astype(np.int32)
#             color = get_colors(j)
#             image = np.asarray(image)
#             cv2.polylines(np.asarray(image),[mask], 1, color.tuple(),3)
#         image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     return image



def display_instance(dataset, image_name, gt, dt ,show_masks = False, show_bbox = True, show_gt = True, alpha = 0.5, show_caption = True ):
    '''

    :param image: h,w,c
    :param dt, gt : dict
    :param title:  (optional) Figure title
    :param figsize:(optional) the size of the image
    :param color: (optional) An array or colors to use with each object
    :param captions:(optional) A list of strings to use as captions for each object
    :return:
    '''
  # input: list of dicts
    def convert_seg_to_np(x):
        rle = x['segmentation']
        arr = maskUtils.decode(rle)
        return arr

    seg_dt = list(map(convert_seg_to_np, dt))

    name, w, h = image_name.split('~')
    # img = dataset._imgpath%name
    try:
        img = os.path.join(dataset.root, name + '.png')
        img = ops.open_slide(img)
    except:
        img = os.path.join(dataset.root,'image', name + '.png')
        img = ops.open_slide(img)
    # pdb.set_trace()
    # img = img.read_region(0, 0, 0, (3152, 2760)).convert("RGB")
    img = img.read_region((int(w),int(h)), 0, (dataset.maxWS, dataset.maxWS)).convert("RGB")
    img = np.asarray(img)
    img1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img2 = img1.copy()
    # canvas = np.zeros_like(img, dtype =  np.uint8)
    # 1. draw masks
    # pdb.set_trace()
    if show_masks:
        for idx, d in enumerate(seg_dt):
            r,g,b = get_colors(idx)
            # visualize masks
            # convert list to numpy
            img1[:, :, 0] = img1[:, :, 0] * ( d == 0 ) + ( d > 0 ) * ((b * d * alpha) + img1[:, :, 0] * (1 - alpha))
            img1[:, :, 1] = img1[:, :, 1] * ( d == 0 ) + ( d > 0 ) * ((g * d * alpha) + img1[:, :, 1] * (1 - alpha))
            img1[:, :, 2] = img1[:, :, 2] * ( d == 0 ) + ( d > 0 ) * ((r * d * alpha) + img1[:, :, 2] * (1 - alpha))
    # 2. show others
    # pdb.set_trace()
    for idx, d in enumerate(seg_dt):
        r,g,b = get_colors(idx)
        # visualize masks
        contour_list = maskToPolygons(d)
        cv2.polylines(img1, contour_list, True, (b,g,r), thickness= 1)
        if show_bbox:
            bbox = dt[idx]['bbox']
            cv2.rectangle(img1, (round(bbox[0]),round( bbox[1])),
                          (round(bbox[2]), round(bbox[3])), (b,g,r), thickness= 1)
        # add information
        class_id = dt[idx]['category_id'][0]
        score = dt[idx]['score']
        if show_caption:
            x = random.randint(int(bbox[1]), round((bbox[1] + bbox[3])/2))
            caption = "{} {:.3f}".format(class_id, score)
            cv2.putText(img1, caption, (round(bbox[0]), x),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r),1,  cv2.LINE_AA )
    # canvas2 = np.zeros_like(img, dtype =  np.uint8)
    # img2 = None
    if show_gt:
        # 1. show masks
        pdb.set_trace()
        seg_gt = list(map(convert_seg_to_np, gt))
        if show_masks:
            for idx, ins in enumerate(seg_gt):
                r, g, b = get_colors(idx )
                img2[:, :, 0] =img2[:, :, 0] * ( ins == 0 ) + ( ins > 0 ) * ((b * ins * alpha) + img2[:, :, 0] * (1 - alpha))
                img2[:, :, 1] =img2[:, :, 1] * ( ins == 0 ) + ( ins > 0 ) * ((g * ins * alpha) + img2[:, :, 1] * (1 - alpha))
                img2[:, :, 2] =img2[:, :, 2] * ( ins == 0 ) + ( ins > 0 ) * ((r * ins * alpha) + img2[:, :, 2] * (1 - alpha))
        # 2. show others
        for idx, ins in enumerate(seg_gt):
            r, g, b = get_colors(idx)
            contour_list = maskToPolygons(ins)
            cv2.polylines(img2, contour_list, True, (b,g,r), thickness=2)
            if show_bbox:
                bbox = gt[idx]['bbox']
                cv2.rectangle(img2, (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3])), (b,g,r),
                              thickness=3)
                x = random.randint(int(bbox[1]), round((bbox[1] + bbox[3]) / 2))
                class_id = gt[idx]['category_id'][0]
                caption = "{}".format(class_id)
                cv2.putText(img2, caption, (round(bbox[0]), x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b,g,r),2,
                            cv2.LINE_AA)

    return img1,img2

def visualize_pseudo_label(mask, image, alpha = 0.5):
    RLES=[]
    for segm in mask.polygons:
        rles = maskUtils.frPyObjects(
            [p.numpy() for p in segm.polygons], 800, 800
        )
        rle = maskUtils.merge(rles)
        RLES.append(rle)
    for idx, cyto in enumerate(RLES):
           cyto_mask = maskUtils.decode(cyto)
           r, g, b = get_colors(int(2 * idx))
           image[:, :, 0] = image[:, :, 0] * (cyto_mask == 0) + (cyto_mask > 0) * (
                       (b * alpha) + image[:, :, 0] * (1 - alpha))
           image[:, :, 1] = image[:, :, 1] * (cyto_mask == 0) + (cyto_mask > 0) * (
                       (g * alpha) + image[:, :, 1] * (1 - alpha))
           image[:, :, 2] = image[:, :, 2] * (cyto_mask == 0) + (cyto_mask > 0) * (
                       (r * alpha) + image[:, :, 2] * (1 - alpha))
    return image

def display_instance_gen_rle(image, cyto_list, nuclei_list, alpha = 0.5):
    h, w, _ = image.shape

    for idx, cyto in enumerate(cyto_list):
        cyto_mask = maskUtils.decode(cyto)
        r, g, b = get_colors(int(2 * idx))
        image[:, :, 0] = image[:, :, 0] * (cyto_mask == 0) + (cyto_mask > 0) * (
                    (b * alpha) + image[:, :, 0] * (1 - alpha))
        image[:, :, 1] = image[:, :, 1] * (cyto_mask == 0) + (cyto_mask > 0) * (
                    (g * alpha) + image[:, :, 1] * (1 - alpha))
        image[:, :, 2] = image[:, :, 2] * (cyto_mask == 0) + (cyto_mask > 0) * (
                    (r * alpha) + image[:, :, 2] * (1 - alpha))
    for idx, cyto in enumerate(nuclei_list):
        cyto_mask = maskUtils.decode(cyto)
        r, g, b = get_colors(int(2 * idx + 1))
        image[:, :, 0] = image[:, :, 0] * (cyto_mask == 0) + (cyto_mask > 0) * (
                    (b * alpha) + image[:, :, 0] * (1 - alpha))
        image[:, :, 1] = image[:, :, 1] * (cyto_mask == 0) + (cyto_mask > 0) * (
                    (g * alpha) + image[:, :, 1] * (1 - alpha))
        image[:, :, 2] = image[:, :, 2] * (cyto_mask == 0) + (cyto_mask > 0) * (
                    (r * alpha) + image[:, :, 2] * (1 - alpha))

    return image
def display_instance_gen(image, cyto_list, nuclei_list, alpha = 0.5):
    h, w, _ = image.shape

    for idx, cyto in enumerate(cyto_list):
        cyto_mask = np.array(cyto, np.int)
        cyto_mask = [list(itertools.chain.from_iterable(cyto_mask.tolist()))]
        cyto_mask =  maskUtils.frPyObjects(cyto_mask, h, w)
        cyto_mask = maskUtils.decode(cyto_mask[0])
        r, g, b = get_colors(int(2 * idx))
        image[:, :, 0] = image[:, :, 0] * (cyto_mask == 0) + (cyto_mask > 0) * ((b* alpha) + image[:, :, 0] * (1 - alpha))
        image[:, :, 1] = image[:, :, 1] * (cyto_mask == 0) + (cyto_mask > 0) * ((g* alpha) + image[:, :, 1] * (1 - alpha))
        image[:, :, 2] = image[:, :, 2] * (cyto_mask == 0) + (cyto_mask > 0) * ((r* alpha) + image[:, :, 2] * (1 - alpha))
    for idx, cyto in enumerate(nuclei_list):
        cyto_mask = np.array(cyto, np.int)
        cyto_mask = [list(itertools.chain.from_iterable(cyto_mask.tolist()))]
        cyto_mask =  maskUtils.frPyObjects(cyto_mask, h, w)
        cyto_mask = maskUtils.decode(cyto_mask[0])
        r, g, b = get_colors(int(2 * idx + 1))
        image[:, :, 0] = image[:, :, 0] * (cyto_mask == 0) + (cyto_mask > 0) * ((b* alpha) + image[:, :, 0] * (1 - alpha))
        image[:, :, 1] = image[:, :, 1] * (cyto_mask == 0) + (cyto_mask > 0) * ((g* alpha) + image[:, :, 1] * (1 - alpha))
        image[:, :, 2] = image[:, :, 2] * (cyto_mask == 0) + (cyto_mask > 0) * ((r* alpha) + image[:, :, 2] * (1 - alpha))

    return image
    # for cyto in cyto_list:
        # cyto_mask = np.array(cyto, np.int)
        # cyto_mask = list(itertools.chain.from_iterable(cyto_mask.tolist()))
        # cyto_rle.append()






def display_instance_dt(dataset, image_name, dt, show_masks=True, show_bbox=True, alpha=0.5,
                     show_caption=True):
    '''

    :param image: h,w,c
    :param dt, gt : dict
    :param title:  (optional) Figure title
    :param figsize:(optional) the size of the image
    :param color: (optional) An array or colors to use with each object
    :param captions:(optional) A list of strings to use as captions for each object
    :return:
    '''

    # input: list of dicts

    def convert_seg_to_np(x):
        rle = x['segmentation']
        arr = maskUtils.decode(rle)
        return arr

    seg_dt = list(map(convert_seg_to_np, dt))

    name, w, h = image_name.split('~')
    # img = dataset._imgpath%name
    img = os.path.join(dataset.root, name )
    img1 = cv2.imread(img)
    # canvas = np.zeros_like(img, dtype =  np.uint8)
    # 1. draw masks
    if show_masks:
        for idx, d in enumerate(seg_dt):
            r, g, b = get_colors(idx)
            # visualize masks
            # convert list to numpy
            img1[:, :, 0] = img1[:, :, 0] * (d == 0) + (d > 0) * ((b * d * alpha) + img1[:, :, 0] * (1 - alpha))
            img1[:, :, 1] = img1[:, :, 1] * (d == 0) + (d > 0) * ((g * d * alpha) + img1[:, :, 1] * (1 - alpha))
            img1[:, :, 2] = img1[:, :, 2] * (d == 0) + (d > 0) * ((r * d * alpha) + img1[:, :, 2] * (1 - alpha))
    # 2. show others
    for idx, d in enumerate(seg_dt):
        r, g, b = get_colors(idx)
        # visualize masks
        contour_list = maskToPolygons(d)
        cv2.polylines(img1, contour_list, True, (b, g, r), thickness=1)
        if show_bbox:
            bbox = dt[idx]['bbox']
            cv2.rectangle(img1, (round(bbox[0]), round(bbox[1])), (round(bbox[2]), round(bbox[3])), (b, g, r),
                          thickness=1)
        # add information
        class_id = dt[idx]['category_id'][0]
        score = dt[idx]['score']
        if show_caption:
            x = random.randint(int(bbox[1]), round((bbox[1] + bbox[3]) / 2))
            caption = "{} {:.3f}".format(class_id, score)
            cv2.putText(img1, caption, (round(bbox[0]), x), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 2, cv2.LINE_AA)
    # canvas2 = np.zeros_like(img, dtype =  np.uint8)


    return img1



