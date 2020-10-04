import logging
import tempfile
import os
import torch
from collections import OrderedDict,defaultdict
import time
import copy
import cv2
import numpy as np
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils.visual import  display_instance
from pycocotools import mask as maskUtils
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import numbers
import pdb

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def do_pap_evaluation(
    dataset,
    predictions,
    output_folder,
    iou_types,
    logger,
    visual_num = 0
):
    logger = logger
    logger.info("Preparing results for Pap format")
    pap_gts, pap_results= prepare_for_pap_segmentation(predictions, dataset)
    results = PapResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")
            res = evaluate_predictions_on_pap(
                pap_gts, pap_results, file_path, iou_type
            )
            results.update(res)
    logger.info(results)
    if output_folder:
        if visual_num > 0:
            visualize_results(dataset, pap_results, pap_gts, output_folder, visual_num)
        torch.save(results, os.path.join(output_folder, "pap_results.pth"))
    return results, pap_results

def visualize_results(dataset,pap_results, pap_gts,output_folder, visual_num, show_bbox = False, show_caption= False):
    # pdb.set_trace()
    vis_save_path = os.path.join(output_folder, 'visual')
    mkdir(vis_save_path)
    visual_imgs_dt = defaultdict(list)
    visual_imgs_gt = defaultdict(list)
    for _, pap_result in enumerate(pap_results):
        original_id = pap_result['image_id']
        visual_imgs_dt[original_id['file_name'] + '~%d~%d' % (
            original_id['location'][0], original_id['location'][1])].append(pap_result)
    if pap_gts is not None:
        for _, pap_gt in enumerate(pap_gts):
            original_id = pap_gt['image_id']
            visual_imgs_gt[original_id['file_name'] + '~%d~%d' % (
                original_id['location'][0], original_id['location'][1])].append(pap_gt)
    show_gt  = True if pap_gts is not None else False

    for i, (k, vis_gt) in enumerate(visual_imgs_gt.items()):

            vis_dt = visual_imgs_dt[k]
            # pdb.set_trace()
            vis_img, gts = display_instance(dataset, k, vis_gt,
                                            vis_dt, show_gt = show_gt,
                                            show_bbox= show_bbox, show_caption= show_caption)
            # vis_img,gts = vis_predict(dataset, vis_gt, vis_dt, k, True)
            save_path = os.path.join(vis_save_path, k + '.jpg')
            cv2.imwrite(save_path, vis_img)
            if show_gt :
                cv2.imwrite(os.path.join(vis_save_path, k + 'gt.jpg'), gts)

def prepare_for_pap_segmentation(predictions, dataset):
    masker = Masker(threshold=0.5, padding=1)
    pap_results = []
    pap_gt = []
    for idx, (image_id, prediction) in enumerate(predictions.items()):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue
        # TODO replace with get_img_info?
        image_width = dataset.maxWS
        image_height =dataset.maxWS
        target = dataset.get_ground_truth(original_id)
        gt_labels = target.get_field("labels").tolist()
        gt_masks = target.get_field("masks")
        gt_boxes = target.bbox.tolist()
        gt_mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in gt_labels]
        pap_gt.extend(
            [{
                "image_id": original_id,
                "category_id": gt_mapped_labels[k],
                "segmentation": rle,
                "bbox": gt_boxes[k]

            }for k, rle in enumerate(gt_masks)]
        )
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # pdb.set_trace()
        # prepare for bbox
        # prediction = prediction.convert("xywh")
        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        # blank_rle = mask_util.encode(np.array(blank_mask[0, :, :], order="F"))
        rles = [
            maskUtils.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")
        # pdb.set_trace()
        try:
            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        except:
            # pdb.set_trace()
            print('bug')
        pap_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "segmentation": rle,
                    "score": scores[k],
                    "bbox": boxes[k]

                }
                for k, rle in enumerate(rles)
            ]
        )

    return  pap_gt, pap_results


class PapResults(object):
    METRICS = {
        "segm": ["AJI", "F1", 'DSC', 'TPRP', 'FNRo', 'FDRo', 'mAP',
                 'AP50', 'AP75','AP85'],
    }

    def __init__(self, *iou_types):
        allowed_types = ("segm")
        # pdb.set_trace()
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in PapResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, pap_eval):
        if pap_eval is None:
            return
        s = pap_eval.stats
        # pdb.set_trace()

        iou_type = pap_eval.params.iouType
        res = self.results[iou_type]
        metrics = PapResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            for k, v in s[metric].items():
                # temp code, turn value in array
                if isinstance(v, numbers.Number):
                    s[metric][k] = v
                elif isinstance(v, list):
                    s[metric][k] = v
                else:
                    s[metric][k] = v[0,0]

            res[metric] = s[metric]

    def __repr__(self):
        return repr(self.results)



def evaluate_predictions_on_pap(
    pap_gt, pap_results, json_result_file, iou_type="segm"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(pap_results, f)

    _eval = Papeval(pap_gt, pap_results, iou_type)
    _eval.evaluate()
    _eval.accumulate()
    _eval.summarize()


    return _eval

def convert_to_xylist(polys):
    new_poly = []
    for poly in polys:
        poly2 = [[i,j] for i,j in zip(poly[::2], poly[1::2])]
        new_poly.append(poly2)
    return new_poly

class Papeval:
    def __init__(self, gts, dts, iou_type,):
        super(Papeval, self).__init__()
        self.iou_type = iou_type
        self.PapGt   =  gts             # ground truth
        self.PapDt   =  dts          # detections
        self.params = {}  # evaluation parameters
        self.params = Params(iouType=iou_type)  # parameters
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id']['file_name'] + '_%d_%d'%(gt['image_id']['location'][0],gt['image_id']['location'][1]) , gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id']['file_name'] + '_%d_%d'%(dt['image_id']['location'][0],dt['image_id']['location'][1])  , dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
        self.createIndex()

        self.params.imgIds = sorted(copy.deepcopy(self.img_id))
        self.params.catIds = sorted(self.catToImgGTs.keys())

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()

        print('Running per image evaluation...')
        p = self.params
        # # add backward compatibility if useSegm is specified in params
        # if not p.useSegm is None:
        #     p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
        #     print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        # print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p


        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]
        self.ious = {(imgId, catId): self.computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}
        # compute self
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        # todo parallel evaluate
        self.evalImgs = [evaluateImg(imgId, catId, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]

        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))



    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return [],[],[]
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [0] * len(g)
        if len(g)!=0:
            gt_area  = maskUtils.area(g)
        else:
            gt_area = None
            # ious = np.zeros((1,len(gt)), dtype=np.double)
            # intersection = np.zeros((1,len(gt)), dtype=np.double)
            # union = np.asarray(map(maskUtils.area, g),dtype= np.double)
            # union = union[np.newaxis,:]
        # pdb.set_trace()
        iouIntUni = maskUtils.iouIntUni(d,g,iscrowd)

        if len(d)== 0 or len(g)== 0:

            ious = []
            dsc = []
            tpp = []
            intersection = []
            if len(d)>0:
                merge_area = copy.deepcopy(d)
            if len(g)>0:
                merge_area = copy.deepcopy(g)
            merge_area = maskUtils.merge( merge_area, intersect=False )
            union =[maskUtils.area(merge_area)]
        else:
            ious, intersection, union = iouIntUni[0],iouIntUni[1],iouIntUni[2]
            intersection[ious <= 0] = 0

            dsc =   2 * intersection/(union + intersection + 1e-10) #[2 * i/(u + i) for i,u in zip(intersection, union)]
            if dsc.max()>1:
                pdb.set_trace()
            # if (intersection/gt_area).max

        return ious, intersection, union, gt_area, dsc


    def compute_F1(self, gt_area, iou, intersection, UseIOU = True):
        TP = 0
        FP = 0
        FN = 0

        PR_thread = [i for i in np.linspace(0.2, 0.9, 28)]
        TPLIST = [0 for i in range(28)]
        FPLIST = [0 for i in range(28)]
        # PLIST = [0 for i in range(28)]
        # RLIST =[0 for i in range(28)]
        F1LIST = [0 for i in range(28)]
        iou_copy  =  copy.deepcopy(iou)
        gt_num =iou.shape[1]
        # gt_map_seg = np.zeros((gt_num,2)) # 0 for map idx, 1 for iou value
        # pdb.set_trace()
        iou_list = iou_copy.T.tolist()
        inter_index_list = list(map(lambda x: x.index(max(x)) if max(x)>0 else -1, iou_list))
        inter_value_list  =list(map(lambda x: max(x), iou_list))
        # gt_map_seg[:,0] = np.asarray(inter_index_list,)
        # gt_map_seg[:,1] = np.asarray(inter_value_list)
        inter_index_set = set(inter_index_list)
        inter_index_set.discard(-1)

        while (len(inter_index_list)  - inter_value_list.count(0)) != len(inter_index_set):
            # find the duplicate index and set another segmented result to ground truth base on criterion

            duplicate_indices = []

            for v in inter_index_set:
                if inter_index_list.count(v) > 1:
                    duplicate_indices = [i for i, x in enumerate(inter_index_list) if x == v]
                    break
            # then get the max iou index in duplicate indices
            if len(duplicate_indices)==0:
                # pdb.set_trace()
                print('bug')
            iou_for_duplicate = list(map(inter_value_list.__getitem__, duplicate_indices))
            # delete the index with max iou
            del duplicate_indices[(iou_for_duplicate.index(max(iou_for_duplicate)))]
            # search for best iou match again
            for i in duplicate_indices:
                iou_list[i][v] = 0
                inter_index_list[i] = iou_list[i].index(max(iou_list[i])) if max(iou_list[i])>0 else -1
                inter_value_list[i] = max(iou_list[i])
            inter_index_set = set(inter_index_list)
            inter_index_set.discard(-1)
        # so far, for each gt, we map a seg result.
        # Then computer ratio = intersect/union
        for gtidx, segidx in enumerate( inter_index_list):
            if segidx != -1:

                if UseIOU:
                    value = iou_list[gtidx][segidx]
                else:
                    _intersection = intersection[gtidx, segidx]
                    value = _intersection/gt_area[gtidx]

                if value > 0.5:
                    TP += 1


                # LIST
                for k, thread in enumerate(PR_thread):
                    if value>thread:
                        TPLIST[k] += 1

        # add unmatched segmented result to FP
        seg_num = iou.shape[0]
        FNLIST = [len(gt_area) - f for f in TPLIST]

        FPLIST = [iou.shape[0] - t for t in TPLIST]
        # FPLIST = [f + (iou.shape[0] - t ) for t,f in zip(TPLIST,FPLIST)]
        # pdb.set_trace()
        PLIST = [t/(t+f) for t,f in zip(TPLIST,FPLIST)]
        RLIST = [t /(t+f) for t,f in zip(TPLIST,FNLIST)]
        itm = 0
        for p,r in zip(PLIST,RLIST):
            if (p+r ) ==0:
                F1LIST[itm] = 0
            else:
                F1LIST[itm] =2 * p * r/(p + r)
            itm+=1

        FN = len(gt_area) - TP
        FP = (iou.shape[0] - TP)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        if (recall + precision) == 0:
            F1 = 0
        else:
            F1 = 2 * precision * recall/(precision + recall)
        return PLIST,RLIST, F1, precision, recall,



    def caclulateMetrics(self, ious, ints, areas, dsc, gt):

        dc_thread = 0.7
        # p = self.params
        try:
            D,G = ious.shape
        except:
            G = len(gt)
            D = 0
        if D == 0:
            gtdsc = np.zeros((G))
            gttpr = np.zeros((G))
            mdsc = 0
            mtpr = 0
            FNR = 0
            FDR = 0
            alldsc = gtdsc[gtdsc>dc_thread]
            alltpr = gttpr[gtdsc>dc_thread]
        else:

            allTPR = ints/areas
            if allTPR.max()>1:

                pdb.set_trace()
            gtmid = - np.ones(( G))
            gtdsc = np.zeros((G))
            gttpr = np.zeros((G))
            # AJI
            # DSC = np.zeros((1, 1))
            dsc_shape = dsc.shape

            while dsc.max()>dc_thread:
                maxind  = np.argmax(dsc)
                # [detect, gt]
                ind = np.unravel_index(maxind, dsc_shape)
                maxdsc = dsc[ind]
                gtmid[ind[1]] = ind[0]
                gtdsc[ind[1]] = maxdsc
                gttpr[ind[1]] = allTPR[ind]
                dsc[ind[0]] = 0
                dsc[:,ind[1]] = 0
            # pdb.set_trace()
            alldsc = gtdsc[gtdsc>dc_thread]
            # mdsc = np.mean(alldsc)
            alltpr = gttpr[gtdsc>dc_thread]
            # mtpr = np.mean(alltpr)
            FNR = (G - np.count_nonzero(gtdsc))
            # mFNR = FNR/G
            FDR = (D - np.count_nonzero(gtdsc))
            # mFDR = FDR/D
        return alldsc,alltpr,FNR,FDR


    def cal_MAP(self, dt, gt, ious, thr = np.linspace(.5, 0.95,
                     np.round((0.95 - .5) / .05) + 1, endpoint=True)):
        T = len(thr)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))

        if not len(ious) == 0:
            for tind, t in enumerate(thr):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 :
                            continue
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtm[tind, dind] = m + 1
                    gtm[tind, m] = dind + 1

        return dtm, gtm


    def evaluateImg(self, imgId, catId,  maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        # todo: compute bbox recall for nuclei and cytoplasm.
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        # prepare for F1 SCORE

        #
        for g in gt:
        #     if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
        #         g['_ignore'] = 1
        #     else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        # gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        # iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious

        ious = self.ious[imgId, catId][0] #if len(self.ious[imgId, catId][0]) > 0 else self.ious[imgId, catId][0]
        intersection = self.ious[imgId, catId][1]# if len(self.ious[imgId, catId][1]) > 0 else self.ious[imgId, catId][1]
        union = self.ious[imgId, catId][2]# if len(self.ious[imgId, catId][2]) > 0 else self.ious[imgId, catId][2]
        area = self.ious[imgId, catId][3]
        dsc = self.ious[imgId, catId][4]
        # if intersection.max()>1:
        #     pdb.set_trace()
        # calculate f1 SCORE
        if len(gt) != 0 and len(dt) != 0:
            PLIST, RLIST, F1, precision, recall = self.compute_F1(area, ious, intersection, p.UseIOU)
        elif len(gt) ==0 and len(dt)>0:
            F1,precision, recall = 1, 0, 1
            PLIST,RLIST = [0 for i in range(28)], [1 for i in range(28) ]
        elif len(gt)>0 and len(dt) == 0:
            F1,precision,recall = 0, 1, 0
            PLIST, RLIST = [1 for i in range(28)], [0 for i in range(28)]
        else:
            F1, precision, recall = 1,1,1
            PLIST, RLIST = [1 for i in range(28)], [1 for i in range(28)]

        # calculate metrics
        mdsc, mtpr, FNR, FDR = self.caclulateMetrics(ious,
                                                     intersection,
                                                     area,
                                                     dsc,gt)
        # pdb.set_trace()
        ap_dtm, ap_gtm = self.cal_MAP(dt,gt,ious)


        # calculate AJI
        dc_thread = 0.6
        iouThrsAJI = [0.5]
        T = len(iouThrsAJI)
        G = len(gt)
        D = len(dt)
        gtm  = - np.ones((T,G))
        dtm  = - np.ones((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        # AJI
        AJI = np.zeros((T,1))
        # IOU = np.zeros((T,1))
        INTERSECTION = np.zeros((T,1))
        UNION = np.zeros((T,1))

        DSC = np.zeros((G,1))
        if not len(ious)==0:

            for tind, t in enumerate(iouThrsAJI):
                for gind, g in enumerate(gt):
                    iou = min([t, 1 - 1e-10])
                    _intersection = 0
                    _union = 0
                    m = -1
                    _dsc = 0
                    for dind, d in enumerate(dt):
                        # if the dt already matched, continue
                        if dtm[tind,dind]>0:
                            continue
                        # continue to next dt unless better match made
                        if ious[dind,gind]< iou:
                            continue
                        # if match successful and best so far, store it
                        iou = ious[dind, gind]
                        _union = union[dind, gind]
                        _intersection = intersection[dind, gind]
                        # _dsc = dsc[dind, gind]
                        m = dind
                    if m==-1:
                        continue

                    dtm[tind, m] = g['image_id']['id']
                    gtm[tind, gind] = dt[m]['image_id']['id']
                    INTERSECTION[tind, 0] = INTERSECTION[tind, 0] + _intersection
                    UNION[tind, 0] = UNION[tind, 0] + _union
                    DSC[gind, 0] = _dsc
                # add missing gt and dt

                miss_gt = np.argwhere(gtm == -1)
                miss_dt = np.argwhere(dtm == -1)
                miss_gt = [gt[gt_index[1]]['segmentation'] for gt_index in miss_gt]
                miss_dt = [dt[dt_index[1]]['segmentation'] for dt_index in miss_dt]
                miss_gt = [maskUtils.area(f) for f in miss_gt]
                miss_dt = [maskUtils.area(f) for f in miss_dt]
                UNION[tind, 0] = UNION[tind, 0] + sum(miss_dt) + sum(miss_gt)

            AJI = np.divide(INTERSECTION, UNION)
            gtIg = np.array([ 0 for g in gt])
            # DSC > 0.7
            # good_ins = DSC[np.where(DSC > dc_thread)]
            # if len(np.nonzero(good_ins)[0]) == 0:
            #     DSC_GOOD = 0
            # else:
            #     DSC_GOOD = np.mean(good_ins)
            # DSC_GOOD = np.asarray(DSC_GOOD).reshape(1,1)
            # if catId =='nuclei':
            #     pdb.set_trace()
            # fno
            # DSC[DSC> dc_thread] = 1
            # DSC[DSC<=dc_thread] = 0
            # FNO =  (G - np.sum(DSC))/G

        else:
            AJI = np.zeros((T,1))
            # DSC_GOOD = np.zeros((T,1))
            # FNO = 1
        # # set unmatched detections outside of area range to ignore
        # a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        # dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category


        return {
                'image_id':     imgId,
                'category_id':  catId,
                'maxDet':       maxDet,
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'map_dtMatches': ap_dtm,
                'map_gtMatches': ap_gtm,
                'dtScores':     [d['score'] for d in dt],
                'AJI': AJI,
                'F1': F1,
                'DSC': mdsc,
                'TPRp': mtpr,
                'FNRo': FNR,
                'FDR': FDR,
                'num_G': G,
                'num_D': D,
                'gtIg':gtIg
                # 'precision': precision,
                # 'recall': recall,
                # 'DSC': DSC_GOOD,
                # 'FNO': FNO,
                # 'PLIST': PLIST,
                # 'RLIST': RLIST
            }

    def createIndex(self):
        # create index
        print('creating index...')
        # anns, cats, imgs = {}, {}, {}
        img_id = []
        # self.cat_name =
        catToImgGTs, catToImgDTs, = defaultdict(list),defaultdict(list)
        for gt in self.PapGt:
            catToImgGTs[gt['category_id']].append(gt['image_id'])
            img_id.append(gt['image_id']['file_name'] + '_%d_%d'%(gt['image_id']['location'][0],gt['image_id']['location'][1]))

        for dt in self.PapDt:
            catToImgDTs[dt['category_id']].append(dt['image_id'])

        print('index created!')

        # create class members
        # self.imgToAnns = imgToAnns
        self.catToImgGTs = catToImgGTs
        self.catToImgDTs = catToImgDTs
        self.img_id = img_id
        # self.imgs = imgs
        # self.cats = cats

    def accumulate(self):
        p = self.params
        catIds = self.params.catIds if self.params.useCats else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        precision = -np.ones((T, R, K, ))  # -1 for the precision of absent categories
        recall = -np.ones((T, K,))
        scores = -np.ones((T, R, K,))
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        # pdb.set_trace()
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 *  I0
            # Na =  I0
            E = [self.evalImgs[Nk + i] for i in i_list]
            E = [e for e in E if not e is None]
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind='mergesort')
            dtScoresSorted = dtScores[inds]

            dtm = np.concatenate(
                [e['map_dtMatches'] for e in E],
                axis=1)[:, inds]
            gtIg = np.concatenate([e['gtIg'] for e in E])
            # dtIg = np.concatenate(
            #     [e['dtIgnore'] for e in E],
            #     axis=1)[:, inds]
            # gtIg = np.concatenate([e['gtIgnore'] for e in E])
            npig = np.count_nonzero(gtIg == 0)
            # if npig == 0:
            #     continue
            tps = dtm>0
            fps = np.logical_not(dtm)

            tp_sum = np.cumsum(tps, axis=1).astype(
                dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(
                dtype=np.float)
            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp + tp + np.spacing(1))
                q = np.zeros((R,))
                ss = np.zeros((R,))

                if nd:
                    recall[t, k ] = rc[-1]
                else:
                    recall[t, k ] = 0

                # numpy is slow without cython optimization for accessing elements
                # use python array gets significant speed improvement
                pr = pr.tolist();q = q.tolist()

                for i in range(nd - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                inds = np.searchsorted(rc, p.recThrs,
                                       side='left')
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                        ss[ri] = dtScoresSorted[pi]
                except:
                    pass
                precision[t, :, k] = np.array(q)
                scores[t, :, k] = np.array(ss)
        # pdb.set_trace()
        self.eval = {
            'params': p,
            'counts': [T, R, K,],
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }


    def summarize(self):
        # have not use dice yet
        # 'mDSC': mdsc,
        # 'TPRp': mtpr,
        # 'FNRo': FNR,
        # 'FDR': FDR
        stats = {}
        AJI = {}
        DSC = {}
        # FNO = {}
        F1_score = {}
        TPR = {}
        FDR = {}
        FNR = {}
        mAP = {}
        AP50= {}
        AP75= {}
        AP85 = {}
        def _summarize(catId = None, iouThr=None, ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} ] = {:0.3f}'
            titleStr = 'Average Precision'
            typeStr = '(AP)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0],
                                              p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            if catId is not None:
                s = s[:, :, catId]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            # print(iStr.format(titleStr, typeStr, iouStr, mean_s))
            return mean_s

        # P = {}
        # R = {}
        # RLIST = {}
        # PLIST = {}
        for catId, cat in enumerate(self._paramsEval.catIds):
            aji = np.zeros((len(self._paramsEval.iouThrs),1))
            _count = 0
            F1 = 0
            # precision = 0
            # plist = [0 for i in range(28)]
            # rlist = [0 for i in range(28)]
            # recall = 0
            dsc = []
            fno = 0
            num_G = 0
            num_D = 0
            tpr = []
            fnr = []
            fdr = []
            # pdb.set_trace()
            for result in self.evalImgs:

                if result is None:
                    # skip len(gt)=0 & len(dt)
                    continue
                if result['category_id'] == cat:
                    aji = aji + result['AJI']
                    F1  = F1 + result['F1']
                    # precision = precision + result['precision']
                    # recall = recall + result['recall']
                    # for t in range(28):
                    #     plist[t] = plist[t] + result['PLIST'][t]
                    #     rlist[t] = rlist[t] + result['RLIST'][t]
                    # dsc.append(result['DSC'])
                    dsc.extend(list( result['DSC']))
                    # fno = fno + result['FNO']
                    num_D  =num_D + result["num_D"]
                    num_G = num_G +result["num_G"]
                    fdr.append(result['FDR'])
                    fnr.append(result['FNRo'])

                    tpr.extend(list(result['TPRp']))
                    # if max(tpr)>1:
                    #     pdb.set_trace()
                    _count+= 1
            aji = np.divide(aji, _count)
            F1 = F1/_count
            # precision = precision/_count
            # recall = recall/ _count
            fdr = sum(fdr)/num_D
            fnr = sum(fnr)/num_G
            # pdb.set_trace()
            tpr = sum(tpr)/(len(tpr)+1e-10)
            dsc = sum(dsc)/(len(dsc)+1e-10)
            # plist = [f/_count for f in plist]
            # rlist = [f/_count for f in rlist]
            # dsc = np.asarray(dsc)
            # dsc = np.mean(dsc[np.where(dsc>0)])
            # pdb.set_trace()

            # fno = fno/_count
            AJI[cat] = aji
            F1_score[cat] = F1
            DSC[cat] = dsc
            TPR[cat] = tpr
            FDR[cat] = fdr
            FNR[cat] = fnr
            mAP[cat] = _summarize(catId)
            AP50[cat] = _summarize(catId, 0.5)
            AP85[cat] = _summarize(catId, 0.85)
            AP75[cat] = _summarize(catId, 0.75)
            # P[cat] = precision
            # PLIST[cat] = plist
            # RLIST[cat] = rlist
            # R[cat] = recall
            # DSC[cat] = dsc
            # FNO[cat] = fno
        mAP['all'] = _summarize()
        AP50['all']= _summarize( iouThr=.5)
        AP75['all'] = _summarize( iouThr=.75)
        AP85['all'] = _summarize(iouThr=.85)
        # pdb.set_trace()

        stats['AJI'] = AJI
        stats['F1'] = F1_score
        # stats['precision'] = P
        # stats['recall'] = R
        stats['DSC'] = DSC
        stats['TPRP'] = TPR
        stats['FNRo'] = FNR
        stats['FDRo'] = FDR
        # stats['FNO'] = FNO
        # stats['PLIST'] = PLIST
        # stats['RLIST'] = RLIST
        # pdb.set_trace()
        stats['mAP'] = mAP
        stats['AP50'] = AP50
        stats['AP75'] = AP75
        stats['AP85'] = AP85



        self.stats = stats


class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        # self.iouThrs = [0.5] #np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        # self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [200]
        self.areaRng = [[0 ** 2, 1e5 ** 2]]
        # self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1
        self.UseIOU = True
        self.iouThrs = np.linspace(.5, 0.95,
                                   np.round((0.95 - .5) / .05) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0, 1.00,
                                   np.round((1.00 - .0) / .01) + 1,
                                   endpoint=True)

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()

        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None