import logging
from .pap_eval import do_pap_evaluation

def pap_evaluation(dataset, predictions, output_folder, box_only, visual_num,**_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("pap evaluation doesn't support box_only, ignored.")
    logger.info("performing pap evaluation, ignored iou_types.")
    return do_pap_evaluation(
        dataset=dataset,
        iou_types=_['iou_types'],
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        visual_num=visual_num

    )

