CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --test_path \
'path-name-to-your-weight-dir, e.g., relation_w1.0_D0.0_do_0.50.5r1.0_temp1.0_cls0.2_hint_0.0' \
--config-file "configs/pap/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
DATASETS.MODE_IN_TEST 'test' MODEL.RELATION_NMS.REG_IOU True \
MODEL.RELATION_NMS.D_LOSS 0. TEST.TTA False MT.T_ADAPT True \
TEST.VISUAL_NUM 0 TEST.GEN False