# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST =  800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.VAL = ("papnuclei_val",)
_C.DATASETS.GEN = ("papnuclei_no_label",)
_C.DATASETS.MODE_IN_TEST = 'val'
# list n-fold for train/test
_C.DATASETS.TRAIN_FOLD= ()
_C.DATASETS.TEST_FOLD= ()
# _C.DATASETS.USE_GEN = False
# _C.DATASETS.NUMBER_GEN = 4000
_C.DATASETS.GEN_FAKE = 0
_C.DATASETS.GEN_TRUE = 0
_C.DATASETS.TUNE_SOURCE = False
_C.DATASETS.NO_LABEL = True
_C.DATASETS.SYN = False
_C.DATASETS.NO_LABEL_SOURCE =False
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False #True

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.MODEL.BACKBONE.OUT_CHANNELS = 256 * 4


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# _C.MODEL.MODEL.RELATION_NMS = 0.05
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
_C.MODEL.ROI_HEADS.NMS_TYPE = 'basic'
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 200


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.DO = 0.
_C.MODEL.ROI_BOX_HEAD.K_HEAD = 1.
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 3
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# add relation model
_C.MODEL.RELATION_NMS = CN()
_C.MODEL.RELATION_NMS.FIRST_N = 100
_C.MODEL.RELATION_NMS.THREAD = (0.5, 0.6, 0.7, 0.8, 0.9)
_C.MODEL.RELATION_NMS.ROI_FEAT_DIM = 1024
_C.MODEL.RELATION_NMS.APPEARANCE_FEAT_DIM = 128
_C.MODEL.RELATION_NMS.GEO_FEAT_DIM = 64
_C.MODEL.RELATION_NMS.FC_DIM = (64, 16)
_C.MODEL.RELATION_NMS.GROUP = 16
_C.MODEL.RELATION_NMS.HID_DIM = (1024, 1024, 1024)
_C.MODEL.RELATION_NMS.CLASS_AGNOSTIC = True
_C.MODEL.RELATION_NMS.USE_RELATION_NMS = False
_C.MODEL.RELATION_NMS.MERGE_METHOD = 0
_C.MODEL.RELATION_NMS.FG_THREAD = 0.05
_C.MODEL.RELATION_NMS.POS_NMS = -1.0
_C.MODEL.RELATION_NMS.CLS_WISE_RELATION = False
_C.MODEL.RELATION_NMS.MUTRELATION = False
_C.MODEL.RELATION_NMS.TAG = '_'
_C.MODEL.RELATION_NMS.CONCAT = True
_C.MODEL.RELATION_NMS.TOPK = 90
_C.MODEL.RELATION_NMS.APPEARANCE_INTER = False
_C.MODEL.RELATION_NMS.USE_IOU = False
_C.MODEL.RELATION_NMS.IOU_METHOD = 'b'
_C.MODEL.RELATION_NMS.WEIGHT = 1.
_C.MODEL.RELATION_NMS.ALPHA = 0.2
_C.MODEL.RELATION_NMS.GAMMA = 1.
_C.MODEL.RELATION_NMS.REG_IOU = False
_C.MODEL.RELATION_NMS.REG_IOU_MSK = True
_C.MODEL.RELATION_NMS.LOSS = 1.
_C.MODEL.RELATION_NMS.D_LOSS = 0.
_C.MODEL.RELATION_NMS.DO = 0.
# add mask relation
_C.MODEL.RELATION_MASK = CN()
_C.MODEL.RELATION_MASK.BINARY = False
_C.MODEL.RELATION_MASK.USE_PRE_FEATURE = False
_C.MODEL.RELATION_MASK.PRE_NORM = False
_C.MODEL.RELATION_MASK.NORM = 1
_C.MODEL.RELATION_MASK.TYPE = 'CAM'
_C.MODEL.RELATION_MASK.SAME_PREDICTOR = False
_C.MODEL.RELATION_MASK.DEEP_SUPER = True
_C.MODEL.RELATION_MASK.CAM = False
_C.MODEL.RELATION_MASK.CIAM = False
_C.MODEL.RELATION_MASK.TRAIN_CENTER_ONLY = False
_C.MODEL.RELATION_MASK.USE_RELATION = False
_C.MODEL.RELATION_MASK.PROTO = False
_C.MODEL.RELATION_MASK.ALPHA = 0.5
_C.MODEL.RELATION_MASK.CENTER_TOPK = 20
_C.MODEL.RELATION_MASK.CENTER_PER_CLASS = 8
_C.MODEL.RELATION_MASK.APPEARANCE_FEAT_DIM = 128
_C.MODEL.RELATION_MASK.GEO_FEAT_DIM = 64
_C.MODEL.RELATION_MASK.FC_DIM = (64, 16)
_C.MODEL.RELATION_MASK.GROUP = 16
_C.MODEL.RELATION_MASK.HID_DIM = (1024, 1024,)
_C.MODEL.RELATION_MASK.TOPK = 90
_C.MODEL.RELATION_MASK.EXTRACTOR_CHANNEL = 1
_C.MODEL.RELATION_MASK.FEATURE_EXTRACTOR =  'RoiAlignMaskFeatureExtractor'
_C.MODEL.RELATION_MASK.RANK = False
_C.MODEL.RELATION_MASK.CLSWIZE = False
_C.MODEL.RELATION_MASK.XY_COOR = True
_C.MODEL.RELATION_MASK.IOU_COOR = False
_C.MODEL.IOU_HEAD = CN()
_C.MODEL.IOU_HEAD.USE_IOU_MASK= False

# Mean Teacher Learning
_C.MT = CN()
_C.MT.ALPHA_RAMPUP = 0.99
_C.MT.ALPHA = 0.999 # teacher model update rate
_C.MT.LAMBDA = 1. # hyper para between supervised loss and consist
# loss
_C.MT.RAMPUP_STEP = 5000
_C.MT.RAMPDOWN_STEP = 2000
_C.MT.SEG_LOSS = 1.
_C.MT.NMS_LOSS = 1.
_C.MT.OBJ_LOSS = 1.
_C.MT.RPN_REG_LOSS = 1.
_C.MT.CLS_LOSS = 1.
_C.MT.BOX_REG_LOSS = 1.
_C.MT.NMS_LOSS_TYPE = 'weighted_bce'
_C.MT.CLS_LOSS_TYPE = 'bce'
_C.MT.REG_LOSS_TYPE = 'smooth_l1' # smooth l1
_C.MT.SEG_LOSS_TYPE = 'bce'
_C.MT.RPN_BOOST_ALPHA = 0.5
_C.MT.TEMP = 0.5
_C.MT.HINT = 0.
_C.MT.FLIP = False
_C.MT.START = 5000
_C.MT.CONSIST_ONLY_NO_LABEL = False
_C.MT.ADV = False
_C.MT.G_LOSS = 1.
_C.MT.CONFIDENCE = 0.
_C.MT.CONF_PAIR = False
_C.MT.SHARPEN = False
_C.MT.CONSIST_TEACHER = False
_C.MT.RANK_FILTER = 0.
_C.MT.IG_LOSS = 0.
_C.MT.MEAN_TARGET = True
_C.MT.CLS_NEG = True
_C.MT.AUG = False
_C.MT.ANNEAL = 0.
_C.MT.TSG_LOSS = 0.
_C.MT.SAME_DIR = False
_C.MT.HARD_NEG = False
_C.MT.START_MT = 1000
_C.MT.FG_HINT = 0.
_C.MT.T_ADAPT = False
_C.MT.CLS_BALANCE_WEIGHT = 1.
_C.MT.AUG_K = 2
_C.MT.AUG_S = 1
_C.MT.N_STEP_UNLABEL=1
# other method
_C.MT.ODKD = False
_C.MT.FFI = False
_C.MT.PLTRAIN = False

_C.SYN = CN()
_C.SYN.MT_LOSS = 0.
_C.SYN.SUP_LOSS = 1.
_C.SYN.WEIGHT_SCALE = 1.
_C.SYN.GAN_IMG = False
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 500#2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 4

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 1
_C.TEST.GEN = False
_C.TEST.TTA = False
_C.TEST.VISUAL_NUM = 0
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
