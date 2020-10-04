# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

class TTABatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        image_1 = to_image_list(transposed_batch[0],
                                self.size_divisible)
        image_2 = to_image_list(transposed_batch[1],
                                self.size_divisible)

        targets = transposed_batch[2]
        img_ids = transposed_batch[3]
        return (image_1,image_2), targets, img_ids

class BatchCollatorWoLabel_Compared(object):
    def __init__(self, size_divisible=0, ):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        image_1 = to_image_list(transposed_batch[0],
                                self.size_divisible)
        image_2 = to_image_list(transposed_batch[1],
                                self.size_divisible)


        img_ids = transposed_batch[2]
        return image_1, image_2,  img_ids

class BatchCollatorWoLabelK(object):
    def __init__(self, size_divisible=0, aug_k = 3):
        self.size_divisible = size_divisible
        self.aug_k = aug_k
    def __call__(self, batch):
        # input
        # [[[teacher],[student],id], [[teacher],[student],id],... ]
        # return [imglist, imglist,...], id

        transposed_batch = list(zip(*batch))
        _batch =transposed_batch[0]
        _batch_list = list(zip(*_batch))
        _batch_tensor = []
        for i in range(self.aug_k):
            _image = to_image_list(_batch_list[i],self.size_divisible)
            _batch_tensor.append(_image)

        idx = transposed_batch[1]

        return _batch_tensor, idx