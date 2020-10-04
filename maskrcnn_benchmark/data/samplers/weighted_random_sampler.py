from torch.utils.data.sampler import Sampler
import torch
from torch._six import int_classes as _int_classes


class WeightedRandomSubSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, data_source, weights, replacement=True):

        if not isinstance(replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(replacement))

        self.gen_fake = data_source.gen_fake
        self.gen_true = data_source.gen_true
        self.source_data = data_source.start_id
        self.weights = torch.tensor(weights, dtype=torch.double)
        self.num_samples = len(data_source)
        self.replacement = replacement

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples,
                                      self.replacement).tolist())

    def __len__(self):
        return self.num_samples
