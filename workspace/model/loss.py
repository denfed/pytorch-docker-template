import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def cross_entropy_loss(output, target, class_weights: list = False):

    if class_weights is not False:
        return F.cross_entropy(output, target, weight=torch.FloatTensor(class_weights).cuda())
    else:
        return F.cross_entropy(output, target)


def binary_cross_entropy_with_logits(output, target, class_weights: list = False):

    if class_weights is not False:
        return F.binary_cross_entropy_with_logits(output, target, pos_weight=torch.FloatTensor(class_weights).cuda())
    else:
        return F.binary_cross_entropy_with_logits(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)
