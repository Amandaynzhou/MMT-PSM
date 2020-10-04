import torch
from torch.nn import functional as F
def entropy_loss(p,q, eps = 10-6):
    '''
    h = p * log(q) where p is prob from teacher
    :param p:
    :param q:
    :return:
    '''
    # pdb.set_trace()
    p =F.softmax(p,dim=1)
    q = F.log_softmax(q,dim=1)
    result = p * q
    result = -torch.mean(result)
    return result



def balanced_BCE(scores, labels, nms_score):
    # weight  = torch.ones_like(scores,).to(labels.device)
    # weight[labels==0] = 0.2
    class_weight =torch.FloatTensor([1, 2]).to(labels.device)

    weight = class_weight[labels.long()]
    weight = weight *  weight.numel()/weight.sum()

    nms_loss = F.binary_cross_entropy(scores, nms_score,
                                      weight)
    return nms_loss

def combined_BCE(scores, labels, nms_score):

    loss1 = F.binary_cross_entropy(scores, labels)
    loss2 = MSEloss(scores,nms_score)
    loss = 0.5*(loss1 + loss2)
    return loss

def smooth_l1(scores,  nms_score ,beta = 0.1):

    n = torch.abs( scores - nms_score)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    return loss.mean()


# def soft_BCE(scores, labels, nms_score):
#     loss = F.nll_loss(torch.log(scores), labels)
def weighted_BCE(scores, labels, nms_score):
    # balanced






    nms_score[labels == 0] = 1 - nms_score[labels == 0]
    # weight avg p/n
    negative_weight = (1 - labels).sum() / nms_score[
        labels == 0].sum()
    positive_weight = labels.sum() / nms_score[labels == 1].sum()
    nms_score[labels == 1] = nms_score[labels == 1] * positive_weight
    nms_score[labels == 0] = nms_score[labels == 0] * \
                             negative_weight
    labels = labels.to(scores.device)
    labels = labels.type(torch.cuda.FloatTensor)
    nms_loss = F.binary_cross_entropy(scores, labels,
                                      nms_score)
    return nms_loss

def BCE(scores, nms_score):

    loss = F.binary_cross_entropy_with_logits(scores, nms_score)
    return loss

def KLloss(score, nms_score):
    score = F.log_softmax(score,dim=1)
    nms_score = F.softmax(nms_score,dim=1)
    # nms_score = F.softmax()
    loss = F.kl_div(score, nms_score)
    return loss
# torch.nn.KLDivLoss
def MSEloss(score, target):

    loss = F.mse_loss(score,target)
    return loss


def regression_loss(type, predict, label):
    if type == 'mse':
        return MSEloss(predict,label)
    elif type == 'kl':
        return KLloss(predict,label)
    else:
        print('error, not implement %s!!!'%type)

def reg_flip_loss(predict, label):
    label[:, 0] = -label[:, 0]
    return MSEloss(predict,label)

def flip_l2_dist(predict, label):
    new_label = label.detach()
    new_label[:,0] = -new_label[:,0]
    dist =  F.mse_loss( predict, new_label,reduction='none')
    return dist


def CE_loss(predict, label):
    loss = F.binary_cross_entropy_with_logits(predict,label)
    return loss
# def CE_loss




def classification_loss(type, predict, label, weight = None):
    if type == 'bce':
        return BCE(predict,label)
    elif type =='kl':
        return KLloss(predict,weight)
    elif type == 'wbce':
        return weighted_BCE(predict,label,weight)
    elif type == 'mse':
        return MSEloss(predict, weight)
    elif type == 'entropy':
        return entropy_loss(predict,weight)
    elif type == 'ce':
        return CE_loss(predict, label)
    # todo wce and ce
    else:
        print('error, not implement %s !!!'%type)