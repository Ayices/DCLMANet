import os
import torch
import numpy as np
import medpy.metric.binary as medpyMetrics
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import medpy
import torch.nn.functional as F
from torch.nn import init
from torch.nn.functional import cross_entropy

#灵敏度、PPV、HD、ASD

class DiceLossSigmoid(nn.Module):
    def __init__(self):
        super(DiceLossSigmoid, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        pre = predict.view(predict.size(0), predict.size(1), -1)
        tar = target.view(target.size(0), target.size(1), -1)

        score = 1 - 2 * (pre * tar + (1 - pre) * (1 - tar) ).sum(-1) / (
                    (pre + tar) + ((1 - pre) + (1 - tar))).sum(-1)
        return score.mean()

def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def Dice_result(output, target, eps=1e-5):

    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return num/den

def softmax_dice_result(output, target):
    WT_result = np.zeros_like(output)
    TC_result = np.zeros_like(output)
    ET_result = np.zeros_like(output)

    WT_result[output >= 1] = 1
    TC_result[output == 1] = 1
    TC_result[output == 4] = 1
    ET_result[output == 4] = 1

    WT_label = np.zeros_like(output)
    TC_label = np.zeros_like(output)
    ET_label = np.zeros_like(output)

    WT_label[target >= 1] = 1
    TC_label[target == 1] = 1
    TC_label[target == 4] = 1
    ET_label[target == 4] = 1

    WT_Dice = Dice_result(WT_result, WT_label)
    TC_Dice = Dice_result(TC_result, TC_label)
    ET_Dice = Dice_result(ET_result, ET_label)

    return WT_Dice, TC_Dice, ET_Dice



def cross_entro(output, target):
    target[target == 1] = 3
    target[target == 2] = 1
    target[target == 4] = 2

    target = target.squeeze(dim = 1)

    weight = torch.tensor([0.01, 0.33, 0.33, 0.33]).cuda()

    loss = F.cross_entropy(output, target, weight=weight)

    return loss

def WT_dice(output, target):

    loss = Dice(output, target)
    return loss

def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 2).float())
    loss2 = Dice(output[:, 2, ...], (target == 4).float())
    loss3 = Dice(output[:, 3, ...], (target == 1).float())

    return (0.01 * loss0) + (0.33*loss1) + (0.33*loss2) + (0.33*loss3), 1-loss0.data, 1-loss1.data, 1-loss2.data, 1-loss3.data

def sigmoid_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss1 = Dice(output[:, 0, ...], (target == 4).float())

    return loss1

def seg_metrics(pre, target, ForSaver=True):
    if pre is None:
        return [{
            'SEN': 0, 'PPV': 0,
            'DSC': 0, 'JC': 0,
            'HD': -1000, 'ASD': -1000, 'ASSD': -1000
        }]
    if pre.size(1) != 1:
        tpre = torch.argmax(pre, dim=1, keepdim=True)
        tpre = F.one_hot(tpre.long(), pre.size(1))
        tpre = torch.Tensor.permute(tpre, [0, 5, 2, 3, 4, 1])
        tpre = tpre[:, 1:, :, :, :, 0]
        tpre = tpre.detach().cpu().numpy()

        target = F.one_hot(target.long(), pre.size(1))
        target = torch.Tensor.permute(target, [0, 5, 2, 3, 4, 1])
        target = target[:, 1:, :, :, :, 0]
        ttarget = target.detach().cpu().numpy()
    else:
        tpre = (pre > 0.5).long()
        tpre = tpre.detach().cpu().numpy()
        ttarget = target.detach().cpu().numpy()

    targetmean = np.sum(np.sum(np.sum(ttarget, 1), 1), 1)
    if len(np.where(targetmean != 0)[0]) != 0:
        metrics_list = []
        for index in np.where(targetmean != 0)[0]:
            tdice = medpy.metric.binary.dc(tpre[index], ttarget[index])
            tsen = medpy.metric.binary.sensitivity(tpre[index], ttarget[index])
            tjc = medpy.metric.binary.jc(tpre[index], ttarget[index])
            tprecision = medpy.metric.binary.precision(tpre[index], ttarget[index])
            if np.sum(tpre[index]) != 0:
                thd = medpy.metric.binary.hd(tpre[index], ttarget[index])
                tasd = medpy.metric.binary.asd(tpre[index], ttarget[index])
                tassd = medpy.metric.binary.assd(tpre[index], ttarget[index])
            else:
                ttpre = np.zeros_like(ttarget[index])
                ttpre[:, pre.size(2) // 2, pre.size(3) // 2] = 1
                thd = medpy.metric.binary.hd(ttpre, ttarget[index])
                tasd = medpy.metric.binary.asd(ttpre, ttarget[index])
                tassd = medpy.metric.binary.assd(ttpre, ttarget[index])
            if ForSaver:
                metrics_list.append(
                    {
                        'SEN': tsen, 'PPV': tprecision,
                        'DSC': tdice, 'JC': tjc,
                        'HD': thd * -1, 'ASD': tasd * -1, 'ASSD': tassd * -1
                    })
            else:
                metrics_list.append(
                    {
                        'SEN': tsen, 'PPV': tprecision,
                        'DSC': tdice, 'JC': tjc,
                        'HD': thd, 'ASD': tasd, 'ASSD': tassd
                    })
        return metrics_list
    else:
        return None

class LogCoshDiceSoftmax(nn.Module):
    def __init__(self, wight=None):
        super(LogCoshDiceSoftmax, self).__init__()
        self.epsilon = 1e-5
        self.wight = wight

    def forward(self, predict, target):
        num = predict.size(0)
        with torch.no_grad():
            if target.size(1) != predict.size(1):
                tar = target.reshape(num, -1)
                tar = F.one_hot(tar.long(), predict.size(1))
                tar = tar.permute(0, 2, 1)
            else:
                tar = target.reshape(num, predict.size(1), -1)
        pre = predict.reshape(num, predict.size(1), -1)
        intersection = (pre * tar).sum(-1)
        union = (pre + tar).sum(-1)
        if self.wight != None:
            score = 1 - (2 * (intersection * self.wight).sum(-1) / (union * self.wight + self.epsilon).sum(-1))
        else:
            score = 1 - (2 * intersection.sum(-1) / (union + self.epsilon).sum(-1))
        return torch.log(torch.exp(score.mean()) + torch.exp(-score.mean()) / 2.0)


class CESoftmax(nn.Module):
    def __init__(self, wight=None):
        super(LogCoshDiceSoftmax, self).__init__()
        self.epsilon = 1e-5
        self.wight = wight

    def forward(self, predict, target):
        num = predict.size(0)
        with torch.no_grad():
            if target.size(1) != predict.size(1):
                tar = target.reshape(num, -1)
                tar = F.one_hot(tar.long(), predict.size(1))
                tar = tar.permute(0, 2, 1)
            else:
                tar = target.reshape(num, predict.size(1), -1)
        pre = predict.reshape(num, predict.size(1), -1)
        if self.wight != None:
            score = -tar * torch.log(pre + 1e-5).sum(1).mean(1).mean()
        else:
            score = -tar * torch.log(pre + 1e-5).sum(1).mean(1).mean()
        return score


class FocalLossSoftmax(nn.Module):
    def __init__(self, gamma=0.7, wight=None):
        super(LogCoshDiceSoftmax, self).__init__()
        self.epsilon = 1e-5
        self.wight = wight
        self.gamma = gamma

    def forward(self, predict, target):
        num = predict.size(0)
        with torch.no_grad():
            if target.size(1) != predict.size(1):
                tar = target.reshape(num, -1)
                tar = F.one_hot(tar.long(), predict.size(1))
                tar = tar.permute(0, 2, 1)
            else:
                tar = target.reshape(num, predict.size(1), -1)
        pre = predict.reshape(num, predict.size(1), -1)
        if self.wight != None:
            score = -(1 - pre) ** self.gamma * tar * torch.log(pre + 1e-5).sum(1).mean(1).mean()
        else:
            score = -(1 - pre) ** self.gamma * tar * torch.log(pre + 1e-5).sum(1).mean(1).mean()
        return score

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num,loss_fn):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.loss_fn = loss_fn
        self.log_vars = nn.Parameter(torch.tensor((0.0,0.0),requires_grad=True)) #1.0, 6.0

    def forward(self, outputs,targets,weights):
        std_1 = torch.exp(self.log_vars[0]) ** 0.5
        std_2 = torch.exp(self.log_vars[1]) ** 0.5

        seg_loss, loss1,loss2,loss3 = self.loss_fn[0](outputs[0], targets[0],weights[0])
        #seg_loss_, loss1,loss2,loss3 = softmax_dice(outputs[0], targets[0],weights[0])
        #seg_loss = self.loss_fn[0](outputs[0], targets[0])
        seg_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[0]) * seg_loss + self.log_vars[0],-1) #

        idh_loss = self.loss_fn[1](outputs[1], targets[1], weights[1])

        idh_loss_1 = torch.sum(0.5 * torch.exp(-self.log_vars[1]) * idh_loss + self.log_vars[1],-1)

        loss = torch.mean(seg_loss_1+idh_loss_1)

        return loss,seg_loss,idh_loss,loss1,loss2,loss3,std_1,std_2,self.log_vars[0],self.log_vars[1]

def idh_cross_entropy(input,target,weight):
    return cross_entropy(input,target,weight=weight,ignore_index=-1)

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, alpha=0.9, gamma=2, focal_enable=False):
        super(SoftDiceLoss, self).__init__()
        self.focal_enable = focal_enable
        self.focal_loss = FocalLoss(alpha, gamma)

    def forward(self, y_pred, y_true, eps=1e-8):   # put 1,2,4 together   (2, 1, 4) 1+4: TC; 4:ET; 1+2+4: WT
        if self.focal_enable:
            focal_loss = self.focal_loss(y_pred, y_true)

        # if self.new_loss:
            # y_pred[:, 0, :, :, :] = torch.sum(y_pred, dim=1)
            # y_pred[:, 1, :, :, :] = torch.sum(y_pred[:, 1:, :, :, :], dim=1)
            # with torch.no_grad():
            #     y_true[:, 0, :, :, :] = torch.sum(y_true, dim=1)
            #     y_true[:, 1, :, :, :] = (torch.sum(y_true[:, 1:, :, :, :], dim=1) != 0).long()

        intersection = torch.sum(torch.mul(y_pred, y_true), dim=[-3, -2, -1])
        union = torch.sum(torch.mul(y_pred, y_pred), dim=[-3, -2, -1]) + torch.sum(torch.mul(y_true, y_true), dim=[-3, -2, -1]) + eps

        dice = 2 * intersection / union   # (bs, 3)
        dice_loss = 1 - torch.mean(dice)  # loss small, better
        # means = torch.mean(dice, dim=2)
        # dice_loss = 1 - 0.5*means[0] - 0.25*means[1] - 0.25*means[2]  # loss small, better

        if self.focal_enable:
            return dice_loss + focal_loss

        return dice_loss

class FocalLoss(_Loss):
    '''
    Focal_Loss = - [alpha * (1 - p)^gamma *log(p)]  if y = 1;
               = - [(1-alpha) * p^gamma *log(1-p)]  if y = 0;
        average over batchsize; alpha helps offset class imbalance; gamma helps focus on hard samples
        其中平衡因子alpha，用来平衡正负样本本身的比例不均; gamma>0使得减少易分类样本的损失,使得更关注于困难的、错分的样本
    '''
    def __init__(self, alpha=0.9, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true, eps=1e-8):

        alpha = self.alpha
        gamma = self.gamma
        focal_ce = - (alpha * torch.pow((1-y_pred), gamma) * torch.log(torch.clamp(y_pred, eps, 1.0)) * y_true
                      + (1-alpha) * torch.pow(y_pred, gamma) * torch.log(torch.clamp(1-y_pred, eps, 1.0)) * (1-y_true))
        focal_loss = torch.mean(focal_ce)

        return focal_loss


def dice_coefficient(outputs, targets, threshold=0.5, eps=1e-8):  # 搞三个dice看 每个label;
    # batch_size = targets.size(0)
    y_pred = outputs[:, :4, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :4, :, :, :]

    num = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred.long(), num)
    y_pred = y_pred.permute(0, 4, 1, 2, 3)
    # y_pred = y_pred > threshold

    y_pred = y_pred.type(torch.FloatTensor)
    back_pred, wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    back_truth, wt_truth, tc_truth, et_truth = combine_labels(y_truth)

    dice0 = dice_coefficient_single_label(back_pred, back_truth, eps)
    dice1 = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    dice2 = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    dice3 = dice_coefficient_single_label(et_pred, et_truth, eps)

    return dice0, dice1, dice2, dice3

def dice_coefficient_wt(outputs, targets, threshold=0.5, eps=1e-8):  # 搞三个dice看 每个label;
    # batch_size = targets.size(0)
    y_pred = outputs[:, :4, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :4, :, :, :]

    num = y_pred.shape[1]
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = F.one_hot(y_pred.long(), num)
    y_pred = y_pred.permute(0, 4, 1, 2, 3)
    # y_pred = y_pred > threshold

    y_pred = y_pred.type(torch.FloatTensor)
    back_pred, wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    back_truth, wt_truth, tc_truth, et_truth = combine_labels(y_truth)

    dice0 = dice_coefficient_single_label(back_pred, back_truth, eps)
    dice1 = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    dice2 = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    dice3 = dice_coefficient_single_label(et_pred, et_truth, eps)

    return dice0, dice1, dice2, dice3


def calculate_accuracy_singleLabel(outputs, targets, threshold=0.5, eps=1e-8):
    # 单类标签计算精度（dice）
    y_pred = outputs[:, 0, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets == 4
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    res = dice_coefficient_single_label(y_pred, y_truth, eps)
    return res


def dice_coefficient_single_label(y_pred, y_truth, eps):
    # dice 计算公式: 2|X∩Y|/|X|+|Y|==>2TP/FP+FN+2TP
    # batch_size = y_pred.size(0)
    intersection = torch.sum(torch.mul(y_pred, y_truth), dim=(-3, -2, -1)) + eps / 2 # axis=?, (bs, 1)
    union = torch.sum(y_pred, dim=(-3, -2, -1)) + torch.sum(y_truth, dim=(-3, -2, -1)) + eps  # (bs, 1)
    dice = 2 * intersection / union
    return dice.mean()
    # return dice / batch_size

def combine_labels(labels):
    """
    Combine label 1 + label 2 + label 4 into WT;      label 1 + label 4 into TC;       label 4 into ET
    GD-enhancing tumor (ET – label 4), the peritumoral edema (ED – label 2),
     and the necrotic and non-enhancing tumor core (NCR/NET –label 1)
    :param labels: torch.Tensor of size (bs, 3, ?,?,?); ? is the crop size 三个通道分别为ed, ncr, et
    :return:
    """
    back = labels[:, 0, :, :, :]
    whole_tumor = labels[:, 1:4, :, :, :].sum(1)  # 3个通道之和
    tumor_core = labels[:, 2:4, :, :, :].sum(1)  # 第2,3个通道之和
    enhanced_tumor = labels[:, 2:3, :, :, :].sum(1)  # 第3个通道
    back[back != 0] = 1
    whole_tumor[whole_tumor != 0] = 1
    tumor_core[tumor_core != 0] = 1
    enhanced_tumor[enhanced_tumor != 0] = 1
    return back, whole_tumor, tumor_core, enhanced_tumor  # (bs, ?, ?, ?)

def specificity(outputs, targets, threshold=0.5):
    # batch_size = targets.size(0)
    y_pred = outputs[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :3, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(y_truth)
    res = dict()
    res["spe_wt"] = specificity_singleLabel(wt_pred, wt_truth)
    res["spe_tc"] = specificity_singleLabel(tc_pred, tc_truth)
    res["spe_et"] = specificity_singleLabel(et_pred, et_truth)

    return res


def specificity_singleLabel(y_pred, y_truth):
    # TN/(FP+TN)
    predBinInv = (y_pred <= 0.5).float()
    targetInv = (y_truth == 0).float()   # 标签0,1互换
    intersection = (predBinInv * targetInv).sum()  # TN
    allNegative = targetInv.sum()   # 所有真实为阴性情况, TN+FP
    return (intersection / allNegative).item()


def hausdorff_distance(outputs, targets, threshold=0.5, confidence_coefficient=95):
    # batch_size = targets.size(0)
    y_pred = outputs[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :3, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(y_truth)
    res = dict()
    res["hd_wt"] = getHd_singleLabel(np.array(wt_pred), np.array(wt_truth), confidence_coefficient)
    res["hd_tc"] = getHd_singleLabel(np.array(tc_pred), np.array(tc_truth), confidence_coefficient)
    res["hd_et"] = getHd_singleLabel(np.array(et_pred), np.array(et_truth), confidence_coefficient)

    return res


def getHd_singleLabel(y_pred, y_truth, confidence_coefficient):
    """
    :param y_pred:
    :param y_truth:
    :param confidence_coefficient: 置信度默认95%
    :return:
    """
    if np.count_nonzero(y_pred) > 0 and np.count_nonzero(y_truth):
        surDist1 = medpyMetrics.__surface_distances(y_pred, y_truth)
        surDist2 = medpyMetrics.__surface_distances(y_truth, y_pred)
        hd = np.percentile(np.hstack((surDist1, surDist2)), confidence_coefficient)
        return hd
    else:
        # Edge cases that medpy cannot handle
        return -1


def mIoU(outputs, targets, threshold=0.5):
    # batch_size = targets.size(0)
    y_pred = outputs[:, :3, :, :, :]  # targets[0,:3,:,:,:]
    y_truth = targets[:, :3, :, :, :]
    y_pred = y_pred > threshold
    y_pred = y_pred.type(torch.FloatTensor)
    wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    wt_truth, tc_truth, et_truth = combine_labels(y_truth)
    res = dict()
    res["mIoU_wt"] = mIoU_singleLabel(wt_pred, wt_truth)
    res["mIoU_tc"] = mIoU_singleLabel(tc_pred, tc_truth)
    res["mIoU_et"] = mIoU_singleLabel(et_pred, et_truth)

    return res


def mIoU_singleLabel(y_pred, y_truth, eps=1e-8):
    # 只关心预测值为真的情况（即前景）,计算方法:并集/交集==>TP/TP+FN+FP,低于precision
    predBin = (y_pred > 0.5).float()
    y_truth = y_truth.float()
    intersection = (predBin * y_truth).sum() + eps  # TP
    allPositive = y_truth.sum()  # 所有真实为阳性, TP+FN
    union = allPositive + predBin.sum() - intersection + eps   # (predBin | y_truth).sum()其实更简单
    if union == 0:
        return 0.0
    return float(intersection) / float(union)

def InitWeights(model, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                pass
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    init_func(model)

if __name__ == "__main__" :
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img = torch.randn(2, 4, 128, 128, 128).cuda()
