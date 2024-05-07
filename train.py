import time
from Init.init import InitParser
import torch
import torch.nn.functional as F
from Init.utils import *
from torch.nn import CrossEntropyLoss
import cv2

args = InitParser()


class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc
    and self.avg to get the avg acc
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(net, loader, optimizer):
    print("train")
    net.train()

    test_dice_meter0 = AvgMeter()
    test_dice_meter1 = AvgMeter()
    test_dice_meter2 = AvgMeter()
    test_dice_meter3 = AvgMeter()
    batch_loss = AvgMeter()

    # for batch_idx, (t1, t1ce, t2, flair, label, label_np) in enumerate(loader):
    for batch_idx, (t1, t1ce, t2, flair, label, label_np) in enumerate(loader):
        # label是处理后的， label_np是未处理的
        t1 = t1.cuda()
        t1ce = t1ce.cuda()
        t2 = t2.cuda()
        flair = flair.cuda()
        label = label.cuda()
        label_net = torch.rand(
            (label_np.shape[0], label_np.shape[1], label_np.shape[2], label_np.shape[3], label_np.shape[4]))
        label_net.copy_(label_np)
        label_net = label_net.squeeze(dim=1).cuda()
        label_np = label_np.cuda()

        data = torch.cat((flair, t1, t1ce, t2), 1)
        # data = t2

        label_net[label_net == 1] = 3
        label_net[label_net == 2] = 1
        label_net[label_net == 4] = 2

        # output = net(data)
        output, loss_contra = net(data, is_train=True)
        # output, t1c_out, flair_out = net(data)
        # output, out_arc, arc_label, loss_contra = net(data, seg_label = label_net, is_train = True)

        # output_loss = output[1]
        # output = output[0] # MISSU


        loss_dice, _, _, _, _ = softmax_dice(output, label_np)


        # Loss_cos = CrossEntropyLoss(reduction='mean')
        # loss_cos = Loss_cos(out_arc, arc_label)
        loss_cross = cross_entro(output, label_np)
        # loss = loss_dice + loss_cross + loss_contra

        # loss = 0.5 * loss_dice + 0.5 * loss_cross + loss_contra
        # loss = loss_dice + loss_cross + output_loss
        loss = loss_dice + loss_cross
        # for loss_dis in output_loss:
        #     loss += loss_dis
        # loss = sigmoid_dice(output, label_np)

        output = output.data.cpu()
        label = label.cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice0, dice1, dice2, dice3 = dice_coefficient(output, label)
        # dice1 = calculate_accuracy_singleLabel(output, label_np)

        batch_loss.update(loss.item())
        test_dice_meter0.update(dice0)
        test_dice_meter1.update(dice1)
        test_dice_meter2.update(dice2)
        test_dice_meter3.update(dice3)

        # loss = loss.detach()
        # loss_dice = loss_dice.detach()
        # loss_cross = loss_cross.detach()

        print(
            "Train Batch {} || Loss: {:.4f} || Training Dice0: {:.4f} | Training Dice1: {:.4f} | Training Dice2: {:.4f} | Training Dice3: {:.4f} ".format(
                str(batch_idx).zfill(4), batch_loss.val, dice0, dice1, dice2, dice3))

        # print(
        #     "Train Batch {} || Loss: {:.4f} | Training Dice1: {:.4f} ".format(
        #         str(batch_idx).zfill(4), batch_loss.val, dice1))

    return test_dice_meter0.avg, \
           test_dice_meter1.avg, \
           test_dice_meter2.avg, \
           test_dice_meter3.avg, \
           batch_loss.avg

