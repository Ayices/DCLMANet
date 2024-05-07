import time
from Init.init import InitParser
import torch
import torch.nn.functional as F
from Init.utils import *

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

def test_epoch(net, loader):
    # we transfer the mode of network to test
    net.eval()
    with torch.no_grad():
        print("test...")
        test_dice_meter0 = AvgMeter()
        test_dice_meter1 = AvgMeter()
        test_dice_meter2 = AvgMeter()
        test_dice_meter3 = AvgMeter()
        batch_loss = AvgMeter()
        # for batch_idx, (t1, t1ce, t2, flair, label, label_np) in enumerate(loader):
        for batch_idx, (t1, t1ce, t2, flair, label, label_np) in enumerate(
                loader):
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

            label_net[label_net == 1] = 3
            label_net[label_net == 2] = 1
            label_net[label_net == 4] = 2

            # output = net(data, is_train=False)  # Give the data to the network
            # output = net(data, seg_label = label_net, is_train=False)  # Give the data to the network
            output = net(data)  # Give the data to the network

            # output = output[0]

            # label = label.squeeze(dim=1)
            # label = label.long()

            # loss = dice_multi_class_loss(output, label)
            # loss = sigmoid_dice(output, label_np)

            loss, _, _, _, _ = softmax_dice(output, label_np)

            # evaluate the cost function

            # output = output.squeeze().data.cpu().numpy()
            # label = label.squeeze().cpu().numpy()

            output = output.data.cpu()
            label = label.cpu()

            dice0, dice1, dice2, dice3 = dice_coefficient(output, label)
            # dice1 = calculate_accuracy_singleLabel(output, label_np)

            # dice = dice_coefff(output, label)

            test_dice_meter0.update(dice0)
            test_dice_meter1.update(dice1)
            test_dice_meter2.update(dice2)
            test_dice_meter3.update(dice3)
            batch_loss.update(loss.item())

            # print("Test {} || Dice0: {:.4f}".format(str(batch_idx).zfill(4), test_dice_meter.val))
            print("Test {} || Dice0: {:.4f} | Dice1: {:.4f} | Dice2: {:.4f} | Dice3: {:.4f}".format(str(batch_idx).zfill(4),
                                                                                                    test_dice_meter0.val,
                                                                                                    test_dice_meter1.val,
                                                                                                    test_dice_meter2.val,
                                                                                                    test_dice_meter3.val))
            # print("Test {} || Dice1: {:.4f} ".format(str(batch_idx).zfill(4),
            #                         test_dice_meter1.val))

        # return test_dice_meter.avg, batch_loss.avg
        return test_dice_meter0.avg,\
               test_dice_meter1.avg, \
               test_dice_meter2.avg, \
               test_dice_meter3.avg, \
               batch_loss.avg
