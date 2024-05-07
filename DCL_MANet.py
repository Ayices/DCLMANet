import os
import torch.nn
from torch.utils.data import DataLoader
from Init.init import InitParser
from Init.utils import *
from Init.Dataset import *
from models.M_3D.FusionTry_3D import DCL_MANet
import time

from train import train_epoch
from test import test_epoch
from predict import check_accuracy_model_multiclass

def list_subdirectories(base_path):
    # 存储子目录的列表
    subdirs = []

    # 遍历给定路径的所有条目
    for entry in os.listdir(base_path):
        # 拼接完整的路径
        full_path = os.path.join(base_path, entry)

        # 检查这个条目是否是目录
        if os.path.isdir(full_path):
            subdirs.append(full_path)

    return subdirs

def main(args):
    # 设置Checkpoint文件夹和Log文件夹
    ckpt_path = os.path.join(args.output_path_1,
                             "Checkpoint_DCLMANet_seg_3DTorch2_batch2")  # /History/Checkpoint2
    log_path = os.path.join(args.output_path_1, "Log_DCLMANet_seg_3DTorch2_batch2")  # /History/Log2

    # 检查文件夹是否存在
    check_dir(args.output_path_1)  # ./History
    check_dir(log_path)
    check_dir(ckpt_path)

    torch.cuda.device_count()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8'
    torch.cuda.set_device(int(args.gpu_id))

    # 训练流程
    if args.do_you_wanna_train is True:

        # 验证数据集地址
        folder_path = args.val_path
        directories = list_subdirectories(folder_path)

        val_list = [{'flair': path + '/' + path[-15:] + '_flair.nii.gz',
                     't1': path + '/' + path[-15:] + '_t1.nii.gz',
                     't1ce': path + '/' + path[-15:] + '_t1ce.nii.gz',
                     't2': path + '/' + path[-15:] + '_t2.nii.gz',
                     'label': path + '/' + path[-15:] + '_seg.nii.gz',
                     } for path in directories]

        # 训练数据集地址
        folder_path = args.train_path
        directories = list_subdirectories(folder_path)

        train_list = [{'flair': path + '/' + path[-15:] + '_flair.nii.gz',
                       't1': path + '/' + path[-15:] + '_t1.nii.gz',
                       't1ce': path + '/' + path[-15:] + '_t1ce.nii.gz',
                       't2': path + '/' + path[-15:] + '_t2.nii.gz',
                       'label': path + '/' + path[-15:] + '_seg.nii.gz',
                       } for path in directories]

        print('Number of training patches per epoch:', len(train_list))
        print('Number of validation patches per epoch:', len(val_list))

        train_Transforms = [
            Random_Crop(),
            Random_Flip(),
            ToTensor()
        ]

        val_Transforms = [
            Random_Crop(),
            ToTensor()
        ]

        train_set = MyDataSet3D(data_list=train_list, transforms=train_Transforms, train=True)
        val_set = MyDataSet3D(data_list=val_list, transforms=val_Transforms, test=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

        net = DCL_MANet(img_ch=4, output_ch=4, base_channel=16)
        print(net.__class__.__name__)
        net = net.cuda()

        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))

        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.amsgrad)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               factor=args.lr_reduce_factor,
                                                               patience=args.lr_reduce_patience,
                                                               mode='max',
                                                               min_lr=args.min_lr)

        best_dice = 0.0
        start_epoch = args.init_epoch

        if args.do_you_wanna_load_weights is False:
            InitWeights(net, 'normal', 0.02)

        if args.do_you_wanna_load_epoch is True:
            checkpoint = torch.load(args.load_path,map_location="cuda:1")
            net.load_state_dict(checkpoint['net'])

        for epoch in range(start_epoch, start_epoch + args.num_epoch):  # define the epochs number
            start_time = time.time()

            # train one epoch
            epoch_train_dice0, epoch_train_dice1, epoch_train_dice2, epoch_train_dice3, epoch_loss = train_epoch(net,
                                                                                                                 train_loader,
                                                                                                                 optimizer
                                                                                                                 )
            # val one epoch
            epoch_val_dice0, epoch_val_dice1, epoch_val_dice2, epoch_val_dice3, epoch_val_loss = test_epoch(net,
                                                                                                            val_loader
                                                                                                            )

            scheduler.step(best_dice)

            if epoch % 20 == 0:
                state = {
                    'net': net.state_dict(),
                }
                torch.save(state, os.path.join(ckpt_path, "Network_{}epoch.pth.gz".format(epoch)))

            if epoch_val_dice3 > best_dice:
                best_dice = epoch_val_dice3
                state = {
                    'net': net.state_dict(),
                }
                torch.save(state,
                           os.path.join(ckpt_path, "Best_Dice3_{:.2f}_{}epoch.pth.gz".format(epoch_val_dice3, epoch)))

            epoch_time = time.time() - start_time

            info_line = "Epoch {} || Loss: {:.4f} | Time(min): {:.2f} | Validation Loss: {:.4f} | Validation Dice0: {:.4f} | Validation Dice1: {:.4f} | Validation Dice2: {:.4f} | Validation Dice3: {:.4f} | lr: {:.6f}".format(
                str(epoch).zfill(3), epoch_loss, epoch_time / 60, epoch_val_loss, epoch_val_dice0, epoch_val_dice1,
                epoch_val_dice2, epoch_val_dice3,
                optimizer.state_dict()['param_groups'][0]['lr']
            )

            print(info_line)
            open(os.path.join(log_path, 'train_log.txt'), 'a').write(info_line + '\n')

        # predict
        if args.do_you_wanna_check_accuracy is True:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7,8'
            torch.cuda.set_device(int(args.gpu_id))

            net = DCL_MANet(img_ch=4, output_ch=4, base_channel=16)
            net_name = net.__class__.__name__
            print(net_name)
            net.cuda()

            load_path = args.load_path
            checkpoint_unet = torch.load(load_path)
            net.load_state_dict(checkpoint_unet['net'])
            net.eval()

            folder_path = args.val_path
            directories = list_subdirectories(folder_path)

            val_list = [{'flair': path + '/' + path[-15:] + '_flair.nii.gz',
                         't1': path + '/' + path[-15:] + '_t1.nii.gz',
                         't1ce': path + '/' + path[-15:] + '_t1ce.nii.gz',
                         't2': path + '/' + path[-15:] + '_t2.nii.gz',
                         'label': path + '/' + path[-15:] + '_seg.nii.gz',
                         } for path in directories]

            print("Now checking accuracy on validation set")

            Dice0, Dice1, Dice2, Dice3 = check_accuracy_model_multiclass(net_name, net, val_list,
                                                                         args.resample, args.new_resolution,
                                                                         args.patch_size[0],
                                                                         args.patch_size[1], args.patch_size[2],
                                                                         args.stride_inplane, args.stride_layer)

            print(load_path)




if __name__ == '__main__':
    seed = 3407
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    parsers = InitParser()  # 初始化参数
    main(parsers)
