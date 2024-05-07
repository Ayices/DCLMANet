import glob
import os
import nibabel as nib
import torch
import numpy as np
import random
from torch.nn.functional import interpolate, upsample_nearest

random.seed(3407)
np.random.seed(3407)

from sklearn.model_selection import KFold

def Normalization(image):
    image = ((image - image.min()) / (image.max() - image.min()))
    return image

def standartion(image):
    image = (image - image.mean()) / image.std()
    return image

class Random_Crop_image(object):
    def __call__(self, sample):
        image = sample

        Random_size_x = 128
        Random_size_y = 128
        Random_size_z = 128

        H = random.randint(0, 240 - Random_size_x)
        W = random.randint(0, 240 - Random_size_y)
        D = random.randint(0, 155 - Random_size_z)

        image = image[H: H + Random_size_x, W: W + Random_size_y, D: D + Random_size_z]

        return image

class Random_Crop(object):
    def __call__(self, sample):
        t1 = sample['t1']
        t1ce = sample['t1ce']
        t2 = sample['t2']
        flair = sample['flair']
        label = sample['label']

        Random_size_x = 128
        Random_size_y = 128
        Random_size_z = 128

        H = random.randint(0, 240 - Random_size_x)
        W = random.randint(0, 240 - Random_size_y)
        D = random.randint(0, 155 - Random_size_z)

        t1 = t1[H: H + Random_size_x, W: W + Random_size_y, D: D + Random_size_z]
        t1ce = t1ce[H: H + Random_size_x, W: W + Random_size_y, D: D + Random_size_z]
        t2 = t2[H: H + Random_size_x, W: W + Random_size_y, D: D + Random_size_z]
        flair = flair[H: H + Random_size_x, W: W + Random_size_y, D: D + Random_size_z]
        label = label[H: H + Random_size_x, W: W + Random_size_y, D: D + Random_size_z]

        return {'t1': t1,
                't1ce': t1ce,
                't2': t2,
                'flair': flair,
                'label': label}

class Random_Flip_image(object):
    def __call__(self, sample):
        image = sample

        if random.random() < 0.5:
            image = np.flip(image, 0)

        if random.random() < 0.5:
            image = np.flip(image, 1)

        if random.random() < 0.5:
            image = np.flip(image, 1)

        return image

class Random_Flip(object):
    def __call__(self, sample):
        t1 = sample['t1']
        t1ce = sample['t1ce']
        t2 = sample['t2']
        flair = sample['flair']
        label = sample['label']

        if random.random() < 0.5:
            t1 = np.flip(t1, 0)
            t2 = np.flip(t2, 0)
            t1ce = np.flip(t1ce, 0)
            flair = np.flip(flair, 0)
            label = np.flip(label, 0)

        if random.random() < 0.5:
            t1 = np.flip(t1, 1)
            t2 = np.flip(t2, 1)
            t1ce = np.flip(t1ce, 1)
            flair = np.flip(flair, 1)
            label = np.flip(label, 1)

        if random.random() < 0.5:
            t1 = np.flip(t1, 1)
            t2 = np.flip(t2, 1)
            t1ce = np.flip(t1ce, 1)
            flair = np.flip(flair, 1)
            label = np.flip(label, 1)

        return {'t1': t1,
                't1ce': t1ce,
                't2': t2,
                'flair': flair,
                'label': label}

class Random_intencity_shift(object):
    def __call__(self, sample, factor=0.1):
        t1 = sample['t1']
        t1ce = sample['t1ce']
        t2 = sample['t2']
        flair = sample['flair']
        label = sample['label']

        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[t1.shape[0], t1.shape[1], t1.shape[2]])
        shift_factor = np.random.uniform(-factor, factor, size=[t1.shape[0], t1.shape[1], t1.shape[2]])

        t1 = t1 * scale_factor + shift_factor
        t2 = t2 * scale_factor + shift_factor
        t1ce = t1ce * scale_factor + shift_factor
        flair = flair * scale_factor + shift_factor

        return {'t1': t1,
                't1ce': t1ce,
                't2': t2,
                'flair': flair,
                'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        t1 = sample['t1']
        t1ce = sample['t1ce']
        t2 = sample['t2']
        flair = sample['flair']
        label = sample['label']
        label_np = sample['label']

        t1 = np.ascontiguousarray(t1)
        t1ce = np.ascontiguousarray(t1ce)
        t2 = np.ascontiguousarray(t2)
        flair = np.ascontiguousarray(flair)
        label = np.ascontiguousarray(label)
        label_np = np.ascontiguousarray(label_np)

        label = preprocess_label(label)

        t1 = torch.from_numpy(t1).float()
        t1ce = torch.from_numpy(t1ce).float()
        t2 = torch.from_numpy(t2).float()
        flair = torch.from_numpy(flair).float()
        label = torch.from_numpy(label).long()
        label_np = torch.from_numpy(label_np).long()

        return {'t1': t1,
                't1ce': t1ce,
                't2': t2,
                'flair': flair,
                'label': label,
                'label_np': label_np,
                }

class ToTensor_image(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image = sample

        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).float()

        return image

def preprocess_label(img, single_label=None):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    back = img == 0
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET) - orange
    ed = img == 2  # Peritumoral Edema (ED) - yellow
    et = img == 4  # GD-enhancing Tumor (ET) - blue
    if not single_label:
        # return np.array([ncr, ed, et], dtype=np.uint8)
        return np.array([back, ed, et, ncr], dtype=np.uint8)   # 组成四个通道分别为back, ed, ncr, et
    elif single_label == "WT":
        # img[ncr] = 1  # 本区域标签本身就是1,可不必再次赋值
        img[ed] = 1
        img[et] = 1
    elif single_label == "TC":
        img[ncr] = 0
        img[ed] = 1
        img[et] = 1
    elif single_label == "ET":
        img[ncr] = 0
        img[ed] = 0
        img[et] = 1
    else:
        raise RuntimeError("the 'single_label' type must be one of WT, TC, ET, and None")
    return img[np.newaxis, :]

def preprocess_label_MSFR(img):

    img[np.where(img==4)] = 3
    seg_all = img.copy()
    seg_t1c = img.copy()
    seg_flair = img.copy()
    seg_t1c[np.where(seg_t1c == 2)] = 0
    seg_flair[np.where(seg_flair == 1)] = 3

    return {'seg_all': seg_all, 'seg_t1c': seg_t1c, 'seg_flair':seg_flair}

class MyDataSet3D_classifier(torch.utils.data.Dataset):

    def __init__(self, data_list,
                 transforms=None,
                 train = False,
                 test = False,
                 num_class = 4
                 ):

        # Init membership variables
        self.data_list = data_list
        self.transforms = transforms
        self.train = train
        self.test = test
        self.num_class = num_class


    def __getitem__(self, item):

        data_dict = self.data_list[item]

        image = nib.load(data_dict).get_fdata()

        if data_dict[-9:-7] == "ce":
            label = 2
        if data_dict[-9:-7] == "t1":
            label = 1
        if data_dict[-9:-7] == "t2":
            label = 3
        if data_dict[-9:-7] == "ir":
            label = 0

        # if data_dict[-9:-7] == "ce":
        #     label = [0,0,1,0]
        # if data_dict[-9:-7] == "t1":
        #     label = [0,1,0,0]
        # if data_dict[-9:-7] == "t2":
        #     label = [0,0,0,1]
        # if data_dict[-9:-7] == "ir":
        #     label = [1,0,0,0]

        if self.transforms:
            for transform in self.transforms:
                image = transform(image)

        image = torch.unsqueeze(image, dim=0).float()
        label_np = np.array(label)
        label_tensor = torch.from_numpy(label_np)

        return image, label_tensor

    def __len__(self):
        return len(self.data_list)

class MyDataSet3D(torch.utils.data.Dataset):

    def __init__(self, data_list,
                 transforms=None,
                 train = False,
                 test = False,
                 num_class = 4
                 ):

        # Init membership variables
        self.data_list = data_list
        self.transforms = transforms
        self.train = train
        self.test = test
        self.num_class = num_class


    def __getitem__(self, item):

        # 如果是加载训练集
        if self.train:
            data_dict = self.data_list[item]
            t1_path = data_dict["t1"]
            t1ce_path = data_dict["t1ce"]
            t2_path = data_dict["t2"]
            flair_path = data_dict["flair"]
            label_path = data_dict["label"]

            # read image and label
            t1_image = nib.load(t1_path).get_fdata()
            t1ce_image = nib.load(t1ce_path).get_fdata()
            t2_image = nib.load(t2_path).get_fdata()
            flair_image = nib.load(flair_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            # 归一化
            t1_image = Normalization(t1_image)
            t1ce_image = Normalization(t1ce_image)
            t2_image = Normalization(t2_image)
            flair_image = Normalization(flair_image)

            sample = {'t1': t1_image,
                      't1ce': t1ce_image,
                      't2': t2_image,
                      'flair': flair_image,
                      'label': label}

            if self.transforms:
                for transform in self.transforms:
                    sample = transform(sample)

            t1 = sample['t1']
            t1ce = sample['t1ce']
            t2 = sample['t2']
            flair = sample['flair']
            label = sample['label']
            label_np = sample['label_np']

            t1 = torch.unsqueeze(t1, dim=0).float()
            t1ce = torch.unsqueeze(t1ce, dim=0).float()
            t2 = torch.unsqueeze(t2, dim=0).float()
            flair = torch.unsqueeze(flair, dim=0).float()
            label_np = torch.unsqueeze(label_np, dim=0).long()

            return t1, t1ce, t2, flair, label, label_np

        else:
            data_dict = self.data_list[item]
            t1_path = data_dict["t1"]
            t1ce_path = data_dict["t1ce"]
            t2_path = data_dict["t2"]
            flair_path = data_dict["flair"]
            label_path = data_dict["label"]

            # read image and label
            t1_image = nib.load(t1_path).get_fdata()
            t1ce_image = nib.load(t1ce_path).get_fdata()
            t2_image = nib.load(t2_path).get_fdata()
            flair_image = nib.load(flair_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            t1_image = Normalization(t1_image)
            t1ce_image = Normalization(t1ce_image)
            t2_image = Normalization(t2_image)
            flair_image = Normalization(flair_image)

            sample = {'t1': t1_image,
                      't1ce': t1ce_image,
                      't2': t2_image,
                      'flair': flair_image,
                      'label': label}

            if self.transforms:
                for transform in self.transforms:
                    sample = transform(sample)

            t1 = sample['t1']
            t1ce = sample['t1ce']
            t2 = sample['t2']
            flair = sample['flair']
            label = sample['label']
            label_np = sample['label_np']

            t1 = torch.unsqueeze(t1, dim=0).float()
            t1ce = torch.unsqueeze(t1ce, dim=0).float()
            t2 = torch.unsqueeze(t2, dim=0).float()
            flair = torch.unsqueeze(flair, dim=0).float()
            label_np = torch.unsqueeze(label_np, dim=0).long()

            return t1, t1ce, t2, flair, label, label_np

    def __len__(self):
        return len(self.data_list)

def resample_image(image, new_size=None, scale_factor=None, mode='trilinear'):
    '''
    :param image: 这是一个tensor [高，宽，深]
    :param new_size: 新的尺寸（原来的倍数）
    :param mode: 目前只用三线性插值
    :return:返回一个tensor [高度，宽度，切片数目]
    '''
    if mode == 'trilinear':
        if new_size == None and scale_factor == None:
            raise Exception('不能同时出现new_size和scale_factor的情况')
        imagetensor = torch.from_numpy(image.copy())
        imagetensor = torch.unsqueeze(imagetensor, dim=0)
        imagetensor = torch.unsqueeze(imagetensor, dim=0)
        imagetensor = interpolate(imagetensor, size=new_size, scale_factor=scale_factor, mode='trilinear',
                                  align_corners=False)
        imagetensor = torch.squeeze(imagetensor)
        return imagetensor.numpy()
    elif mode == 'nearest':
        if new_size == None and scale_factor == None:
            raise Exception('不能同时出现new_size和scale_factor的情况')
        imagetensor = torch.from_numpy(image.copy())
        imagetensor = torch.unsqueeze(imagetensor, dim=0)
        imagetensor = torch.unsqueeze(imagetensor, dim=0)
        imagetensor = interpolate(imagetensor, size=new_size, scale_factor=scale_factor, mode='nearest')
        return torch.squeeze(imagetensor).numpy()

class Resample(object):
    def __init__(self, new_resolution, check):
        self.name = 'Resample'
        # self.new_resolution = [new_resolution[0], new_resolution[1], new_resolution[2]]
        self.new_resolution = new_resolution
        self.check = check

    def __call__(self, sample):
        t1 = sample['t1']
        t1ce = sample['t1ce']
        t2 = sample['t2']
        flair = sample['flair']
        label = sample['label']

        check = self.check

        if check is True:
            t1 = resample_image(t1, new_size=self.new_resolution, mode='trilinear')
            t1ce = resample_image(t1ce, new_size=self.new_resolution, mode='trilinear')
            t2 = resample_image(t2, new_size=self.new_resolution, mode='trilinear')
            flair = resample_image(flair, new_size=self.new_resolution, mode='trilinear')
            label = resample_image(label, new_size=self.new_resolution, mode='nearest')

            return {'t1': t1,
                    't1ce': t1ce,
                    't2': t2,
                    'flair': flair,
                    'label': label}

        if check is False:
            return {'t1': t1,
                't1ce': t1ce,
                't2': t2,
                'flair': flair,
                'label': label}

class DataSet_to_statistic(torch.utils.data.Dataset):

    def __init__(self, data_list,
                 transforms=None,
                 train = False,
                 test = False,
                 num_class = 4
                 ):

        # Init membership variables
        self.data_list = data_list
        self.transforms = transforms
        self.train = train
        self.test = test
        self.num_class = num_class


    def __getitem__(self, item):

        data_dict = self.data_list[item]
        t1_path = data_dict["t1"]
        t1ce_path = data_dict["t1ce"]
        t2_path = data_dict["t2"]
        flair_path = data_dict["flair"]
        label_path = data_dict["label"]

        # read image and label
        t1_image = nib.load(t1_path).get_fdata()
        t1ce_image = nib.load(t1ce_path).get_fdata()
        t2_image = nib.load(t2_path).get_fdata()
        flair_image = nib.load(flair_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # 归一化
        t1_image = Normalization(t1_image)
        t1ce_image = Normalization(t1ce_image)
        t2_image = Normalization(t2_image)
        flair_image = Normalization(flair_image)

        sample = {'t1': t1_image,
                  't1ce': t1ce_image,
                  't2': t2_image,
                  'flair': flair_image,
                  'label': label}

        if self.transforms:
            for transform in self.transforms:
                sample = transform(sample)

        t1 = sample['t1']
        t1ce = sample['t1ce']
        t2 = sample['t2']
        flair = sample['flair']
        label = sample['label']
        label_np = sample['label_np']

        return t1, t1ce, t2, flair, label, label_np

    def __len__(self):
        return len(self.data_list)

import pickle as pkl

if __name__ == '__main__':
    # path = '/home/public/huxubin/3DBrats/Data_folder/train_set/'
    # train_list, val_list = create_list_K_fold(path, 1)
    # print('Number of training patches per epoch:', len(train_list))
    # print('Number of validation patches per epoch:', len(val_list))
    # with open("train_list.pkl", "wb") as f:
    #     pkl.dump(train_list,f)
    # with open("val_list.pkl", "wb") as f:
    #     pkl.dump(val_list,f)
    f = open("train_list.pkl", "rb")
    train_list = pkl.load(f)
