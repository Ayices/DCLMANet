import numpy as np
import os
from Init.Dataset import *
import math
from tqdm import tqdm
import torch.nn.functional as F
from Init.Dataset import preprocess_label
from Init.utils import combine_labels, dice_coefficient_single_label
import nibabel as nib
import datetime
import matplotlib.pyplot as plt
import medpy
from MySegMetrics import *
from SegMetricUtils import *
import pickle

class LayerActivations:
    features = None

    def __init__(self, model, layer_num):
        self.hook = model.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, MRI_tensor, output):
        self.features = output

    def remove(self):
        self.hook.remove()

def check_dir(path):              # if folder does not exist, create it
    if not os.path.exists(path):
        os.mkdir(path)

def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for patch in ijk_patch_indices:
        image_patch = image[..., patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
        image_batches.append(image_patch)
    return image_batches


def metrics_pre(outputs, targets, eps=1e-8):
    y_pred = outputs
    y_truth = targets

    back_pred, wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    back_truth, wt_truth, tc_truth, et_truth = combine_labels(y_truth)

    wt_pred, tc_pred, et_pred = wt_pred.squeeze().numpy(), tc_pred.squeeze().numpy(), et_pred.squeeze().numpy()
    wt_truth, tc_truth, et_truth = wt_truth.squeeze().numpy(), tc_truth.squeeze().numpy(), et_truth.squeeze().numpy()

    #

    # # sensitive
    sen1 = medpy.metric.binary.sensitivity(wt_pred, wt_truth)
    sen2 = medpy.metric.binary.sensitivity(tc_pred, tc_truth)
    sen3 = medpy.metric.binary.sensitivity(et_pred, et_truth)
    #
    # ppv
    ppv1 = medpy.metric.binary.precision(wt_pred, wt_truth)
    ppv2 = medpy.metric.binary.precision(tc_pred, tc_truth)
    ppv3 = medpy.metric.binary.precision(et_pred, et_truth)

    #
    # jaccard
    if np.sum(wt_truth) == 0 or np.sum(wt_pred) == 0:
        jac1 = 0
    else:
        jac1 = medpy.metric.binary.jc(wt_pred, wt_truth)

    if np.sum(tc_truth) == 0 or np.sum(tc_pred) == 0:
        # hd3 = medpy.metric.binary.hd(et_pred, et_truth)
        jac2 = 0
    else:
        jac2 = medpy.metric.binary.jc(tc_pred, tc_truth)

    if np.sum(et_truth) == 0 or np.sum(et_pred) == 0:
        # hd3 = medpy.metric.binary.hd(et_pred, et_truth)
        jac3 = 0
    else:
        jac3 = medpy.metric.binary.jc(et_pred, et_truth)



    # HD
    if np.sum(wt_truth) == 0 or np.sum(wt_pred) == 0:
        hd1 = 0
    else:
        hd1 = medpy.metric.binary.hd95(wt_pred, wt_truth)

    if np.sum(tc_truth) == 0 or np.sum(tc_pred) == 0:
        # hd3 = medpy.metric.binary.hd(et_pred, et_truth)
        hd2 = 0
    else:
        hd2 = medpy.metric.binary.hd95(tc_pred, tc_truth)
    # hd2 = medpy.metric.binary.hd(tc_pred, tc_truth)
    if np.sum(et_truth) == 0 or np.sum(et_pred) == 0:
        # hd3 = medpy.metric.binary.hd(et_pred, et_truth)
        hd3 = 0
    else:
        hd3 = medpy.metric.binary.hd95(et_pred, et_truth)


    # ASSD
    if np.sum(tc_truth) == 0 or np.sum(tc_pred) == 0:
        assd1 = 0
    else:
        assd1 = medpy.metric.binary.assd(wt_pred, wt_truth)

    if np.sum(tc_truth) == 0 or np.sum(tc_pred) == 0:
        # hd3 = medpy.metric.binary.hd(et_pred, et_truth)
        assd2 = 0
    else:
        assd2 = medpy.metric.binary.assd(tc_pred, tc_truth)
    # assd2 = medpy.metric.binary.assd(tc_pred, tc_truth)
    if np.sum(et_truth) == 0 or np.sum(et_pred) == 0:
        # hd3 = medpy.metric.binary.hd(et_pred, et_truth)
        assd3 = 0
    else:
        assd3 = medpy.metric.binary.assd(et_pred, et_truth)


    metrics_list = {
        'SEN1': sen1,
        'SEN2': sen2,
        'SEN3': sen3,
        'PPV1': ppv1,
        'PPV2': ppv2,
        'PPV3': ppv3,
        'JAC1': jac1,
        'JAC2': jac2,
        'JAC3': jac3,
        'HD1': hd1,
        'HD2': hd2,
        'HD3': hd3,
        'ASSD1': assd1,
        'ASSD2': assd2,
        'ASSD3': assd3,
    }
    return metrics_list


def dice_pre(outputs, targets, threshold=0.5, eps=1e-8):  # 搞三个dice看 每个label;
    # batch_size = targets.size(0)
    y_pred = outputs
    y_truth = targets

    back_pred, wt_pred, tc_pred, et_pred = combine_labels(y_pred)
    back_truth, wt_truth, tc_truth, et_truth = combine_labels(y_truth)

    dice0 = dice_coefficient_single_label(back_pred, back_truth, eps)
    dice1 = dice_coefficient_single_label(wt_pred, wt_truth, eps)
    dice2 = dice_coefficient_single_label(tc_pred, tc_truth, eps)
    dice3 = dice_coefficient_single_label(et_pred, et_truth, eps)

    return dice0, dice1, dice2, dice3


def inference_multiclass(model, flair_path, t1_path, t1ce_path, t2_path
                         , label_path, result_path, resample, resolution,
                         patch_size_x, patch_size_y, patch_size_z,
                         stride_inplane, stride_layer, batch_size=1, segmentation=True):
    Transforms = [
        # Resample(resolution, resample),
        ToTensor()
    ]

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
              'label': label}  # 样本

    for transform in Transforms:  # 重采样
        sample = transform(sample)

    flair_tfm, t1_tfm, t1ce_tfm, t2_tfm, label_tfm = sample['flair'], sample['t1'], sample['t1ce'], sample['t2'], \
                                                     sample['label_np']

    pre = torch.zeros((1, 4, flair_tfm.shape[-3], flair_tfm.shape[-2], flair_tfm.shape[-1]))

    inum = int(math.ceil((flair_tfm.shape[-3] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((flair_tfm.shape[-2] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((flair_tfm.shape[-1] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                istart = i * stride_inplane
                if istart + patch_size_x > flair_tfm.shape[-3]:  # for last patch
                    istart = flair_tfm.shape[-3] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > flair_tfm.shape[-2]:  # for last patch
                    jstart = flair_tfm.shape[-2] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > flair_tfm.shape[-1]:  # for last patch
                    kstart = flair_tfm.shape[-1] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indices.append([istart, iend, jstart, jend, kstart, kend])
                patch_total += 1

    batches_flair = prepare_batch(flair_tfm, ijk_patch_indices)
    batches_t1 = prepare_batch(t1_tfm, ijk_patch_indices)
    batches_t1ce = prepare_batch(t1ce_tfm, ijk_patch_indices)
    batches_t2 = prepare_batch(t2_tfm, ijk_patch_indices)
    batches_label = prepare_batch(label_tfm, ijk_patch_indices)

    for i in range(len(batches_flair)):
        batch_flair = batches_flair[i]
        batch_t1 = batches_t1[i]
        batch_t1ce = batches_t1ce[i]
        batch_t2 = batches_t2[i]
        batch_label = batches_label[i]

        batch_flair = torch.unsqueeze(batch_flair, 0)
        batch_flair = torch.unsqueeze(batch_flair, 0)
        batch_t1 = torch.unsqueeze(batch_t1, 0)
        batch_t1 = torch.unsqueeze(batch_t1, 0)
        batch_t1ce = torch.unsqueeze(batch_t1ce, 0)
        batch_t1ce = torch.unsqueeze(batch_t1ce, 0)
        batch_t2 = torch.unsqueeze(batch_t2, 0)
        batch_t2 = torch.unsqueeze(batch_t2, 0)
        batch_label = torch.unsqueeze(batch_label, 0)
        batch_label = torch.unsqueeze(batch_label, 0)

        data = torch.cat((batch_flair, batch_t1, batch_t1ce, batch_t2), 1).cuda()
        batch_label = batch_label.cuda()
        output = model(data)
        # output, loss_contra = model(data, is_train=True, label=batch_label)
        # output = model(data, is_train=False)
        # output = model(data, seg_label=batch_flair, is_train=False)
        # output = F.softmax(output[0], dim=1)
        output = output.data.cpu()
        # output = output[0].data.cpu()
        istart = ijk_patch_indices[i][0]
        iend = ijk_patch_indices[i][1]
        jstart = ijk_patch_indices[i][2]
        jend = ijk_patch_indices[i][3]
        kstart = ijk_patch_indices[i][4]
        kend = ijk_patch_indices[i][5]
        pre[..., istart:iend, jstart:jend, kstart:kend] += output

    pre = torch.squeeze(torch.argmax(pre, dim=1))
    pre[np.where(pre == 2)] = 4
    pre[np.where(pre == 1)] = 2
    pre[np.where(pre == 3)] = 1

    pre_preprocess = preprocess_label(pre.numpy())
    label_preprocess = preprocess_label(label)

    pre_preprocess = torch.tensor(pre_preprocess)
    label_preprocess = torch.tensor(label_preprocess)

    pre_preprocess = torch.unsqueeze(pre_preprocess, 0)
    label_preprocess = torch.unsqueeze(label_preprocess, 0)





    back_pred, wt_pred, tc_pred, et_pred = combine_labels(pre_preprocess)
    back_truth, wt_truth, tc_truth, et_truth = combine_labels(label_preprocess)

    wt_pred, tc_pred, et_pred = wt_pred.squeeze().numpy(), tc_pred.squeeze().numpy(), et_pred.squeeze().numpy()
    wt_truth, tc_truth, et_truth = wt_truth.squeeze().numpy(), tc_truth.squeeze().numpy(), et_truth.squeeze().numpy()

    metrics = {}

    pre_preprocess_metric = np.concatenate(
        (wt_pred[..., np.newaxis], tc_pred[..., np.newaxis], et_pred[..., np.newaxis]), axis=3)
    label_preprocess_metric = np.concatenate(
        (wt_truth[..., np.newaxis], tc_truth[..., np.newaxis], et_truth[..., np.newaxis]), axis=3)

    # cmatrix = ConfusionMatrix(pre_preprocess_metric, label_preprocess_metric)
    #
    # tp = get_TP(pre_preprocess_metric, label_preprocess_metric, cmatrix)
    # tn = get_TN(pre_preprocess_metric, label_preprocess_metric, cmatrix)
    # fp = get_FP(pre_preprocess_metric, label_preprocess_metric, cmatrix)
    # fn = get_FN(pre_preprocess_metric, label_preprocess_metric, cmatrix)

    # dsc = DiceCoefficient(pre_preprocess_metric, label_preprocess_metric)
    # metrics['dsc'] = dsc
    # jc = JaccardCoefficient(pre_preprocess_metric, label_preprocess_metric)
    # metrics['jac'] = jc
    # sen = Recall(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['sen'] = sen
    # spe = Specificity(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['spe'] = spe
    # fpr = FPR(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['fpr'] = fpr
    # fnr = FNR(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['fnr'] = fnr
    # ppv = Precision(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['ppv'] = ppv
    # f1 = F1Score(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['f1'] = f1
    # gce = GlobalConsistencyError(pre_preprocess_metric, label_preprocess_metric)
    # metrics['gce'] = gce
    # vd = VolumetricDistance(pre_preprocess_metric, label_preprocess_metric)
    # metrics['vd'] = vd
    # vs = VolumtricSimilarity(pre_preprocess_metric, label_preprocess_metric)
    # metrics['vs'] = vs
    # mi = MutualInformation(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['mi'] = mi
    # voi = VariationofInformation(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp, fn)
    # metrics['voi'] = voi
    # icc = InterclassCorrelation(pre_preprocess_metric, label_preprocess_metric)
    # metrics['icc'] = icc
    # pbd = ProbabilisticDistance(pre_preprocess_metric, label_preprocess_metric)
    # metrics['pbd'] = pbd
    # kappa = Kappa(pre_preprocess_metric, label_preprocess_metric, tp, tn, fp ,fn)
    # metrics['kappa'] = kappa

    for i in range(3):
        # cmatrix = ConfusionMatrix(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # tp = get_TP(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), cmatrix)
        # tn = get_TN(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), cmatrix)
        # fp = get_FP(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), cmatrix)
        # fn = get_FN(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), cmatrix)

        # dsc = DiceCoefficient(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # string_dice = 'dsc' + str(i+1)
        # metrics[string_dice] = dsc
        # jc = JaccardCoefficient(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # string_jac = 'jac' + str(i+1)
        # metrics[string_jac] = jc
        # sen = Recall(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_sen = 'sen' + str(i+1)
        # metrics[string_sen] = sen
        # spe = Specificity(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_spe = 'spe' + str(i + 1)
        # metrics[string_spe] = spe
        # fpr = FPR(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_fpr = 'fpr' + str(i + 1)
        # metrics[string_fpr] = fpr
        # fnr = FNR(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_fnr = 'fnr' + str(i + 1)
        # metrics[string_fnr] = fnr
        # ppv = Precision(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_ppv = 'ppv' + str(i + 1)
        # metrics[string_ppv] = ppv
        # f1 = F1Score(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_f1 = 'f1' + str(i + 1)
        # metrics[string_f1] = f1
        # gce = GlobalConsistencyError(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # string_gce = 'gce' + str(i + 1)
        # metrics[string_gce] = gce
        # vd = VolumetricDistance(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # string_vd = 'vd' + str(i + 1)
        # metrics[string_vd] = vd
        vs = VolumtricSimilarity(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        string_vs = 'vs' + str(i + 1)
        metrics[string_vs] = vs[1]
        # mi = MutualInformation(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_mi = 'mi' + str(i + 1)
        # metrics[string_mi] = mi
        # voi = VariationofInformation(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_voi = 'voi' + str(i + 1)
        # metrics[string_voi] = voi
        # icc = InterclassCorrelation(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # string_icc = 'icc' + str(i + 1)
        # metrics[string_icc] = icc
        # pbd = ProbabilisticDistance(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2))
        # string_pbd = 'pbd' + str(i + 1)
        # metrics[string_pbd] = pbd
        # kappa = Kappa(one_hot(pre_preprocess_metric[...,i],2),one_hot(label_preprocess_metric[...,i],2), tp, tn, fp, fn)
        # string_kappa = 'kappa' + str(i + 1)
        # metrics[string_kappa] = kappa



    dice0, dice1, dice2, dice3 = dice_pre(pre_preprocess, label_preprocess)
    metrics_list = metrics_pre(pre_preprocess, label_preprocess)

    metrics.update(metrics_list)

    return pre, dice0, dice1, dice2, dice3, metrics


def inference_all_multiclass(savetxtname, model, image_list, resample, resolution, patch_size_x, patch_size_y,
                             patch_size_z,
                             stride_inplane, stride_layer, batch_size, segmentation):
    flair = (image_list["flair"])
    t1 = (image_list["t1"])
    t1ce = (image_list["t1ce"])
    t2 = (image_list["t2"])
    label = (image_list["label"])

    a = (flair.split('/')[-2])  # dgx

    if not os.path.isdir('./result/results_' + savetxtname):
        os.mkdir('./result/results_' + savetxtname)

    label_directory = os.path.join(str('./result/results_'+ savetxtname +'/results_' + a + '.nii'))

    result, dice0, dice1, dice2, dice3, metrics_list = inference_multiclass(model, flair, t1, t1ce, t2, label,
                                                                            './prova.nii', resample, resolution,
                                                                            patch_size_x,
                                                                            patch_size_y, patch_size_z, stride_inplane,
                                                                            stride_layer, batch_size,
                                                                            segmentation=segmentation)

    debugmask = result.numpy().astype(np.float64)
    new_mask = nib.Nifti1Image(debugmask, np.eye(4))
    nib.save(new_mask, label_directory)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), label_directory))


    info_line_dice = a + "| Dice1 score:{:.4f} | Dice2 score:{:.4f} | Dice3 score:{:.4f} | ".format(dice1, dice2, dice3)
    info_line_sen = a + "| Sen1 :{:.4f} | Sen2 : {:.4f} | Sen3 : {:.4f} |".format(metrics_list['SEN1'],
                                                                                  metrics_list['SEN2'],
                                                                                  metrics_list['SEN3'])
    info_line_PPV = a + "| PPV1 :{:.4f} | PPV2 : {:.4f} | PPV3 : {:.4f} |".format(metrics_list['PPV1'],
                                                                                  metrics_list['PPV2'],
                                                                                  metrics_list['PPV3'])
    info_line_HD = a + "| HD1 :{:.4f} | HD2 : {:.4f} | HD3 : {:.4f} |".format(metrics_list['HD1'],
                                                                              metrics_list['HD2'],
                                                                              metrics_list['HD3'])
    info_line_ASSD = a + "| ASSD1 :{:.4f} | ASSD2 : {:.4f} | ASSD3 : {:.4f} |".format(metrics_list['ASSD1'],
                                                                                      metrics_list['ASSD2'],
                                                                                      metrics_list['ASSD3'])
    info_line_VS = a + "| vs1 :{:.4f} | vs2 : {:.4f} | vs3 : {:.4f} |".format(metrics_list['vs1'],
                                                                                      metrics_list['vs2'],
                                                                                      metrics_list['vs3'])

    print(info_line_dice)
    print(info_line_sen)
    print(info_line_PPV)
    print(info_line_HD)
    print(info_line_ASSD)
    print(info_line_VS)
    check_dir(os.path.join('./result/results_' + savetxtname))
    open(os.path.join(('./result/results_' + savetxtname), 'result.txt'), 'a').write(
        info_line_dice + '\n')
    open(os.path.join(('./result/results_' + savetxtname), 'result.txt'), 'a').write(
        info_line_sen + '\n')
    open(os.path.join(('./result/results_'+ savetxtname), 'result.txt'), 'a').write(
        info_line_PPV + '\n')
    open(os.path.join(('./result/results_'+ savetxtname), 'result.txt'), 'a').write(
        info_line_HD + '\n')
    open(os.path.join(('./result/results_'+ savetxtname), 'result.txt'), 'a').write(
        info_line_ASSD + '\n')
    open(os.path.join(('./result/results_' + savetxtname), 'result.txt'), 'a').write(
        info_line_VS + '\n')
    print('************* Next image coming... *************')

    return dice0, dice1, dice2, dice3, metrics_list


def check_accuracy_model_multiclass(savetxtname, model, images_list, resample, new_resolution, patch_size_x, patch_size_y,
                                    patch_size_z, stride_inplane, stride_layer):
    np_dice0 = []
    np_dice1 = []
    np_dice2 = []
    np_dice3 = []
    np_sen1 = []
    np_sen2 = []
    np_sen3 = []
    np_ppv1 = []
    np_ppv2 = []
    np_ppv3 = []
    np_hd1 = []
    np_hd2 = []
    np_hd3 = []
    np_assd1 = []
    np_assd2 = []
    np_assd3 = []
    np_vs1 = []
    np_vs2 = []
    np_vs3 = []

    check_dir(os.path.join('./result/results_' + savetxtname))
    save_file = os.path.join('./result/results_' + savetxtname + '/save_metric.pkl')

    print("0/%i (0%%)" % len(images_list))
    for i in tqdm(range(len(images_list))):

        label_path = (images_list[i]["label"])
        label = nib.load(label_path).get_fdata()
        if label[np.where(label == 2)].shape[0] < 200:
            continue

        Np_dice0, Np_dice1, Np_dice2, Np_dice3, metrics_list = inference_all_multiclass(savetxtname=savetxtname,
                                                                                        model=model,
                                                                                        image_list=images_list[i],
                                                                                        resample=resample,
                                                                                        resolution=new_resolution,
                                                                                        patch_size_x=patch_size_x,
                                                                                        patch_size_y=patch_size_y,
                                                                                        patch_size_z=patch_size_z,
                                                                                        stride_inplane=stride_inplane,
                                                                                        stride_layer=stride_layer,
                                                                                        batch_size=1,
                                                                                        segmentation=True)

        np_dice0.append(Np_dice0)
        np_dice1.append(Np_dice1)
        np_dice2.append(Np_dice2)
        np_dice3.append(Np_dice3)

        np_sen1.append(metrics_list['SEN1'])
        np_sen2.append(metrics_list['SEN2'])
        np_sen3.append(metrics_list['SEN3'])

        np_ppv1.append(metrics_list['PPV1'])
        np_ppv2.append(metrics_list['PPV2'])
        np_ppv3.append(metrics_list['PPV3'])

        np_hd1.append(metrics_list['HD1'])
        np_hd2.append(metrics_list['HD2'])
        if metrics_list['HD3'] !=0:
            np_hd3.append(metrics_list['HD3'])

        np_assd1.append(metrics_list['ASSD1'])
        np_assd2.append(metrics_list['ASSD2'])
        if metrics_list['ASSD3'] != 0:
            np_assd3.append(metrics_list['ASSD3'])
        np_vs1.append(metrics_list['vs1'])
        np_vs2.append(metrics_list['vs2'])
        np_vs3.append(metrics_list['vs3'])

    np_dice0 = np.array(np_dice0)
    np_dice1 = np.array(np_dice1)
    np_dice2 = np.array(np_dice2)
    np_dice3 = np.array(np_dice3)

    np_sen1 = np.array(np_sen1)
    np_sen2 = np.array(np_sen2)
    np_sen3 = np.array(np_sen3)

    np_ppv1 = np.array(np_ppv1)
    np_ppv2 = np.array(np_ppv2)
    np_ppv3 = np.array(np_ppv3)

    np_hd1 = np.array(np_hd1)
    np_hd2 = np.array(np_hd2)
    np_hd3 = np.array(np_hd3)

    np_assd1 = np.array(np_assd1)
    np_assd2 = np.array(np_assd2)
    np_assd3 = np.array(np_assd3)

    np_vs1 = np.array(np_vs1)
    np_vs2 = np.array(np_vs2)
    np_vs3 = np.array(np_vs3)

    print("net: " + savetxtname)

    print('Mean volumetric Dice0:', np_dice0.mean())
    print('Mean volumetric Dice1:', np_dice1.mean())
    print('Mean volumetric Dice2:', np_dice2.mean())
    print('Mean volumetric Dice3:', np_dice3.mean())

    print('Mean volumetric Sen1:', np_sen1.mean())
    print('Mean volumetric Sen2:', np_sen2.mean())
    print('Mean volumetric Sen3:', np_sen3.mean())

    print('Mean volumetric PPV1:', np_ppv1.mean())
    print('Mean volumetric PPV2:', np_ppv2.mean())
    print('Mean volumetric PPV3:', np_ppv3.mean())

    print('Mean volumetric HD1:', np_hd1.mean())
    print('Mean volumetric HD2:', np_hd2.mean())
    print('Mean volumetric HD3:', np_hd3.mean())

    print('Mean volumetric ASSD1:', np_assd1.mean())
    print('Mean volumetric ASSD2:', np_assd2.mean())
    print('Mean volumetric ASSD3:', np_assd3.mean())

    print('Mean volumetric vs1:', np_vs1.mean())
    print('Mean volumetric vs2:', np_vs2.mean())
    print('Mean volumetric vs3:', np_vs3.mean())

    save_metric = [np_dice1.mean(), np_dice2.mean(),np_dice3.mean(),
                   np_sen1.mean(),np_sen2.mean(),np_sen3.mean(),
                   np_ppv1.mean(),np_ppv2.mean(),np_ppv3.mean(),
                   np_hd1.mean(),np_hd2.mean(),np_hd3.mean(),
                   np_assd1.mean(),np_assd2.mean(),np_assd3.mean(),
                   np_vs1.mean(),np_vs2.mean(),np_vs3.mean()
                   ]

    save_metric_array = np.array(save_metric)

    with open(save_file, 'wb') as f:
        pickle.dump(save_metric_array, f)

    # 加载：
    # with open('my_array.pkl', 'rb') as f:
    #     loaded_array = pickle.load(f)

    return np_dice0.mean(), np_dice1.mean(), np_dice2.mean(), np_dice3.mean()
