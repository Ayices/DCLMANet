# Paper Name

Glioma segmentation based on dense contrastive learning and multimodal features recalibration

# Paper Abstract

Accurate segmentation of different regions of gliomas from multimodal magnetic resonance (MR) images is crucial for glioma grading and precise diagnosis, but many existing segmentation methods are difficult to effectively utilize multimodal MR image information to recognize accurately the lesion regions with small size, low contrast and irregular shape. To address this issue, this work proposes a novel 3D glioma segmentation model DCL-MANet. DCL-MANet has an architecture of multiple encoders and one single decoder. Each encoder is used to extract MR image features of a given modality. To overcome the entangle problems of multimodal semantic features, a dense contrastive learning (DCL) strategy is presented to extract the modality-specific and common features. Following that, feature recalibration block (RFB) based on modality-wise attention is used to recalibrate the semantic features of each modality, enabling the model to focus on the features that are beneficial for glioma segmentation. These recalibrated features are input into the decoder to obtain the segmentation results. To verify the superiority of the proposed method, we compare it with several state-of-the-art (SOTA) methods in terms of Dice, average symmetric surface distance (ASSD), HD95 and volumetric similarity (Vs). The comparison results show that the average Dice, ASSD, HD95 and Vs of DCL-MANet on all tumor regions are improved at least by 0.66%, 3.47%, 8.94% and 1.07% respectively. For small enhance tumor (ET) region, the corresponding improvement can be up to 0.37%, 7.83%, 11.32%, and 1.35%, respectively. In addition, the ablation results demonstrate the effectiveness of the proposed DCL and RFB, and combining them can significantly increase Dice (1.59%) and Vs (1.54%) while decreasing ASSD (40.51%) and HD95 (45.16%) on ET region. The proposed DCL-MANet could disentangle multimodal features and enhance the semantics of modality-dependent features, providing a potential means to accurately segment small lesion regions in gliomas.

# Paper Link

The corresponding paper for this code is [here](https://iopscience.iop.org/article/10.1088/1361-6560/ad387f/meta).


