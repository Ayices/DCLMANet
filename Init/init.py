class InitParser(object):
    def __init__(self):

        self.do_you_wanna_train = False                                            # 'Training will start'
        self.do_you_wanna_load_epoch = False                                       # 'Load old model'
        self.do_you_wanna_check_accuracy = True                                    # 'Model will be tested after the training or only this is done'


        # gpu setting
        self.multi_gpu = True                                                         # 'Decide to use one or more GPUs'
        self.gpu_id = '1'                                                           # 'Select the GPUs for training and testing'
        # optimizer setting

        self.lr = 0.001
        # 'Learning rate'
        self.weight_decay = 2e-5                                                         # 'Weight decay'

        self.lr_self = 1e-3
        self.min_lr = 1e-5

        self.lr_reduce_factor = 0.7
        self.lr_reduce_patience = 10

        self.amsgrad = True
        self.resample = True                                                          # 'Decide or not to rescale the images to a new resolution'
        self.new_resolution = [128, 128, 128]
        # self.new_resolution = (1, 1, 1)
        self.patch_size = [128, 128, 128]                                             # "Input dimension for the Unet3D"
        self.drop_ratio = 0                                                           # "Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1"
        self.min_pixel = 0.1                                                          # "Percentage of minimum non-zero pixels in the cropped label"
        self.batch_size = 2                                                           # 'Batch size: the greater more GPUs are used and faster is the training'
        self.num_epoch = 200                                                          # "Number of epochs"
        self.init_epoch = 1
        self.stride_inplane = 32                                                      # "Stride size in 2D plane"
        self.stride_layer = 32                                                        # "Stride size in z direction"

        self.K_fold = 1

        self.criterion = 'softmax_dice'

        # path setting
        self.train_path = '"/remote-home/MRISegmentation/Data_folder/train_2020"'         # Training data folder
        self.val_path = '"/remote-home/MRISegmentation/Data_folder/val_2020"'             # Validation data folder

        self.history_dir = './History'
        self.load_path = \
            "./Best_Dice3_0.79_201epoch.pth.gz"
        if self.do_you_wanna_check_accuracy == True:
            print(self.load_path)

        self.output_path = "./History_Second/"
        self.output_path_1 = "./History_One_weitiao/"



