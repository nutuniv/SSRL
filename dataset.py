import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.patch_mode = args.patch_mode
        self.data_dir = args.data_dir
        self.num_segments = args.num_segments
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-test.list'
            else:
                self.rgb_list_file = 'list/shanghai-train.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/ucf-test.list'
            else:
                self.rgb_list_file = 'list/ucf-train.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        self.patch_list = []
        data_root = self.data_dir
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    
                else:
                    self.list = self.list[:63]
                    
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'SH_Train_i3d/' + self.list[i]
                    self.patch_list.append(self.list[i].replace('SH_Train_i3d','SH_PATCH35_i3d').replace('_i3d','_patch35_i3d'))


            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                else:
                    self.list = self.list[:810]
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'UCF_Train_i3d/' + self.list[i]
                    self.patch_list.append(self.list[i].replace('UCF_Train_i3d', 'UCF_Train_patch47_i3d_32').replace('_i3d','_patch47_i3d'))
        else:
            if self.dataset == 'shanghai':
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'SH_Test_i3d/' + self.list[i]
                    self.patch_list.append(self.list[i].replace('SH_Test_i3d','SH_PATCH35_i3d').replace('_i3d','_patch35_i3d'))

            elif self.dataset == 'ucf':
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'UCF_Test_i3d' + self.list[i]
                    self.patch_list.append(self.list[i].replace('UCF_Test_i3d', 'UCF_Test_patch47_i3d').replace('_i3d','_patch47_i3d'))

    def __getitem__(self, index):
        label = self.get_label() # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.patch_mode:
            patch_features = np.load(self.patch_list[index].strip('\n'), allow_pickle=True)
            patch_features = np.array(patch_features, dtype=np.float32)
            if self.dataset == 'shanghai':
                lastfeat = patch_features[-1][np.newaxis, :, :]
                if features.shape[0] > patch_features.shape[0]:
                    while features.shape[0] > patch_features.shape[0]:
                        patch_features = np.concatenate((patch_features, lastfeat), axis=0)
                elif features.shape[0] < patch_features.shape[0]:
                    patch_features = patch_features[:features.shape[0]]
            if self.test_mode or self.dataset == 'shanghai':
                features = np.concatenate((features, patch_features), axis=1)
            else:
                features = np.concatenate((features, patch_features), axis=0)


        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features,index
        else:
            if self.dataset == 'shanghai':
            #process 10-cropped snippet feature
               features = features.transpose(1, 0, 2)  # [10, B, T, F]
               divided_features = []
               for feature in features:
                   feature = process_feat(feature, self.num_segments)  # divide a video into 32 segments
                   divided_features.append(feature)
               divided_features = np.array(divided_features, dtype=np.float32)
            else:
               divided_features = features

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame



class Multi_Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.multi_patch_mode = args.multi_patch_mode
        self.multi_patch_size = args.multi_patch_size
        self.data_dir = args.data_dir
        self.num_segments = args.num_segments
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-test.list'
            else:
                self.rgb_list_file = 'list/shanghai-train.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/ucf-test.list'
            else:
                self.rgb_list_file = 'list/ucf-train.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        self.patch_list = {}
        data_root = self.data_dir
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                else:
                    self.list = self.list[:63]
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'SH_Train_i3d/' + self.list[i]
                for size in self.multi_patch_size:
                    self.patch_list[str(size)] = [data_root + 'SH_PATCH'+str(size)+'_i3d/' + video_name.split('/')[-1].replace('_i3d','_patch'+str(size)+'_i3d') for video_name in self.list]


            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                else:
                    self.list = self.list[:810]
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'UCF_Train_i3d_32/' + self.list[i]
                for size in self.multi_patch_size:
                    self.patch_list[str(size)] = [video_name.replace('UCF_Train_i3d','UCF_Train_patch'+str(size)+'_i3d').replace('_i3d','_patch'+str(size)+'_i3d') for video_name in self.list]

        else:
            if self.dataset == 'shanghai':
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'SH_Test_i3d/' + self.list[i]
                for size in self.multi_patch_size:
                    self.patch_list[str(size)] = [data_root + 'SH_PATCH'+str(size)+'_i3d/' + video_name.split('/')[-1].replace('_i3d','_patch'+str(size)+'_i3d') for video_name in self.list]


            elif self.dataset == 'ucf':
                for i in range(len(self.list)):
                    self.list[i] = data_root + 'UCF_Test_i3d/' + self.list[i]
                for size in self.multi_patch_size:
                    self.patch_list[str(size)] = [video_name.replace('UCF_Test_i3d','UCF_Test_patch' + str(size) +'_i3d').replace('_i3d','_patch' + str(size) + '_i3d') for video_name in self.list]

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.multi_patch_mode:
            patch_feat_all = []
            for key in self.patch_list.keys():
                feat = np.load(self.patch_list[key][index].strip('\n'), allow_pickle=True)
                patch_feat = np.array(feat,dtype=np.float32)
                if self.dataset == 'shanghai':
                    lastfeat = patch_feat[-1][np.newaxis, :, :]
                    while features.shape[0] > patch_feat.shape[0]:
                        patch_feat = np.concatenate((patch_feat, lastfeat), axis=0)
                    if features.shape[0] < patch_feat.shape[0]:
                        patch_feat = patch_feat[:features.shape[0]]
                patch_feat_all.append(patch_feat)
            if self.test_mode or self.dataset == 'shanghai':
                patch_feat_all = np.concatenate(patch_feat_all, axis=1)
                features = np.concatenate((features, patch_feat_all), axis=1)
            else:
                patch_feat_all = np.concatenate(patch_feat_all, axis=0)
                features = np.concatenate((features, patch_feat_all), axis=0)


        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features, index
        else:
            if self.dataset == 'shanghai':
                features = features.transpose(1, 0, 2)  # [10, B, T, F]
                divided_features = []
                for feature in features:
                    feature = process_feat(feature, self.num_segments)  # divide a video into 32 segments
                    divided_features.append(feature)
                divided_features = np.array(divided_features, dtype=np.float32)
            else:
                divided_features = features

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame

