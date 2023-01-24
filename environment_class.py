import numpy as np
import sys
import random
import math
#import lmdb
from io import BytesIO
import h5py
import matplotlib
import scipy.misc as smi
import scipy.io as sio
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import PIL.Image as Image



class coding_env(object):
    def __init__(self, start, end):
        self.n_actions = 51 - 22 + 1
        self.n_features = 64 * 64 * 2 + 15
        self.start = start
        self.end = end
        #self.bit_ratio = np.load("bit-ratio.npy")
        # self.instance_all = np.zeros((51 - 22 + 1, 2401, 18, 18))
        self.bpp_all = np.zeros((51 - 22 + 1, 2401, 81))
        # self.mse_all = np.zeros((51 - 22 + 1, 2401, 81))
        self.class_idx = np.load('class_index.npy')
        for i in range(51 - 22 + 1):
            self.bpp_all[i, 1:2401, :] = sio.loadmat('/data/lixin/imagenet/imageNet_64/imageNet_info/imageNet_QP' + str(i + 22) + '_bpp.mat')['bpp'][:2400]
            # self.mse_all[i, 1:2401, :] = sio.loadmat('D:/Bit Allocation/lixin/imagenet/imageNet_64/imageNet_info/imageNet_QP' + str(i + 22) + '_mse.mat')['mse'][:2400]
            # self.bpp_all[i, 1:2401, :] = np.transpose(h5py.File('D:/Bit Allocation/lixin/imageNet/imageNet_info/imageNet_QP' + str(i + 22) + '_bpp.mat')['bpp'])[:2400]
            # self.mse_all[i, 1:2401, :] = np.transpose(h5py.File('D:/Bit Allocation/lixin/imageNet/imageNet_info/imageNet_QP' + str(i + 22) + '_mse.mat')['mse'])[:2400]

    def reset(self, is_train, idx=None):
        if is_train:
            self.index = random.randint(self.start, self.end)
            # print self.index
        else:
            self.index = idx
        self.index = self.class_idx[self.index] #dataset shuffle 
        self.reward_before = 0
        load_path_image = '/data/lixin/imagenet/large_resolution/' + "%05d" % self.index + '.png'
        load_path_mask = '/data/lixin/imagenet/imageNet_mask/CAM_22/' + str(self.index) + '.png'

        self.image = (smi.imread(load_path_image)).astype('uint8')
        self.image = np.array(Image.fromarray(self.image).resize((576, 576)).convert('L'))[:, :, np.newaxis]
        self.mask = np.array(Image.fromarray((smi.imread(load_path_mask)).astype('uint8')).resize((576, 576)))[:, :, np.newaxis]
        self.height = int(self.mask.shape[0] // 64)
        self.width = int(self.mask.shape[1] // 64)
        self.mask_used = np.zeros((51 - 22 + 1, int(self.height * 64), int(self.width * 64)))
        for i in range(51 - 22 + 1):
            self.mask_used[i, :, :] = np.array(Image.fromarray(smi.imread(
                '/data/lixin/imagenet/imageNet_mask/CAM_' + str(i + 22) + '/' + str(self.index) + '.png')).resize((576, 576)))

        self.action_list = np.zeros((self.height, self.width))
        self.distortion_list = np.zeros((self.height, self.width))
        self.ratio_list = np.zeros((self.height, self.width))
        self.seq_all = self.height * self.width
        self.mask_ratio = np.average(self.mask) / 255.0
        self.seq = 0
        self.current_mask = np.reshape(self.mask[int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
                                       int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64, 0],
                                       (64, 64, 1))
        self.current_image = self.image[int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
                             int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64, :]
        self.current_ratio = np.average(self.current_mask) / 255.0
        self.ratio_list[int(self.seq / self.width), int(self.seq % self.width)] = self.current_ratio
        for z in range(int(self.seq_all)):
            self.ratio_list[int(z / self.width), int(z % self.width)] = np.average(
                self.mask[int(z / self.width) * 64:int(z / self.width + 1) * 64,
                int(z % self.width) * 64: int(z % self.width + 1) * 64, 0]) / 255.0
        down_num = 0
        up_num = 0
        left_num = 0
        right_num = 0
        if self.seq // self.width == self.height - 1:
            down_ratio = 0.
            down_num = 0.
        else:
            down_ratio = self.ratio_list[int(self.seq / self.width) + 1, self.seq % self.width]
            if down_ratio > 0:
                down_num = 1

        if self.seq // self.width == 0:
            up_ratio = 0.
            up_num = 0.
        else:
            up_ratio = self.ratio_list[int(self.seq / self.width) - 1, self.seq % self.width]
            if up_ratio > 0:
                up_num = 1

        if self.seq % self.width == 0:
            left_ratio = 0.
            left_num = 0.
        else:
            left_ratio = self.ratio_list[int(self.seq / self.width), self.seq % self.width - 1]
            if left_ratio > 0:
                left_num = 1

        if self.seq % self.width == self.width - 1:
            right_ratio = 0.
            right_num = 0.
        else:
            right_ratio = self.ratio_list[int(self.seq / self.width), self.seq % self.width + 1]
            if right_ratio > 0:
                right_num = 1
        if self.current_ratio > 0:
            current_num = 1
        else:
            current_num = 0
        up_qp = 0
        left_qp = 0

        return np.concatenate((np.ndarray.flatten(np.concatenate((self.current_image, self.current_mask), axis=2)),
                               np.array([up_num, down_num, left_num, right_num, current_num, up_ratio, down_ratio, left_ratio, right_ratio, self.mask_ratio, self.current_ratio, up_qp, left_qp, self.seq,
                                         self.seq_all])))

    def step(self, action):
        self.current_mask = np.reshape(self.mask[int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
                                       int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64, 0],
                                       (64, 64, 1))
        self.current_image = self.image[int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
                             int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64, :]
        self.current_ratio = np.average(self.current_mask) / 255.0
        self.mask_reward = np.reshape(
            self.mask_used[action, int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
            int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64], (64, 64, 1))

        distortion = len(self.current_mask[(self.current_mask - self.mask_reward) > 75])/16
        self.distortion_list[int(self.seq / self.width), int(self.seq % self.width)] = distortion
        action_reward = self.bpp_all[0, self.index, self.seq] - self.bpp_all[action, self.index, self.seq]
        self.action_list[int(self.seq / self.width), int(self.seq % self.width)] = action_reward

        reward = action_reward - 0.1*distortion
        self.seq += 1
        if self.seq == self.seq_all:
            observation = np.zeros((64 * 64 * 2 + 15,))
            done = True
            info = [np.sum(self.distortion_list), np.sum(self.action_list)]

        else:
            done = False
            self.current_mask = np.reshape(
                self.mask[int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
                int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64, 0], (64, 64, 1))
            self.current_image = self.image[int(self.seq / self.width) * 64:int(self.seq / self.width + 1) * 64,
                                 int(self.seq % self.width) * 64: int(self.seq % self.width + 1) * 64]
            self.current_ratio = np.average(self.current_mask) / 255.0

            if self.seq // self.width == self.height - 1:
                down_ratio = 0.
                down_num = 0.
            else:
                down_ratio = self.ratio_list[int(self.seq / self.width) + 1, self.seq % self.width]
                if down_ratio > 0:
                    down_num = 1
                else:
                    down_num = 0

            if self.seq // self.width == 0:
                up_ratio = 0.
                up_num = 0.
                up_qp = 0
            else:
                up_ratio = self.ratio_list[int(self.seq / self.width) - 1, self.seq % self.width]
                if down_ratio > 0:
                    up_num = 1
                else:
                    up_num = 0
                up_qp = self.action_list[int(self.seq / self.width) - 1, self.seq % self.width]

            if self.seq % self.width == 0:
                left_ratio = 0.
                left_num = 0.
                left_qp = 0
            else:
                left_ratio = self.ratio_list[int(self.seq / self.width), self.seq % self.width - 1]
                if down_ratio > 0:
                    left_num = 1
                else:
                    left_num = 0
                left_qp = self.action_list[int(self.seq / self.width), self.seq % self.width - 1]

            if self.seq % self.width == self.width - 1:
                right_ratio = 0.
                right_num = 0.
            else:
                right_ratio = self.ratio_list[int(self.seq / self.width), self.seq % self.width + 1]
                if down_ratio > 0:
                    right_num = 1
                else:
                    right_num = 0
            if self.current_ratio > 0:
                current_num = 1
            else:
                current_num = 0
            observation = np.concatenate((np.ndarray.flatten(np.concatenate((self.current_image, self.current_mask), axis=2)),
                               np.array([up_num, down_num, left_num, right_num, current_num, up_ratio, down_ratio, left_ratio, right_ratio, self.mask_ratio, self.current_ratio, up_qp, left_qp, self.seq,
                                         self.seq_all])))
            info = None
        return observation, reward, done, info


if __name__ == '__main__':
    print("environment for coding")



