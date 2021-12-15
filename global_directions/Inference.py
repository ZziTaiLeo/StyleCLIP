import os
import sys

sys.path.append('/home/ming13/editor/pycharm-2021.1.1/workplace/styleClip/StyleCLIP/global_directions')
from manipulate import Manipulator
import tensorflow as tf
import numpy as np
import torch
from torch.nn import DataParallel
import clip
from MapTS import GetBoundary, GetDt


class StyleCLIP():

    def __init__(self, dataset_name='ffhq'):
        print('load clip')
        # 分配设备
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        # 读取模型
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        #self.model = DataParallel(self.model, [0, 1])
        self.LoadData(dataset_name)

    def LoadData(self, dataset_name):
        tf.keras.backend.clear_session()
        M = Manipulator(dataset_name=dataset_name)
        np.set_printoptions(suppress=True)
        fs3 = np.load('./npy/' + dataset_name + '/fs3.npy')

        self.M = M
        self.fs3 = fs3
        # 加载e4e data
        w_plus = torch.load(r'./data/ffhq/latents.pt').cpu().numpy()
        # w_plus = np.load('./data/' + dataset_name + '/w_plus.npy')
        # W to S空间
        self.M.dlatents = M.W2S(w_plus)
        # 保存S空间的数据
        # save_S_space = r'./result/lips/'
        # if not os.path.exists(save_S_space):
        #     os.mkdir(save_S_space)
        # torch.save(self.M.dlatents, save_S_space+'S_space.pt')
        # channels通道数
        if dataset_name == 'ffhq':
            self.c_threshold = 20
        else:
            self.c_threshold = 100
        self.SetInitP()

    # 初始化
    def SetInitP(self):
        self.M.alpha = [3]
        self.M.num_images = 1

        self.target = ''
        self.neutral = ''
        self.GetDt2()
        img_index = 0
        # 操作latents的每一张图片的张量
        self.M.dlatent_tmp = [tmp[img_index:(img_index + 1)] for tmp in self.M.dlatents]

    def GetDt2(self):
        classnames = [self.target, self.neutral]
        # dt = delta_t
        dt = GetDt(classnames, self.model)

        self.dt = dt
        num_cs = []
        # numpy.arange(start, stop, step, dtype = None) 这里有20个值
        betas = np.arange(0.1, 0.3, 0.01)
        for i in range(len(betas)):
            boundary_tmp2, num_c = GetBoundary(self.fs3, self.dt, self.M, threshold=betas[i])
            print('beta %i :' % i, betas[i])
            num_cs.append(num_c)

        num_cs = np.array(num_cs)
        select = num_cs > self.c_threshold

        if sum(select) == 0:
            self.beta = 0.1
        else:
            self.beta = betas[select][-1]

    def GetCode(self):
        boundary_tmp2, num_c = GetBoundary(self.fs3, self.dt, self.M, threshold=self.beta)
        codes = self.M.MSCode(self.M.dlatent_tmp, boundary_tmp2)
        return codes

    def GetImg(self):

        codes = self.GetCode()
        out = self.M.GenerateImg(codes)
        img = out[0, 0]
        return img

    def GetOriImg(self, latent):
        out = self.M.GenerateImg(latent)
        img = out[0, 0]
        return img


# %%
if __name__ == "__main__":
    style_clip = StyleCLIP()
    self = style_clip
