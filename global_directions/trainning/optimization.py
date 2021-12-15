import argparse
import math
import os
import sys

import PIL.Image
import torchvision
from PIL import ImageTk
from PIL import Image, ImageTk
from torch import optim
from tqdm import tqdm
import torch
import numpy as np
import sys
import torchvision.transforms as transforms

sys.path.append('/home/ming13/editor/pycharm-2021.1.1/workplace/styleClip/StyleCLIP/')
from global_directions.Inference import StyleCLIP


class optimization():
    def __init__(self, args):
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        self.style_clip = StyleCLIP(dataset_name='ffhq')
        self.style_clip.neutral = args.set_neutral
        self.style_clip.target = args.set_target
        self.alpha = args.set_alpha
        self.beta = args.set_beta
        latent_code_init = torch.load(args.latents_dir).cuda()
        latent = latent_code_init.detach().clone()
        latent.requires_grad = True
        # 读取fs3文件
        self.fs3 = np.load(
            '/home/ming13/editor/pycharm-2021.1.1/workplace/styleClip/StyleCLIP/global_directions/npy/ffhq/fs3.npy')
        # 优化后图片文件夹位置
        save_file = os.path.join(args.save_dir, args.save_file)
        os.makedirs(save_file, exist_ok=True)
        # 分配设备
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # 优化器
        alpha = self.alpha
        beta = self.beta
        super_param = torch.tensor([alpha, beta])
        optimizer = optim.Adam([super_param], lr=args.lr)
        # for idx in range(len(latent_code_init) - 1):
        for idx in range(1, 2):
            # 步数
            pbar = tqdm(range(args.step))  # 默认300
            loss = 0
            img1_gen_from_w = self.style_clip.preprocess(
                PIL.Image.open(r'./inference_inversion/{:05d}.jpg'.format(idx))).unsqueeze(0).to(device)
            img2_gen_from_w = self.style_clip.preprocess(
                PIL.Image.open(r'./inference_inversion/{:05d}.jpg'.format(idx + 1))).unsqueeze(0).to(device)
            img1_gen_from_s = self.style_clip.preprocess(
                Image.fromarray(self.s_to_img(alpha=alpha, beta=beta, index=idx))).unsqueeze(0).to(device)
            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]['lr'] = lr
                img2_gen_from_s = self.style_clip.preprocess(
                    Image.fromarray(self.s_to_img(alpha=alpha, beta=beta, index=idx + 1))).unsqueeze(0).to(device)

                loss += compute_loss(img1_gen_from_w, img2_gen_from_w,
                                     img1_gen_from_s, img2_gen_from_s,
                                     clip_model=self.style_clip.model
                                     )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description(
                    (
                        f"loss: {loss.item():.4f};"
                        f"alpha:{alpha:.2f}"
                        f"beta:{beta:.2f}"
                    )
                )
                if i % 50 == 0:
                    img2_gen_from_s.save(os.path.join(save_file, '{:05d}.jpg'.format(i)))
            # 删除变量
            del img1_gen_from_w
            del img2_gen_from_w
            del img1_gen_from_s
            del img2_gen_from_s

    # 将某一层转成img
    def s_to_img(self, alpha, beta, index):
        self.style_clip.M.alpha = [float(alpha)]
        self.style_clip.beta = beta

        img_index = index

        self.style_clip.M.img_index = img_index
        #
        self.style_clip.M.dlatent_tmp = [tmp[img_index:(img_index + 1)] for tmp in self.style_clip.M.dlatents]
        self.style_clip.GetDt2()
        img2 = self.style_clip.GetImg()
        return img2

    def w_to_img(self, index):
        self.style_clip.M.img_index = index
        return self.style_clip.GetImg()


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def compute_loss(img1, img2, img_s1, img_s2, clip_model):
    Image_img1 = clip_model.encode_image(img1)
    Image_img2 = clip_model.encode_image(img2)
    Image_img_s1 = clip_model.encode_image(img_s1)
    Image_img_s2 = clip_model.encode_image(img_s2)
    return abs(torch.dist(Image_img2, Image_img1) - torch.dist(Image_img_s1, Image_img_s2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataset_name', type=str, default='ffhq',
                        help='name of dataset, for example, ffhq')
    parser.add_argument('--latents_dir', type=str, default='./global_direction/latents.pt',
                        help='path to your latent file')
    parser.add_argument('--set_neutral', type=str, default='face with lips'
                        )
    parser.add_argument('--set_target', type=str, default='face with red lips'
                        )
    parser.add_argument('--set_alpha', type=float, default=2., help='Change feature intensity')
    parser.add_argument('--set_beta', type=float, default=0.16, help='set the hreshold')
    parser.add_argument('--save_dir', type=str, default='./result', help='save_images_dir under the ./result')
    parser.add_argument('--save_file', type=str, default='', help='save_images_dir under the ./result')
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--step", type=int, default=300)
    parser.add_argument("--ckpt", type=str, default="../pretrained_models/stylegan2-ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")

    args = parser.parse_args()
    self = optimization(args)
