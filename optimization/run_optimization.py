import argparse
import math
import os
import sys
import time

sys.path.append('/home/ming13/editor/pycharm-2021.1.1/workplace/styleClip/StyleCLIP/')
sys.path.append('/home/ming13/editor/pycharm-2021.1.1/workplace/styleClip/StyleCLIP/utils.py')
sys.path.append('/home/ming13/editor/pycharm-2021.1.1/workplace/styleClip/StyleCLIP/models/stylegan2/model.py')
import torch
import torchvision
from torch import optim
from tqdm import tqdm
import torch
from criteria.clip_loss import CLIPLoss
from models.stylegan2.model import Generator
import clip
from utils import ensure_checkpoint_exists


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def main(args, file,last_latent):
    ensure_checkpoint_exists(args.ckpt)
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)

    g_ema = Generator(args.stylegan_size, 512, 8)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.cuda()
    mean_latent = g_ema.mean_latent(4096)
    # 读入文件

    # #读入.pt文件,这里开始逐步读入文件
    # if args.latent_path:
    #     latent_code_init = torch.load(args.latent_path).cuda()
    # elif args.mode == "edit":
    #     latent_code_init_not_trunc = torch.randn(1, 512).cuda()
    #     with torch.no_grad():
    #         _, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
    #                                     truncation=args.truncation, truncation_latent=mean_latent)
    # else:
    #     latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    if file:
        latent_code_init = torch.load(file).cuda()
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            _, latent_code_init = g_ema([latent_code_init_not_trunc], return_latents=True,
                                        truncation=args.truncation, truncation_latent=mean_latent)
    else:
        latent_code_init = mean_latent.detach().clone().repeat(1, 18, 1)

    # 克隆潜码
    latent = latent_code_init.detach().clone()
    latent.requires_grad = True

    clip_loss = CLIPLoss(args)
    # 将潜码拉入计算
    optimizer = optim.Adam([latent], lr=args.lr)
    # 步数
    pbar = tqdm(range(args.step))

    # int-1以对应上张量的第一个维度
    idx = int((file.split('/')[-1]).split('.')[0]) - 1
    # 周期
    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)

        c_loss = clip_loss(img_gen, text_inputs)

        if args.mode == "edit":
            l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = c_loss + args.l2_lambda * l2_loss
        else:
            loss = c_loss
        if args.constraint_w:
            # 第一张不需要上范数
            if idx > 0:
                l2_constraint_w = torch.norm((last_latent - latent), p=2, keepdim=False)
                loss += l2_constraint_w*args.w_lambda

        # 每次梯度清空
        optimizer.zero_grad()
        loss.backward()
        # 计算
        optimizer.step()
        # 输出loss
        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        # if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
        #     with torch.no_grad():
        #         img_gen, _ = g_ema([latent], input_is_latent=True, randomize_noise=False)
        #
        #     torchvision.utils.save_image(img_gen, f"results/{str(i).zfill(5)}.png", normalize=True, range=(-1, 1))
    # 这里选择是否拼接。

    if args.mode == "edit":
        with torch.no_grad():
            img_orig, _ = g_ema([latent_code_init], input_is_latent=True, randomize_noise=False)
        final_result = img_gen
    #      拼接
    #       final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    return final_result,latent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a person with purple hair",
                        help="the text that guides the editing/generation")
    parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt",
                        help="pretrained StyleGAN2 weights")
    parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    # lr默认0.1
    parser.add_argument("--lr", type=float, default=0.05)
    # 默认是300
    parser.add_argument("--step", type=int, default=200, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"],
                        help="choose between edit an image an generate a free one")
    parser.add_argument("--l2_lambda", type=float, default=0.008,
                        help="weight of the latent distance (used for editing only)")
    parser.add_argument("--latent_path", type=str, default=None,
                        help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                             "the mean latent in a free generation, and from a random one in editing. "
                             "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="used only for the initial latent vector, and only when a latent code path is"
                             "not provided")
    #   parser.add_argument("--save_intermediate_image_every", type=int, default=20, help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--constraint_w", type=bool, default=False)
    parser.add_argument("--sleep_ten_mins", type=bool, default=False)
    parser.add_argument("--w_lambda", type=float, default=0,help='to control the difference between adjacent')

    args = parser.parse_args()

    torch.cuda.empty_cache()
    files = os.listdir(args.latent_path)
    files.sort()
    # 编号从1开始
    idx = 1
    latent = torch.zeros((1,18,518))
    for file in files:
        print(idx)
        result_image,latent = main(args, args.latent_path + file,last_latent=latent)
        print('latent:',latent)
        image_name = '%05d.jpg' % idx
        torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, image_name),
                                     normalize=True, scale_each=True, range=(-1, 1))
        idx += 1
        # 避免高负荷操作，程序休息10分钟
        if idx % 50 == 0 and args.sleep_ten_mins:
            time.sleep(600)
    torch.cuda.empty_cache()