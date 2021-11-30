import os
from typing import Union

import clip
import numpy as np
import PIL.Image
# 目的 把读入的图片通过逆仿射变换变回去
# 加载仿射变换矩阵
import torch
import pandas as pd
import argparse
import numpy as np
def export_csv(file,list_input):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    align_path = '../global_directions/compare/align/'
    result_root = r'embedding_result/'
    if not os.path.exists(result_root+file):
        os.makedirs(result_root+file)
    result_file = os.path.join(result_root,file)

    files = os.listdir(align_path)
    files.sort()
    image_input = []
    text_input = list_input
    image_features = None
    for file in files:
        image_input.append(file.split('.')[0])  # 图片按顺序加入list
        image = preprocess(PIL.Image.open(align_path + file)).unsqueeze(0).to(device)
        with torch.no_grad():
            if image_features is None:
                image_features = model.encode_image(image)
            else:
                image_features = torch.cat([image_features, model.encode_image(image)], dim=0)

    image_emebdding = pd.DataFrame(data=image_features.data, index=image_input)

    text = clip.tokenize(text_input).to(device)
    text_features = model.encode_text(text)
    text_embedding = pd.DataFrame(data=text_features, index=text_input)

    image_emebdding.to_csv(result_file+"/image_emebdding.csv")
    text_embedding.to_csv(result_file+'/text_embedding.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='export image and text embedding')
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--neutral', type=str)
    parser.add_argument('--target', type=str)
    args = parser.parse_args()
    neutral = args.neutral
    target = args.target
    list_text = [neutral,target]
    result_file = args.result_file
    export_csv(result_file,list_text)