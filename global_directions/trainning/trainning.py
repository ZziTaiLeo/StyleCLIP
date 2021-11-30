import sys
import torchvision
from torch import optim
from tqdm import tqdm
import torch
from criteria.clip_loss import CLIPLoss
from models.stylegan2.model import Generator
import clip

def main():
