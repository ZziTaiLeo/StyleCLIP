import torchvision
import argparse
from argparse import Namespace
from optimization.run_optimization import main



parser = argparse.ArgumentParser()
parser.add_argument("--description", type=str, default="a person with purple hair",
                    help="the text that guides the editing/generation")
parser.add_argument("--ckpt", type=str, default="./pretrained_models/stylegan2-ffhq-config-f.pt",
                    help="pretrained StyleGAN2 weights")
parser.add_argument("--stylegan_size", type=int, default=1024, help="StyleGAN resolution")
parser.add_argument("--lr_rampup", type=float, default=0.05)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--step", type=int, default=30, help="number of optimization steps")
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
# parser.add_argument("--save_intermediate_image_every", type=int, default=20,
#                     help="if > 0 then saves intermidate results during the optimization")
parser.add_argument("--results_dir", type=str, default="results")

args = vars(parser.parse_args())
result_image = main(Namespace(**args))
torchvision.utils.save_image(result_image.detach().cpu(), f"results/final_result.png", normalize=True, scale_each=True,
                             range=(-1, 1))