import torchvision
import argparse
from argparse import Namespace
from PIL import Image

from utils import ensure_checkpoint_exists
from mapper.scripts.inference import run

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default="./results", type=str, help='Path to experiment output directory')
parser.add_argument('--checkpoint_path', default="./pretrained_models/mapper/taylor_swift.pt", type=str,
                    help='Path to model checkpoint')
parser.add_argument('--couple_outputs', default=True, action='store_true',
                    help='Whether to also save inputs + outputs side-by-side')
parser.add_argument('--mapper_type', default='LevelsMapper', type=str, help='Which mapper to use')
parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
parser.add_argument('--no_medium_mapper', default=False, action="store_true")
parser.add_argument('--no_fine_mapper', default=False, action="store_true")
parser.add_argument('--stylegan_size', default=1024, type=int)
parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
parser.add_argument('--latents_test_path', default="./latents_test/example_celebs.pt", type=str,
                    help="The latents for the validation")
parser.add_argument('--test_workers', default=0, type=int, help='Number of test/inference dataloader workers')
parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')

args = vars(parser.parse_args())
run(Namespace(**args))