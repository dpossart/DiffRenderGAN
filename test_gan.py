from __future__ import print_function
import argparse
import numpy as np
import numpy.random
import json
from skimage.segmentation import find_boundaries
from skimage import measure
from skimage.morphology import skeletonize, binary_dilation
from torch.autograd import Variable
from tqdm import tqdm

from model.generator import Generator
from train_gan import Tensor, NET_G, CHECKPOINTS_DIR
from utils.distributions import sample_particles
from utils.util import load_state, save_output_as_tif, create_dir, set_seed


def enforce_cross_pattern_skeleton(skeleton):
    """
    Enforces a cross-pattern skeleton by detecting and correcting diagonal connections.

    Args:
        skeleton (numpy.ndarray): Binary skeletonized mask.

    Returns:
        numpy.ndarray: Corrected skeletonized mask.
    """
    from scipy import ndimage as ndi

    # Define diagonal correlation patterns
    diagonal_pattern1 = np.array([[0, 1, 0],
                                  [0, 0, 1],
                                  [0, 0, 0]])

    diagonal_pattern2 = np.array([[0, 0, 0],
                                  [0, 0, 1],
                                  [0, 1, 0]])

    # Detect diagonal connections
    diagonal_hits1 = ndi.correlate(skeleton.astype(int), diagonal_pattern1) == 2
    diagonal_hits2 = ndi.correlate(skeleton.astype(int), diagonal_pattern2) == 2

    # Create a corrected skeleton copy
    skeleton_corrected = skeleton.copy()
    skeleton_corrected[diagonal_hits1] = 1
    skeleton_corrected[diagonal_hits2] = 1

    return skeleton_corrected


def post_process_shape_mask(shape_mask, dilate, min_mask_size):
    """
    Postprocesses render shape mask removing too small instances,
    and optionally dilating contours.

    Args:
        shape_mask (numpy.ndarray): Binary mask of shape.
        dilate (bool): Whether to dilate the boundaries.

    Returns:
        numpy.ndarray: Processed shape mask.
    """
    # Round the values
    shape_mask = np.around(shape_mask, 0).astype(np.uint8)
    # Pad the mask for better boundary detection at borders
    shape_mask = np.pad(shape_mask, pad_width=((12, 12), (12, 12)), mode='constant', constant_values=0)

    boundaries = find_boundaries(shape_mask, connectivity=2, mode="outer")
    boundaries = skeletonize(boundaries > 0)
    boundaries = enforce_cross_pattern_skeleton(boundaries)

    shape_mask[shape_mask > 0] = 255
    shape_mask[boundaries > 0] = 0

    # Remove too small annotation occurrences
    labels = measure.label(shape_mask, connectivity=2)
    for l in np.unique(labels):
        if np.sum(labels == l) < min_mask_size:
            shape_mask[labels == l] = 0

    boundaries2 = find_boundaries(shape_mask, connectivity=2, mode="outer")

    if dilate:
        boundaries2 = binary_dilation(boundaries2)

    shape_mask[shape_mask > 0] = 1
    shape_mask[(boundaries2 > 0)] = 2
    # Remove mask padding
    shape_mask = shape_mask[12:-12, 12:-12]
    return shape_mask


def test(args):
    """
    Tests a trained generator model by generating synthetic images and masks.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    # Load configuration from the training checkpoint
    tag = args.load_tag
    load_cp = args.load_cp
    out_dir = args.output_directory
    n_fakes = args.n_fakes
    rng_seed = args.rng_seed
    dilate = args.dilate
    bg_portion = args.bg_portion
    bg_contrast = args.max_bg_portion
    min_mask_size = args.min_mask_size
    set_seed(rng_seed)

    with open(f"{CHECKPOINTS_DIR}/{tag}/{tag}.json", 'r') as file:
        train_args = json.load(file)
    print(train_args)

    mesh_dict, scene_param_dict = train_args['mesh_dict'], train_args['scene_param_dict']

    # Create DiffRenderer instance
    gen = Generator(scene_param_dict, mesh_dict)
    load_state(gen, f'{NET_G}_{load_cp}', dir=f"{CHECKPOINTS_DIR}/{tag}")
    gen = gen.to("cuda")
    gen.eval()

    # Generate fake images and masks
    fake_path = f'{out_dir}/imgs/'
    mask_path = f'{out_dir}/masks/'
    create_dir(fake_path)
    create_dir(mask_path)

    with tqdm(total=n_fakes, desc='Generating Images and Masks') as pbar:
        generated_images = 0

        while generated_images < n_fakes:
            # Sample a new position for every attempt
            z = Variable(Tensor(sample_particles(gen.n_meshes, mesh_dict)), requires_grad=False)

            fake = gen.test(z)
            fake = np.array(fake.detach().cpu().numpy() * 255.0, dtype=np.uint8)
            mask = gen.render_mask()

            # Ensure portion of background is present in synth images

            if (np.sum(mask == 0) / mask.size) < bg_portion:
                continue

            mean_particles = np.mean(fake[0][0][mask > 0])
            mean_bg = np.mean(fake[0][0][mask == 0])

            mask = post_process_shape_mask(mask, dilate, min_mask_size)

            # Remove images were background/particle contrast is missing

            if mean_bg * bg_contrast > mean_particles:
                continue

            # Save outputs if the condition is met
            save_output_as_tif(fake, fake_path, str(generated_images))
            save_output_as_tif(mask, mask_path, str(generated_images))

            pbar.set_postfix(
                {'Fake Image': f'{fake_path}{generated_images}.tif', 'Mask': f'{mask_path}{generated_images}.tif'})
            pbar.update(1)
            generated_images += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DiffRenderGAN Testing')
    parser.add_argument('--load_tag', type=str, required=True, help='Training Tag to load configuration')
    parser.add_argument('--load_cp', type=int, required=True, help='Epoch of the checkpoint for testing')
    parser.add_argument('--output_directory', type=str, required=True, help='Output directory')
    parser.add_argument('--n_fakes', type=int, required=True, help="Number of fakes")
    parser.add_argument('--dilate', type=bool, default=False, help="Dilate mask contour")
    parser.add_argument('--rng_seed', type=int, default=777, help="Random Seed")
    parser.add_argument('--bg_portion', type=float, default=0.15, help="Random Seed")
    parser.add_argument('--max_bg_portion', type=float, default=1.15, help="Random Seed")
    parser.add_argument('--min_mask_size', type=int, default=50, help="Random Seed")

    test(parser.parse_args())
