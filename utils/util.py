import os
import pathlib
import random
import shutil
from PIL import Image

from sklearn.model_selection import KFold
import numpy as np
import torch
from patchify import patchify
from tifffile import imwrite

CHECKPOINTS_DIR = 'checkpoints'


def save_state(model, name, dir=CHECKPOINTS_DIR):
    """Save the state of a PyTorch model.

    Args:
        model: The PyTorch model.
        name: The name for the checkpoint file.
        dir: The directory to save the checkpoint file. Default is CHECKPOINTS_DIR.
    """
    create_dir(dir)
    torch.save(model.state_dict(), f'{dir}/{name}.pth')


def load_state(model, name, dir=CHECKPOINTS_DIR):
    """Load the state of a PyTorch model.

    Args:
        model: The PyTorch model.
        name: The name for the checkpoint file.
        dir: The directory to load the checkpoint file from.
    """
    state = torch.load(f'{dir}/{name}.pth', map_location='cuda')
    model.load_state_dict(state)


def save_output_as_tif(output, dir, name):
    """Save output as TIFF file.

    Args:
        output: The output data.
        dir: The directory to save the TIFF file.
        name: The name of the TIFF file.
    """
    if not isinstance(output, np.ndarray):
        output = output.detach().cpu().numpy()
        output *= 255
        output = output.astype(np.uint8)

    imwrite(f'{dir}/{name}.tif', output)


def print_parser_args(args):
    """Print parser arguments.

    Args:
        args: The argparse arguments.
    """
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')


def create_dir(path):
    """Create a directory if it doesn't exist.

    Args:
        path: The path of the directory.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """Create a PyTorch DataLoader.

    Args:
        dataset: PyTorch dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.

    Returns:
        DataLoader.
    """
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers)


def crop_and_save_patches(input_dir: str, output_dir: str, patch_size: tuple, step_size: int):
    """Crop images in the input directory and save them to the output directory.

    Args:
        input_dir: Path to the directory containing input images.
        output_dir: Path to the directory where patches will be saved.
        patch_size: Size of the patches (height, width).
        step_size: Degree of overlap. Step size between patches.
    """
    # Create the output directory if it doesn't exist
    create_dir(output_dir)

    # List all files in the input directory
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    for file_name in input_files:
        # Read the image
        img_path = os.path.join(input_dir, file_name)
        crop_img_file(img_path, output_dir, patch_size, step_size)


def crop_img_file(img_path, output_dir, patch_size, step_size):
    """Crop an image file into patches and save them to a specified output directory.

    Args:
    img_path: Path to the input image file.
    output_dir: Path to the output directory.
    patch_size: Size of the patches (height, width).
    step_size: Degree of overlap. Step size between patches.
    """
    # Open and convert the image to a Numpy array
    img = Image.open(img_path, 'r')
    img = np.asarray(img)

    # Create non-overlapping patches with the specified patch size and overlap
    patches = patchify(img, patch_size, step=step_size)

    # Save each patch as a TIFF image
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j]
            patch_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_patch_{i}_{j}.tif"
            patch_path = os.path.join(output_dir, patch_name)

            # Save the patch as a TIFF image
            Image.fromarray(patch).save(patch_path)


def get_sorted_paths(images_dir, masks_dir, sort_key):
    """Retrieves sorted image and mask paths.

    Args:
        images_dir: Directory containing image files.
        masks_dir: Directory containing mask files.
        sort_key: Comparator.

    Returns:
        Sorted lists of image and mask paths.
    """
    image_paths = sorted(list(pathlib.Path(images_dir).glob('*')), key=sort_key)
    mask_paths = sorted(list(pathlib.Path(masks_dir).glob('*')), key=sort_key)
    return image_paths, mask_paths



def set_seed(seed):
    """Set seed for random number generators.

    Args:
        seed: Seed value.
    """
    print("Setting seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_kfold_splits(images_dir, masks_dir, output_dir, n_splits=5, random_state=5897):
    """
    Creates K-fold cross-validation splits for images and masks.

    Parameters:
    images_dir (str): Path to the folder containing images.
    masks_dir (str): Path to the folder containing masks.
    output_dir (str): Path to save the folds.
    n_splits (int): Number of folds for cross-validation.
    random_state (int): Random seed for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(os.listdir(images_dir))
    masks = os.listdir(masks_dir)

    assert len(images) == len(masks), "The number of images and masks must match."

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, val_idx) in enumerate(kf.split(images), 1):
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        train_images_dir = os.path.join(fold_dir, "train", "images")
        train_masks_dir = os.path.join(fold_dir, "train", "masks")
        val_images_dir = os.path.join(fold_dir, "test", "images")
        val_masks_dir = os.path.join(fold_dir, "test", "masks")

        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(train_masks_dir, exist_ok=True)
        os.makedirs(val_images_dir, exist_ok=True)
        os.makedirs(val_masks_dir, exist_ok=True)

        for idx in train_idx:
            shutil.copy(os.path.join(images_dir, images[idx]), train_images_dir)
            shutil.copy(os.path.join(masks_dir, masks[idx]), train_masks_dir)

        for idx in val_idx:
            shutil.copy(os.path.join(images_dir, images[idx]), val_images_dir)
            shutil.copy(os.path.join(masks_dir, masks[idx]), val_masks_dir)

if __name__ == '__main__':
    pass
    input_imgs_dir = "../Data/Ag/imgs"

    output_imgs_dir = "../Data/Ag/imgs_crop"

    bg_dir = "../Data/Ag/bg"
    #crop_and_save_patches(input_imgs_dir, output_imgs_dir, patch_size=(256,256))

    create_dir(bg_dir)
    #
    def rename_images_and_create_empty_images(folder_path, start_number):
        # Check if the given folder path exists
        if not os.path.isdir(folder_path):
            print(f"The folder {folder_path} does not exist.")
            return

        # Create a new folder for empty images
        parent_folder = os.path.dirname(folder_path)
        new_folder_path = os.path.join(parent_folder, "empty_images")
        os.makedirs(new_folder_path, exist_ok=True)

        # List all files in the given folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Sort files to ensure consistent renaming
        files.sort()

        # Initialize the counter with the start_number
        counter = start_number

        for filename in files:
            # Get the file extension
            file_extension = os.path.splitext(filename)[1]

            # Create a new filename with the counter
            new_filename = f"{counter}.tif"

            # Create full path for the old and new filenames
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_file_path, new_file_path)

            print(f"Renamed: {old_file_path} to {new_file_path}")

            # Create an empty grayscale image with the same name in the new folder
            empty_image_path = os.path.join(new_folder_path, new_filename)
            create_empty_image(empty_image_path, file_extension)

            print(f"Created empty image: {empty_image_path}")

            # Increment the counter
            counter += 1


    def create_empty_image(image_path, file_extension):
        # Create a new empty grayscale image (256x256)
        empty_image = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))
        empty_image.save(image_path)

    rename_images_and_create_empty_images("../Data/Ag/imgs_crop/", start_number=1000)