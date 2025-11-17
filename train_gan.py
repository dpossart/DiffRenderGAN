from __future__ import print_function
import argparse
import json
import sys
from tqdm import tqdm
import torch
from torch.autograd import Variable
import wandb
import albumentations as A
from utils.util import (
    print_parser_args,
    create_dataloader,
    create_dir,
    save_output_as_tif,
    save_state,
    load_state,
    set_seed,
)

sys.path.append("CycleGAN")
from CycleGAN.models.networks import define_D, GANLoss, init_weights
from model.generator import Generator
from utils.distributions import sample_particles, draw_sample
from cleanfid.fid import compute_fid
from datasets.gan_dataset import GANDataset

# Constants
BATCH_SIZE = 1
CHECKPOINTS_DIR = "experiments/"
NET_G = "G"
NET_D = "D"
OPT_G = "OPT_G"
OPT_D = "OPT_D"
GAN_TRANSFORMS = A.Compose([
    A.Flip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ToFloat(),
])
Tensor = torch.cuda.FloatTensor


def extract_params(args):
    """
       Extracts mesh and scene parameters from the argument parser.

       Args:
           args (Namespace): Parsed command-line arguments.

       Returns:
           tuple: (mesh_dict, scene_param_dict)
    """
    mesh_dict = {
        "mesh_path": args.mesh_path,
        "mesh_unit": args.mesh_unit,
        "mesh_base_sample_limit": args.mesh_base_sample_limit,
        "mesh_scale": args.mesh_scale,
        "mesh_position3d": args.mesh_position3d,
        "mesh_max_agglomerates": args.mesh_max_agglomerates,
    }
    # Adding size distribution parameters based on the selected distribution
    if args.bimodal:
        mesh_dict["mesh_size_dist"] = "bimodal"
        mesh_dict["bimodal_params"] = {
            "mean1": args.bimodal[0],
            "stddev1": args.bimodal[1],
            "p": args.bimodal[2],
            "mean2": args.bimodal[3],
            "stddev2": args.bimodal[4],
        }
    elif args.gaussian:
        mesh_dict["mesh_size_dist"] = "gaussian"
        mesh_dict["gaussian_params"] = {
            "mean": args.gaussian[0],
            "stddev": args.gaussian[1],
        }
    elif args.lognormal:
        mesh_dict["mesh_size_dist"] = "lognormal"
        mesh_dict["lognormal_params"] = {
            "mu": args.lognormal[0],
            "sigma": args.lognormal[1],
        }
    # Adding particle placement options
    mesh_dict["particle_placement"] = "random" if args.random_placement else "poisson"
    scene_param_dict = {
        "n_hidden": args.n_hidden_units,
        "emission1": args.emission1,
        "emission2": args.emission2,
        "spp": args.spp,
        "gaussdev": args.gaussdev,
        "noise_limit_min": args.noise_limit[0],
        "noise_limit_max": args.noise_limit[1],
        "stage_bsdf_limit_min": args.stage_bsdf_limit[0],
        "stage_bsdf_limit_max": args.stage_bsdf_limit[1],
        "particle_bsdf_limit_min": args.particle_bsdf_limit[0],
        "particle_bsdf_limit_max": args.particle_bsdf_limit[1],
        "p_base_color": args.p_base_color,
        "p_roughness": args.p_roughness,
        "p_anisotropic": args.p_anisotropic,
        "p_metallic": args.p_metallic,
        "p_spec_trans": args.p_spec_trans,
        "p_specular": args.p_specular,
        "p_spec_tint": args.p_spec_tint,
        "p_sheen": args.p_sheen,
        "p_sheen_tint": args.p_sheen_tint,
        "p_flatness": args.p_flatness,
        "p_clearcoat": args.p_clearcoat,
        "p_clearcoat_gloss": args.p_clearcoat_gloss,
        "s_base_color": args.s_base_color,
        "s_roughness": args.s_roughness,
        "s_anisotropic": args.s_anisotropic,
        "s_metallic": args.s_metallic,
        "s_spec_trans": args.s_spec_trans,
        "s_specular": args.s_specular,
        "s_spec_tint": args.s_spec_tint,
        "s_sheen": args.s_sheen,
        "s_sheen_tint": args.s_sheen_tint,
        "s_flatness": args.s_flatness,
        "s_clearcoat": args.s_clearcoat,
        "s_clearcoat_gloss": args.s_clearcoat_gloss,
    }
    return mesh_dict, scene_param_dict


def validate(gen, positions, dataset_real_path, epoch, tag):
    """
    Validates the generator by computing FID scores.

    Args:
        gen (Generator): The trained generator model.
        positions (list): List of particle positions for validation.
        dataset_real_path (str): Path to real dataset images.
        epoch (int): Current epoch number.
        tag (str): Experiment identifier.

    Returns:
        float: Computed FID score.

    """
    fake_path = f"{CHECKPOINTS_DIR}/{tag}/output/{epoch}"
    create_dir(fake_path)

    with tqdm(total=len(positions), desc="Validation Step Generating Fake Images") as pbar:
        for n, position in enumerate(positions):
            fake = gen.forward(position)
            save_output_as_tif(fake, fake_path, str(n))
            pbar.update(1)
            pbar.set_postfix({"Image": n})

    fid = compute_fid(dataset_real_path, fake_path)
    return fid


def train(args):
    """
    Trains the GAN model using the provided command-line arguments.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    wandb_tag = args.wandb_tag
    experiment = wandb.init(project=wandb_tag, resume="allow", anonymous="must")

    print_parser_args(args)
    rng_seed = args.rng_seed
    set_seed(rng_seed)
    tag = args.tag

    dataset_path = args.dataset_path
    n_epochs = args.n_epochs

    lr_D = args.lr_D
    lr_G = args.lr_G
    load_cp = args.load_cp
    init_type = args.init_type

    mesh_dict, scene_param_dict = extract_params(args)

    exp_dir = f"{CHECKPOINTS_DIR}/{tag}/"
    create_dir(exp_dir)
    exp_args = {'mesh_dict': mesh_dict, 'scene_param_dict': scene_param_dict}
    with open(f"{CHECKPOINTS_DIR}/{tag}/{tag}.json", "w") as file:
        json.dump(exp_args, file)

    # Setup models
    gen = Generator(scene_param_dict, mesh_dict)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_G, betas=(0.5, 0.999))
    init_weights(gen, init_type=init_type)

    disc = define_D(1, 64, "basic", init_type=init_type)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_D, betas=(0.5, 0.999))

    if load_cp > 0:
        load_state(gen, f"{NET_G}_{load_cp}", dir=f"{CHECKPOINTS_DIR}/{tag}")
        load_state(disc, f"{NET_D}_{load_cp}", dir=f"{CHECKPOINTS_DIR}/{tag}")

    disc, gen = disc.to("cuda"), gen.to("cuda")

    gan_criterion = GANLoss(gan_mode="lsgan").to("cuda")

    train_dataset = GANDataset(img_dir=dataset_path, transform=GAN_TRANSFORMS)
    train_dataloader = create_dataloader(train_dataset, 1)

    positions = [Variable(Tensor(sample_particles(gen.n_meshes, mesh_dict)), requires_grad=False) for _ in
                 range(len(train_dataloader))]

    fid = validate(gen, positions, dataset_path, -1, tag)

    experiment.log({
        'FID': fid,
        'Render Parameters': gen.create_param_distributions(positions)
    })

    for epoch in range(load_cp, n_epochs):
        disc_loss, gen_loss = [], []
        gen.train()
        with tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch}") as pbar:
            for i, data in enumerate(train_dataloader, 0):
                real = data.to('cuda')
                position = draw_sample(positions)

                render_fake = gen(position)

                opt_g.zero_grad()
                err_render = gan_criterion(disc(render_fake), True)
                err_render.backward()
                opt_g.step()

                opt_d.zero_grad()
                err_disc_real = gan_criterion(disc(real), True)
                err_disc_render_fake = gan_criterion(disc(render_fake.detach()), False)
                err_disc = (err_disc_real + err_disc_render_fake) * 0.5
                err_disc.backward()
                opt_d.step()

                pbar.update(1)
                pbar.set_postfix({"Loss_D": err_disc.item(), "Loss_G": err_render.item()})

                disc_loss.append(err_disc.item())
                gen_loss.append(err_render.item())

        gen.eval()
        fid = validate(gen, positions, dataset_path, epoch, tag)
        experiment.log({
            'Discriminator loss': sum(disc_loss) / len(disc_loss),
            'Generator loss': sum(gen_loss) / len(gen_loss),
            'Fake': wandb.Image(render_fake[0].detach().cpu()),
            'Real': wandb.Image(real[0].detach().cpu()),
            'FID': fid,
            'Render Parameters': gen.create_param_distributions(positions)
        })
        save_state(gen, f'{NET_G}_{epoch}', dir=f"{CHECKPOINTS_DIR}/{tag}")
        save_state(disc, f'{NET_D}_{epoch}', dir=f"{CHECKPOINTS_DIR}/{tag}")


if __name__ == '__main__':
    # Add arguments with tuple type
    def tuple_type(s):
        try:
            values = tuple(float(x.strip()) for x in s.split(','))
            if len(values) != 2:
                raise argparse.ArgumentTypeError("Tuple must contain exactly two elements")
            return values
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid tuple format")


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='DiffRenderGAN Training')
    parser.add_argument("--tag", type=str, required=True, help="Experiment tag")
    parser.add_argument("--wandb_tag", type=str, default='DiffRenderGAN_Experiments', help="Wandb project tag")

    # Mesh parameters
    parser.add_argument("--mesh_path", type=str, required=True, help="Path to directory containing mesh files")
    parser.add_argument("--mesh_unit", type=float, required=True, help="Mesh size unit")
    parser.add_argument("--mesh_base_sample_limit", type=float, default=1.0,
                        help="Base seed distribution norm limit in image")
    parser.add_argument("--mesh_scale", type=float, required=True, help="Scaling factor for mesh samples")

    # Size distribution options
    size_dist_group = parser.add_mutually_exclusive_group(required=True)
    size_dist_group.add_argument("--bimodal", nargs=5, type=float,
                                 metavar=('mean1', 'stddev1', 'p', 'mean2', 'stddev2'),
                                 help="Parameters for bimodal distribution ('mean1', 'stddev1', 'p', 'mean2', 'stddev2')")
    size_dist_group.add_argument("--gaussian", nargs=2, type=float, metavar=('mean', 'stddev'),
                                 help="Parameters for Gaussian distribution (mean, stddev)")
    size_dist_group.add_argument("--lognormal", nargs=2, type=float, metavar=('mu', 'sigma'),
                                 help="Parameters for lognormal distribution (mu, sigma)")

    parser.add_argument("--mesh_position3d", default=False, action="store_true", help="Enable 3D positioning of meshes")
    parser.add_argument("--mesh_max_agglomerates", type=int, default=1, help="Maximum number of agglomerates")

    # Particle placement options
    parser.add_argument("--random_placement", action="store_true", help="Enable random particle placement")
    parser.add_argument("--poisson_placement", action="store_true", help="Enable Poisson-based particle placement")

    # Dataset and training parameters
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Directory path containing temp images for training')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr_D', type=float, default=0.0001, help='Learning rate for the discriminator')
    parser.add_argument('--lr_G', type=float, default=0.0002, help='Learning rate for the render generator')
    parser.add_argument("--init_type", type=str, default='xavier', help="normal | xavier | kaiming | orthogonal")
    parser.add_argument("--n_hidden_units", type=int, default=128, help="")
    parser.add_argument('--load_cp', type=int, default=0, help='Epoch of the checkpoint to load if resuming training')
    parser.add_argument('--rng_seed', type=int, default=777, help='Random seed for reproducibility')

    # Rendering parameters
    parser.add_argument("--spp", type=int, default=121, help="Samples per pixel during rendering")
    # See https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_rfilters.html#gaussian-filter-gaussian
    parser.add_argument("--gaussdev", type=float, default=0.5,
                        help="Deviation for Gaussian reconstruction filter after rendering")
    parser.add_argument('--emission1', type=float, default=1.0, help="Scene emission")
    parser.add_argument('--emission2', type=float, default=0.0, help="Scene emission")

    # Add arguments with tuple type
    parser.add_argument('--particle_bsdf_limit', type=tuple_type, default=(0.01, 1),
                        help="Tuple describing the minimum and maximum particle BSDF limit")
    parser.add_argument('--stage_bsdf_limit', type=tuple_type, default=(0.01, 1),
                        help="Tuple describing the minimum and maximum stage BSDF limit")
    parser.add_argument('--noise_limit', type=tuple_type, default=(0.01, 1),
                        help="Tuple describing the minimum and maximum noise deviation limit")

    # Scene parameters for optimization
    # See https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html#the-principled-bsdf-principled
    # Particle related BSDFs
    parser.add_argument('--p_base_color', type=float, default=0.5, help="Color of the material. Optimization if < 0,")
    parser.add_argument('--p_roughness', type=float, default=0.5, help="Roughness of the material. Optimization if < 0")
    parser.add_argument('--p_anisotropic', type=float, default=0.0, help="Degree of anisotropy. Optimization if < 0")
    parser.add_argument('--p_metallic', type=float, default=0.0,
                        help="Metallicness of the material. Optimization if < 0")
    parser.add_argument('--p_spec_trans', type=float, default=0.0,
                        help="Blends BRDF and BSDF major lobe. Optimization if < 0")
    parser.add_argument('--p_specular', type=float, default=0.5,
                        help="Fresnel reflection coefficient. Optimization if < 0")
    parser.add_argument('--p_spec_tint', type=float, default=0.0,
                        help="Tint applied to dielectric reflection lobe. Optimization if < 0")
    parser.add_argument('--p_sheen', type=float, default=0.0, help="Rate of the sheen lobe. Optimization if < 0")
    parser.add_argument('--p_sheen_tint', type=float, default=0.0,
                        help="Tint applied to sheen lobe. Optimization if < 0")
    parser.add_argument('--p_flatness', type=float, default=0.0,
                        help="Blends between diffuse response and fake subsurface scattering. Optimization if < 0")
    parser.add_argument('--p_clearcoat', type=float, default=0.0,
                        help="Rate of the secondary isotropic specular lobe. Optimization if < 0")
    parser.add_argument('--p_clearcoat_gloss', type=float, default=0.0,
                        help="Roughness of the secondary specular lobe. Optimization if < 0")
    # Stage related BSDFs
    parser.add_argument('--s_base_color', type=float, default=0.5, help="Color of the material. Optimization if < 0,")
    parser.add_argument('--s_roughness', type=float, default=0.5, help="Roughness of the material. Optimization if < 0")
    parser.add_argument('--s_anisotropic', type=float, default=0.0, help="Degree of anisotropy. Optimization if < 0")
    parser.add_argument('--s_metallic', type=float, default=0.0,
                        help="Metallicness of the material. Optimization if < 0")
    parser.add_argument('--s_spec_trans', type=float, default=0.0,
                        help="Blends BRDF and BSDF major lobe. Optimization if < 0")
    parser.add_argument('--s_specular', type=float, default=0.5,
                        help="Fresnel reflection coefficient. Optimization if < 0")
    parser.add_argument('--s_spec_tint', type=float, default=0.0,
                        help="Tint applied to dielectric reflection lobe. Optimization if < 0")
    parser.add_argument('--s_sheen', type=float, default=0.0, help="Rate of the sheen lobe. Optimization if < 0")
    parser.add_argument('--s_sheen_tint', type=float, default=0.0,
                        help="Tint applied to sheen lobe. Optimization if < 0")
    parser.add_argument('--s_flatness', type=float, default=0.0,
                        help="Blends between diffuse response and fake subsurface scattering. Optimization if < 0")
    parser.add_argument('--s_clearcoat', type=float, default=0.0,
                        help="Rate of the secondary isotropic specular lobe. Optimization if < 0")
    parser.add_argument('--s_clearcoat_gloss', type=float, default=0.0,
                        help="Roughness of the secondary specular lobe. Optimization if < 0")

    args = parser.parse_args()
    train(args)
