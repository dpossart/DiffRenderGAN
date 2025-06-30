import drjit as dr
import mitsuba as mi
import numpy as np
import torch
import torch.nn as nn

from model.regression_model import RegressionModel
from utils.rendering.particle_scene import build_particle_scene, create_sensor

mi.set_variant("cuda_ad_mono")

MASK_INTEGRATOR = mi.load_dict({
    "type": "aov",
    "aovs": "shape:shape_index",
    "integrator": {
        "type": "path",
    },
})

# Wrapper for the mask sensor
MASK_SCENE = mi.load_string(f"""
<scene version="3.0.0">
{create_sensor(0.3)}
</scene>
""")


class Generator(nn.Module):
    """
    Generator model consisting of a regression model for estimating scene parameters
    and a differentiable renderer for rendering an image.
    """

    def __init__(self, scene_params_dict, mesh_params_dict):
        """
        Initialize Generator.

        Args:
            scene_params_dict: Dictionary of scene parameters.
            mesh_params_dict: Dictionary of mesh parameters.
        """
        super(Generator, self).__init__()

        # Build particle scene
        self.n_meshes, self.d_bsdf_keys, self.d_stage_keys, scene = build_particle_scene(scene_params_dict,
                                                                                         mesh_params_dict)
        self.scene = mi.load_string(scene)
        self.scene_params = mi.traverse(self.scene)

        self.mesh_scale = mesh_params_dict['mesh_scale']
        self.mesh_unit = mesh_params_dict['mesh_unit']
        self.spp = scene_params_dict['spp']
        self.particle_bsdf_limit_min = scene_params_dict['particle_bsdf_limit_min']
        self.particle_bsdf_limit_max = scene_params_dict['particle_bsdf_limit_max']
        self.noise_limit_min = scene_params_dict['noise_limit_min']
        self.noise_limit_max = scene_params_dict['noise_limit_max']
        self.stage_bsdf_limit_min = scene_params_dict['stage_bsdf_limit_min']
        self.stage_bsdf_limit_max = scene_params_dict['stage_bsdf_limit_max']

        # Build the linear model for parameter estimation
        self.params_model = RegressionModel(self.n_meshes * 4, len(self.d_bsdf_keys) + 1 + len(self.d_stage_keys),
                                            n_hidden=scene_params_dict['n_hidden'])
        self.init_mesh_positions = self._init_mesh_positions()
        self.positions = {}

    def _init_mesh_positions(self):
        """
        Initialize the positions of the meshes in the scene.
        """
        return {
            f"particle_{n + 1}.vertex_positions":
                dr.unravel(mi.Point3f,
                           self.scene_params[f"particle_{n + 1}.vertex_positions"])
            for n in range(self.n_meshes)
        }

    def forward(self, position):
        """
        Forward pass of the model.

        Args:
            position: Tensor of size 1 x n_meshes x 4.

        Returns:
            Generated fake image.
        """
        self.position_meshes_in_scene(position.tolist())
        noise_std, particle_bsdf, stage_bsdf = self.forward_linear_model(torch.unsqueeze(position, 0))

        fake = self.forward_render(particle_bsdf[0], stage_bsdf[0]).permute((2, 0, 1))
        fake = torch.clamp(fake, 0.0, 1.0)
        fake = fake + torch.randn_like(fake) * noise_std.view(-1, 1, 1)
        fake = torch.clamp(fake, 0.0, 1.0)
        fake = torch.unsqueeze(fake, 0)
        return fake

    def test(self, position, bg_only=False):
        """
        Forward pass of the model.

        Args:
            position: Tensor of size 1 x n_meshes x 4.

        Returns:
            Generated fake image.
        """

        if bg_only:
            p, c = position.shape
            self.position_meshes_in_scene([[30 for _ in range(c)] for _ in range(p)])
        else:
            self.position_meshes_in_scene(position.tolist())

        noise_std, particle_bsdf, stage_bsdf = self.forward_linear_model(torch.unsqueeze(position, 0))
        fake = self.forward_render(particle_bsdf[0], stage_bsdf[0]).permute((2, 0, 1))
        fake = torch.clamp(fake, 0.0, 1.0)
        fake = fake + torch.randn_like(fake) * noise_std.view(-1, 1, 1)
        fake = torch.clamp(fake, 0.0, 1.0)
        fake = torch.unsqueeze(fake, 0)
        return fake

    def forward_linear_model(self, z):
        """
        Forward pass of the linear model.
        """
        z = z.view(z.size(0), -1)
        params = self.params_model(z)

        stage_bsdf = np.linalg.norm(self.stage_bsdf_limit_max - self.stage_bsdf_limit_min) * params[:, :len(
            self.d_stage_keys)] + self.stage_bsdf_limit_min
        particle_bsdf = np.linalg.norm(self.particle_bsdf_limit_max - self.particle_bsdf_limit_min) * params[:,
                                                                                                      len(self.d_stage_keys):-1] + self.particle_bsdf_limit_min
        noise_dev = np.linalg.norm(self.noise_limit_max - self.noise_limit_min) * params[:, -1:] + self.noise_limit_min

        return noise_dev, particle_bsdf, stage_bsdf

    @dr.wrap_ad(source="torch", target="drjit")
    def forward_render(self, particle_bsdf, stage_bsdf, seed=1):
        """
        Forward rendering pass using Mitsuba.

        Args:
            particle_bsdf: Shader properties.

        Returns:
            Rendered image.
        """
        for j, scene_key in enumerate(self.d_stage_keys):
            key = f"stage.bsdf.{scene_key[2:]}"
            key = f'{key}.value' if scene_key != 's_specular' else key
            self.scene_params[key] = dr.ravel(stage_bsdf[j])

        for p in range(self.n_meshes):
            for i, scene_key in enumerate(self.d_bsdf_keys):
                key = f"particle_{p + 1}_bsdf.{scene_key[2:]}"
                key = f'{key}.value' if scene_key != 'p_specular' else key
                self.scene_params[key] = dr.ravel(particle_bsdf[i])

        self.scene_params.update()
        return mi.render(self.scene, params=self.scene_params, spp=self.spp)

    def render_mask(self):
        """
        Render a shape mask using the current positions of the meshes in the scene.

        Returns:
            Rendered mask.
        """
        init_stage_pos = dr.unravel(mi.Point3f, self.scene_params["stage.vertex_positions"])
        # trafo = mi.Transform4f.translate((30, 0, 0)) # Mitsuba 3.4
        # Reposition stage mesh out of view
        trafo = mi.Transform4f().translate((30, 0, 0))  # Mitsuba 3.6
        transformed_positions = trafo @ init_stage_pos
        self.scene_params["stage.vertex_positions"] = dr.ravel(transformed_positions)
        self.scene_params.update()
        sensors = MASK_SCENE.sensors()[0]
        mi.render(self.scene, params=self.scene_params, spp=144, integrator=MASK_INTEGRATOR, sensor=sensors)
        multichannel = sensors.film().bitmap()
        split_channels = dict(multichannel.split())
        self.scene_params["stage.vertex_positions"] = dr.ravel(init_stage_pos)
        self.scene_params.update()
        return np.asarray(split_channels["shape"])

    def position_meshes_in_scene(self, positions, offset=1.0):
        """
        Positions meshes in the scene based on input positions and offset.

        Args:
            positions: List of tuples containing (x, y, z, r) values for positioning.
            offset: Scaling factor for x, y, and z coordinates. Default is 0.9.
        """
        for i, position in enumerate(positions):
            x, y, z, r = position
            x, y, z = x * offset * self.mesh_unit, y * offset * self.mesh_unit, z * offset * self.mesh_unit
            # trafo = mi.Transform4f.translate((x, z, y)).scale(r * self.mesh_scale) # Mitsuba 3.4
            trafo = mi.Transform4f().translate((x, z, y)).scale(r * self.mesh_scale)  # Mitsuba 3.6
            transformed_positions = (
                    trafo @
                    self.init_mesh_positions[f"particle_{i + 1}.vertex_positions"])

            self.scene_params[f"particle_{i + 1}.vertex_positions"] = dr.ravel(transformed_positions)

        self.scene_params.update()

    def create_param_distributions(self, positions):
        """
        Create Generator parameter distributions.

        Args:
            positions: List of all mesh positions used during training.

        Returns:
            Histogram of parameter distributions.
        """
        zs = torch.stack(positions, dim=0)
        noise, particle_bsdf, stage_bsdf = self.forward_linear_model(zs)

        particle_bsdf_cpu = particle_bsdf.cpu().detach()
        noise_cpu = noise.cpu().detach()
        stage_bsdf_cpu = stage_bsdf.cpu().detach()

        histogram = {}
        for j, scene_key in enumerate(self.d_bsdf_keys):
            histogram[scene_key] = particle_bsdf_cpu[:, j].view(-1)

        for i, scene_key in enumerate(self.d_stage_keys):
            histogram[scene_key] = stage_bsdf_cpu[:, i].view(-1)

        histogram["noise_std"] = noise_cpu.view(-1)
        return histogram
