import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray)
        z_vals = z_vals.to('cuda')
        # TODO (Q1.4): Sample points from z values
        # print(ray_bundle.origins.device)      # Expected: [N, 3] (N = number of rays)
        # print(ray_bundle.directions.device)   # Expected: [N, 3]
        # print(z_vals.device)                  # Expected: [N, 64] or [N, 64, 1]
        z_vals = z_vals.view(1, 64, 1)
        sample_points = ray_bundle.origins + z_vals * ray_bundle.directions.unsqueeze(1)
        print("shape of sample_points: ", sample_points.shape)
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}