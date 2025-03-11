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
        # z_vals = z_vals.view(1, 64, 1)
        O = ray_bundle.directions.shape[0]
        N = z_vals.shape[0] 
        origins = ray_bundle.origins.unsqueeze(1).repeat(O, N, 1)
        directions = ray_bundle.directions.unsqueeze(1).repeat(1, N, 1) 
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(O, 1, 1)
        
        print("origins: ", origins.shape)      # Expected: [N, 3] (N = number of rays)
        print("directions: ", directions.shape)   # Expected: [N, 3]
        print("vals: ",z_vals.shape)                  # Expected: [N, 64] or [N, 64, 1]
        
        sample_points = origins + z_vals * directions
        print("sample points: ", sample_points.shape)
        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}