import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
# class StratifiedRaysampler(torch.nn.Module):
#     def __init__(
#         self,
#         cfg
#     ):
#         super().__init__()

#         self.n_pts_per_ray = cfg.n_pts_per_ray
#         self.min_depth = cfg.min_depth
#         self.max_depth = cfg.max_depth

#     def forward(
#         self,
#         ray_bundle,
#     ):
#         # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
#         z_vals = torch.linspace(start=self.min_depth, end=self.max_depth, steps=self.n_pts_per_ray, device='cuda')
#         # TODO (Q1.4): Sample points from z values
#         # z_vals = z_vals.view(1, 64, 1)
#         O = ray_bundle.origins.shape[0]
#         N = z_vals.shape[0] 
#         origins = ray_bundle.origins.unsqueeze(1).repeat(1, N, 1)
#         directions = ray_bundle.directions.unsqueeze(1).repeat(1, N, 1) 
#         z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(O, 1, 1)
        
#         # print("origins: ", origins.shape)      # Expected: [N, 3] (N = number of rays)
#         # print("directions: ", directions.shape)   # Expected: [N, 3]
#         # print("vals: ",z_vals.shape)                  # Expected: [N, 64] or [N, 64, 1]
        
#         sample_points = origins + z_vals * directions
#         # print("sample points: ", sample_points.shape)
#         # Return
#         return ray_bundle._replace(
#             sample_points=sample_points,
#             sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
#         )

# New Inverse transform Sampling
class StratifiedRaysampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(self, ray_bundle, density=None):
        """
        Samples points along the ray with more points in regions of higher density.

        Args:
            ray_bundle: Object containing ray origins and directions.
            density: Tensor of shape [O, self.n_pts_per_ray], representing density along each ray.

        Returns:
            Updated ray_bundle with sampled points and lengths.
        """
        if density == None:
            z_vals = torch.linspace(start=self.min_depth, end=self.max_depth, steps=self.n_pts_per_ray, device='cuda')
            # TODO (Q1.4): Sample points from z values
            # z_vals = z_vals.view(1, 64, 1)
            O = ray_bundle.origins.shape[0]
            N = z_vals.shape[0] 
            origins = ray_bundle.origins.unsqueeze(1).repeat(1, N, 1)
            directions = ray_bundle.directions.unsqueeze(1).repeat(1, N, 1) 
            z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(O, 1, 1)
            
            # print("origins: ", origins.shape)      # Expected: [N, 3] (N = number of rays)
            # print("directions: ", directions.shape)   # Expected: [N, 3]
            # print("vals: ",z_vals.shape)                  # Expected: [N, 64] or [N, 64, 1]
            
            sample_points = origins + z_vals * directions
            # print("sample points: ", sample_points.shape)
            # Return
            return ray_bundle._replace(
                sample_points=sample_points,
                sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
            )
        
        else:
            O = ray_bundle.origins.shape[0]  # Number of rays

            # Compute uniform z values (initial guess)
            z_vals_uniform = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray, device='cuda')
            z_vals_uniform = z_vals_uniform.unsqueeze(0).expand(O, -1)  # Shape: [O, self.n_pts_per_ray]

            # Compute CDF from density (normalized)
            density += 1e-5  # Avoid zeros
            pdf = density / torch.sum(density, dim=-1, keepdim=True)  # Probability distribution
            cdf = torch.cumsum(pdf, dim=-1)  # Cumulative distribution function
            cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # Ensure starts at 0

            # Sample more points in high-density regions
            u = torch.rand(O, self.n_pts_per_ray, device='cuda')  # Uniform samples
            z_vals = torch.searchsorted(cdf, u, right=True)  # Get indices in CDF
            z_vals = torch.clamp(z_vals, 1, self.n_pts_per_ray) - 1  # Clamp to avoid out of bounds

            # Convert indices to actual z values
            z_vals = torch.gather(z_vals_uniform, dim=1, index=z_vals)  # Shape: [O, self.n_pts_per_ray]
            z_vals = z_vals.unsqueeze(-1)  # Shape: [O, self.n_pts_per_ray, 1]

            # Compute sample points
            origins = ray_bundle.origins.unsqueeze(1).expand(-1, self.n_pts_per_ray, -1)  # [O, N, 3]
            directions = ray_bundle.directions.unsqueeze(1).expand(-1, self.n_pts_per_ray, -1)  # [O, N, 3]
            sample_points = origins + z_vals * directions  # [O, N, 3]

            # Return updated ray bundle
            return ray_bundle._replace(
                sample_points=sample_points,
                sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
            )
    

sampler_dict = {
    'stratified': StratifiedRaysampler
}