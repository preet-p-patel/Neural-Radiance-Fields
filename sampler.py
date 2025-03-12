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
            N = self.n_pts_per_ray  # Number of depth samples per ray

            # Compute initial uniform z values
            z_vals_uniform = torch.linspace(self.min_depth, self.max_depth, N, device='cuda')  # [N]
            z_vals_uniform = z_vals_uniform.unsqueeze(0).unsqueeze(-1).expand(O, N, 1)  # [O, N, 1]

            # Compute CDF from density (normalize to create probability distribution)
            density = density.squeeze(-1) + 1e-5  # Remove extra dimension and avoid zeros
            pdf = density / torch.sum(density, dim=-1, keepdim=True)  # PDF: [O, N]
            cdf = torch.cumsum(pdf, dim=-1)  # CDF: [O, N]
            cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # Add 0 at start → [O, N+1]

            # Sample more points in high-density regions
            u = torch.rand(O, N, device='cuda')  # Uniform random samples [O, N]
            indices = torch.searchsorted(cdf, u, right=True)  # Find indices in CDF [O, N]
            indices = torch.clamp(indices, 1, N) - 1  # Clamp to valid range [1, N] → [0, N-1]

            # Convert indices to actual depth values
            z_vals = torch.gather(z_vals_uniform, dim=1, index=indices.unsqueeze(-1))  # [O, N, 1]

            # Compute sample points
            origins = ray_bundle.origins.unsqueeze(1).expand(-1, N, -1)  # [O, N, 3]
            directions = ray_bundle.directions.unsqueeze(1).expand(-1, N, -1)  # [O, N, 3]
            sample_points = origins + z_vals * directions  # [O, N, 3]

            # Return updated ray bundle
            return ray_bundle._replace(
                sample_points=sample_points,
                sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
            )
    

sampler_dict = {
    'stratified': StratifiedRaysampler
}