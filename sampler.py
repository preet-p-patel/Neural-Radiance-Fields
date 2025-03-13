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

# New Inverse transform Sampling for fine/coarse sampling
class StratifiedRaysampler(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(self, ray_bundle, density=None):
        if density == None:
            z_vals = torch.linspace(start=self.min_depth, end=self.max_depth, steps=self.n_pts_per_ray, device='cuda')
            # TODO (Q1.4): Sample points from z values
            
            O = ray_bundle.origins.shape[0]
            N = z_vals.shape[0] 
            origins = ray_bundle.origins.unsqueeze(1).repeat(1, N, 1)
            directions = ray_bundle.directions.unsqueeze(1).repeat(1, N, 1) 
            z_vals = z_vals.unsqueeze(0).unsqueeze(-1).repeat(O, 1, 1)
            
            sample_points = origins + z_vals * directions
            # Return
            return ray_bundle._replace(
                sample_points=sample_points,
                sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
            )
        
        else:
            O = ray_bundle.origins.shape[0]  
            N = self.n_pts_per_ray  

            z_vals_uniform = torch.linspace(self.min_depth, self.max_depth, N, device='cuda')  
            z_vals_uniform = z_vals_uniform.unsqueeze(0).unsqueeze(-1).expand(O, N, 1)  

            density = density.squeeze(-1) + 1e-5  
            pdf = density / torch.sum(density, dim=-1, keepdim=True)  
            cdf = torch.cumsum(pdf, dim=-1)  
            cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  

            u = torch.rand(O, N, device='cuda')  
            indices = torch.searchsorted(cdf, u, right=True)
            indices = torch.clamp(indices, 1, N) - 1

            z_vals = torch.gather(z_vals_uniform, dim=1, index=indices.unsqueeze(-1))  

            origins = ray_bundle.origins.unsqueeze(1).expand(-1, N, -1)  
            directions = ray_bundle.directions.unsqueeze(1).expand(-1, N, -1) 
            sample_points = origins + z_vals * directions

            return ray_bundle._replace(
                sample_points=sample_points,
                sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
            )
    

sampler_dict = {
    'stratified': StratifiedRaysampler
}