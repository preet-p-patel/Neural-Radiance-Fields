import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

# Question 8.1 - Rendering multiple SDF using primitives
class ComplexSceneSDF(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.primitives = torch.nn.ModuleList([
            TorusSDF(cfg.torus1),
            TorusSDF(cfg.torus2),
            TorusSDF(cfg.torus3),
            TorusSDF(cfg.torus4)
        ])

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.torus1.center.val).float().unsqueeze(0), requires_grad=cfg.torus1.center.opt
        )

        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.torus1.radii.val).float().unsqueeze(0), requires_grad=cfg.torus1.radii.opt
        )
        
        self.smooth_union_k = cfg.smooth_union_k if hasattr(cfg, "smooth_union_k") else 0.0

    def smooth_union(self, d1, d2, k):
        if k > 0.0:
            res = torch.exp(-k * d1) + torch.exp(-k * d2)
            return -torch.log(torch.clamp(res, min=1e-6)) / k
        else:
            return torch.minimum(d1, d2)

    def forward(self, points):
        sdf = self.primitives[0](points)
        
        for primitive in self.primitives[1:]:
            sdf = self.smooth_union(sdf, primitive(points), self.smooth_union_k)
        
        return sdf

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
    'complex_sdf': ComplexSceneSDF
}

# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim

        n_x = cfg.n_hidden_neurons_xyz
        n_d = cfg.n_hidden_neurons_dir

        self.layer1 = torch.nn.Linear(embedding_dim_xyz, n_x) 
        self.layer2 = torch.nn.Linear(n_x, n_x)
        self.layer3 = torch.nn.Linear(n_x, n_x)
        self.layer4 = torch.nn.Linear(n_x, n_x)
        self.layer5 = torch.nn.Linear(n_x, n_x)
        
        self.layer6 = torch.nn.Linear(n_x + embedding_dim_xyz, n_x)
        self.layer7 = torch.nn.Linear(n_x, n_x)
        self.layer8 = torch.nn.Linear(n_x, n_x)
        
        self.density_output = torch.nn.Linear(n_x, 1) 
        self.bottleneck = torch.nn.Linear(n_x, n_d) 
       
        self.color_layer1 = torch.nn.Linear(n_d + embedding_dim_dir, n_d)
        self.color_output = torch.nn.Linear(n_d, 3)  
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()



    def forward(self, ray_bundle):
        pts = ray_bundle.sample_points
        dir = ray_bundle.directions

        B, N, _ = pts.shape

        emb_pos = self.harmonic_embedding_xyz(pts)
        emb_dir = self.harmonic_embedding_dir(dir).unsqueeze(1).repeat(1, N, 1)

        x = self.relu(self.layer1(emb_pos))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        
        x = torch.cat([x, emb_pos], dim=-1)
        
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))
        
        density = self.relu(self.density_output(x))
        
        bottleneck_features = self.relu(self.bottleneck(x))
       
        color_input = torch.cat([bottleneck_features, emb_dir], dim=-1)
        
        color_features = self.relu(self.color_layer1(color_input))
        color = self.sigmoid(self.color_output(color_features)) 
        
        density = density.reshape(B, N, 1)
        color = color.reshape(B, N, 3)
        
        return {
            'density': density,
            'feature': color}

         


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        n_x = cfg.n_hidden_neurons_distance

        self.layer1 = torch.nn.Linear(embedding_dim_xyz, n_x) 
        self.layer2 = torch.nn.Linear(n_x, n_x)
        self.layer3 = torch.nn.Linear(n_x, n_x)
        self.layer4 = torch.nn.Linear(n_x, n_x)
        self.layer5 = torch.nn.Linear(n_x, n_x)

        self.layer6 = torch.nn.Linear(n_x + embedding_dim_xyz, n_x)
        self.layer7 = torch.nn.Linear(n_x, n_x)
        self.layer8 = torch.nn.Linear(n_x, n_x)

        self.dist_out = torch.nn.Linear(n_x, 1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        self.color1 = torch.nn.Linear(n_x + embedding_dim_xyz, n_x)
        self.color2 = torch.nn.Linear(n_x, n_x)
        self.color_out = torch.nn.Linear(n_x, 3)

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        emb_pos = self.harmonic_embedding_xyz(points)

        x = self.relu(self.layer1(emb_pos))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))

        x = torch.cat([x, emb_pos], dim=-1)
        
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8(x))

        distance = (self.dist_out(x))   
        distance.view(-1,1)
        return distance
        
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        emb_pos = self.harmonic_embedding_xyz(points)

        x = self.relu(self.layer1(emb_pos))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))

        x = torch.cat([x, emb_pos], dim=-1)

        x = self.relu(self.color1(x))
        x = self.relu(self.color2(x))
        color = self.sigmoid(self.color_out(x))
        color.view(-1,1)
        return color
    
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3)
        emb_pos = self.harmonic_embedding_xyz(points)

        x = self.relu(self.layer1(emb_pos))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))

        x = torch.cat([x, emb_pos], dim=-1)
        
        dist = self.relu(self.layer6(x))
        dist = self.relu(self.layer7(dist))
        dist = self.relu(self.layer8(dist))
        distance = self.dist_out(dist)

        color = self.relu(self.color1(x))
        color = self.relu(self.color2(color))
        color = self.sigmoid(self.color_out(color))
        
        color.view(-1,1)   
        distance.view(-1,1)

        return distance, color
        
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
