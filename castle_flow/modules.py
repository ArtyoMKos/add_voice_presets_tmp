import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from audiokit.nn.utils import timestep_embedding


class FusionLayer(nn.Module):
    def __init__(self, hidden_size: int, input_sizes: list[int]) -> None:
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Conv1d(size, hidden_size, 1) for size in input_sizes
        ])
        self.fusion = nn.Conv1d(len(input_sizes) * hidden_size, hidden_size, 1)
        
    def forward(
        self,
        inputs: list[torch.Tensor],
        mask: torch.Tensor
    ) -> torch.Tensor:
        inputs = [self.projections[i](x) * mask for i, x in enumerate(inputs)]
        inputs = torch.cat(inputs, dim=1)
        output = self.fusion(inputs) * mask
        return output


class FlowStepEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = timestep_embedding(x, self.hidden_size)
        x = self.proj(x)
        return x


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(
        self,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
        low_dim_project: bool = False
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if low_dim_project:
            self.in_proj = WNConv1d(
                input_dim, 
                codebook_dim, 
                kernel_size=1
            )
            self.out_proj = WNConv1d(
                codebook_dim, 
                input_dim, 
                kernel_size=1
            )
        else:
            codebook_dim = input_dim
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        self.low_dim_project = low_dim_project

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        if self.low_dim_project:
            z_e = self.in_proj(z)*mask  # z_e : (B x D x T)
        else:
            z_e = z

        z_q, indices = self.decode_latents(z_e)
        z_q_st = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass
        if self.low_dim_project:
            z_q_out = self.out_proj(z_q_st)
        else:
            z_q_out = z_q_st
        z_q = z_q * mask
        z_q_out = z_q_out * mask   
        return {
            'z_q': z_q,
            'z_e': z_e,
            'z_q_out': z_q_out,
            'indices': indices
        }
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices
