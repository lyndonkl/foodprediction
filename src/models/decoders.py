from __future__ import annotations

from torch import nn, Tensor


class DotProductDecoder(nn.Module):
    def forward(self, z_src: Tensor, z_dst: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        return (z_src[src] * z_dst[dst]).sum(dim=-1)


