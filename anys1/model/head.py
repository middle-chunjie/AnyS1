"""PointerHead and OptionInteractionHead (oc) for AnyS1-4B (§3.3).

The backbone reads a question's candidate options causally in their listed order,
so option i cannot see options listed after it. ``OptionInteractionHead`` passes
the ``<decide>`` query and the K option keys (in pointer space, d_p=256) through
one pre-norm bidirectional self-attention + MLP block without positional encoding,
making option comparison permutation-equivariant over candidate orderings. Both
residual output projections are zero-initialized so initial logits match
``PointerHead`` exactly.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PointerHead(nn.Module):
    def __init__(self, d: int, dp: int = 256) -> None:
        super().__init__()
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1.0 / math.sqrt(dp)
        self.temperature: float = 1.0

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        """Compute option logits [K] from <decide> state [d] and option states [K, d]."""
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature

    def many(
        self,
        h_decide: torch.Tensor,
        h_opts: torch.Tensor,
        owner: torch.Tensor,
    ) -> torch.Tensor:
        """Batched forward across Q questions simultaneously."""
        z = (self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


class OptionInteractionHead(PointerHead):
    """PointerHead + one permutation-equivariant bidirectional option-comparison block."""

    def __init__(self, d: int, dp: int = 256, heads: int = 4, ff: int = 2) -> None:
        super().__init__(d, dp)
        self.norm1 = nn.LayerNorm(dp)
        self.norm2 = nn.LayerNorm(dp)
        self.attn = nn.MultiheadAttention(dp, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dp, ff * dp),
            nn.GELU(),
            nn.Linear(ff * dp, dp),
        )
        for lin in (self.attn.out_proj, self.ff[2]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _mix(self, X: torch.Tensor, pad: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(X)
        X = X + self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        return X + self.ff(self.norm2(X))

    def _out(self, z: torch.Tensor) -> torch.Tensor:
        return z if self.training or self.temperature == 1.0 else z / self.temperature

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:
        X = self._mix(torch.cat([self.q(h_decide)[None], self.k(h_opts)])[None])[0]
        return self._out((X[1:] @ X[0]) * self.scale)

    def many(
        self,
        h_decide: torch.Tensor,
        h_opts: torch.Tensor,
        owner: torch.Tensor,
    ) -> torch.Tensor:
        Q = h_decide.shape[0]
        ks = torch.bincount(owner, minlength=Q)
        starts = torch.cumsum(ks, 0) - ks
        slot = torch.arange(owner.shape[0], device=owner.device) - starts[owner]
        qv, kv = self.q(h_decide), self.k(h_opts)
        X = qv.new_zeros(Q, 1 + int(ks.max()), qv.shape[-1])
        X[:, 0] = qv
        X[owner, 1 + slot] = kv
        pad = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        pad[:, 0] = False
        pad[owner, 1 + slot] = False
        X = self._mix(X, pad)
        return self._out((X[owner, 1 + slot] * X[owner, 0]).sum(-1) * self.scale)


HEADS = {
    "pointer": PointerHead,
    "oc": OptionInteractionHead,
}
