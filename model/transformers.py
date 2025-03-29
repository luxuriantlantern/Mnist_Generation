import torch
import torch.nn as nn
import math

class TranformerBlock(nn.Module):
    def __init__(self, 
                 input_dim: int, # input_dim 
                 hidden_dim: int, # hidden_dim
                 num_heads:int, 
                 dropout:float=0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor,) -> torch.Tensor:
        """
        Transformer block with self-attention and feed-forward network.
        :param x: input tensor, shape = [B, N, C]
        :param condition: condition tensor (optional), shape = [B, N, C]
        :return: 
        """
        residual = x
        x = self.norm(x)
        x, _ = self.self_attn(x, x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual

        return x

class PositionEmbedding(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 input_dim : int = 2,) -> None:
        """"
        Positional embedding for transformer
        :param hidden_dim: hidden dimension
        :param input_dim: input dimension
        :return: None
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.freq_dim = hidden_dim // input_dim // 2
        self.freq = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freq = 1.0 / (10000 ** self.freq)

    def _sin_cos_embedding(self,
                           x : torch.Tensor) -> torch.Tensor:
        """"
        Sinusoidal positional embedding
        :param x: input tensor, shape = [B * N * C]
        :return: positional embedding, shape = [B * N * C, hidden_dim // input_dim]
        """
        self.freqs = self.freq.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([out.sin(), out.cos()], dim=-1)
        return out

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        """
        Positional embedding for transformer
        :param x: input tensor, shape = [B, N, C]
        :return: embedded tensor, shape = [B, N, C]
        """
        B, N, C = x.shape
        assert N == self.input_dim, "Input dimension must be equal to input_dim"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(B, N, -1)
        if embed.shape[2] < self.hidden_dim:
            embed = torch.cat([embed, torch.zeros(B, N, self.hidden_dim - embed.shape[2], device=embed.device)], dim=-1)
        return embed

class TimeEmbedding(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 input_dim : int = 1,) -> None:
        """"
        Time embedding for transformer
        :param hidden_dim: hidden dimension
        :param input_dim: input dimension
        :return: None
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.freq_dim = hidden_dim // input_dim // 2
        self.freq = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freq = 1.0 / (10000 ** self.freq)

        self.time_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _sin_cos_embedding(self,
                           t : torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding
        :param t: time tensor, shape = [B * N * C]
        :return: embedded tensor, shape = [B * N * C, hidden_dim // input_dim]
        """
        self.freqs = self.freq.to(t.device)
        out = torch.outer(t, self.freqs)
        out = torch.cat([out.sin(), out.cos()], dim=-1)
        return out

    def forward(self,
                t : torch.Tensor) -> torch.Tensor:
        """
        Time embedding for transformer
        :param t: time tensor, shape = [B, N, C]
        :return: embedded tensor, shape = [B, N, C]
        """
        B, N, C = t.shape
        assert N == self.input_dim, "Input dimension must be equal to input_dim"
        embed = self._sin_cos_embedding(t.reshape(-1))
        embed = embed.reshape(B, N, -1)
        if embed.shape[2] < self.hidden_dim:
            embed = torch.cat([embed, torch.zeros(B, N, self.hidden_dim - embed.shape[2], device=embed.device)], dim=-1)
        embed = self.time_mlp(embed)
        return embed

class TransformerCrossBlock(nn.Module):
    """
    Transformer cross block for cross attention
    """
    def __init__(self,
                 hidden_dim : int,
                 num_heads : int,
                 dropout : float = 0.1,
                 ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_kv = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self,
                x : torch.Tensor,
                context : torch.Tensor) -> torch.Tensor:
        """
        Transformer cross block for cross attention
        :param x: input tensor, shape = [B, N, C]
        :param context: context tensor, shape = [B, N, C * 2]
        :return: output tensor, shape = [B, N, C]
        """
        # self attention layer
        residual = x
        x = self.norm(x)
        self_attn_out, _ = self.self_attn(x, x, x)
        self_attn_out = self.dropout(self_attn_out)
        self_attn_out = self.dropout(self_attn_out)
        x = residual + self_attn_out

        # cross attention layer
        residual = x
        x = self.norm(x)

        q = self.to_q(x) # [B, N, C]
        k, v = self.to_kv(context).chunk(2, dim=-1) # [B, N, C]
        B, N, C = q.shape

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]

        out = torch.einsum('bhnd,bhmd->bnm', q, k) * self.scale
        out = torch.softmax(out, dim=-1)

        out = torch.einsum('bnm,bhmd->bhnd', out, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)

        x = self.dropout(out) + residual

        # feed forward layer
        residual = x
        x = self.norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + residual

        return x

class Transformer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.position_embedding = PositionEmbedding(hidden_dim)
        self.time_embedding = TimeEmbedding(hidden_dim)

        self.layers = nn.ModuleList(
            [TranformerBlock(input_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.cross_layers = nn.ModuleList(
            [TransformerCrossBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x : torch.Tensor,
                t : float) -> torch.Tensor:
        """
        Forward pass for transformer
        :param x: pixels tensor, shape = [B, N, C]
        :param t: time,
        :return: shape = [B, N, C]
        """
