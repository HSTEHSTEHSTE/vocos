from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from vocos.modules import AdaLayerNorm, CausalConv1d, CausalConvNeXtBlock, ConvNeXtBlock, ResBlock1


class Backbone(nn.Module):
    """Base class for the generator's backbone. It preserves the same temporal resolution across all layers."""

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                        C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                    and H denotes the model dimension.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class VocosBackbone(Backbone):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
                                                None means non-conditional model. Defaults to None.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class SpeakerMatrixEncoder(nn.Module):
    """Encode a static per-speaker matrix into a compact FiLM condition.

    The matrix is treated as a speaker descriptor, not as a linear operator on
    the input features. Its rows are encoded independently, mixed with a small
    Transformer, and attention-pooled once per utterance.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        condition_dim: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or condition_dim <= 0:
            raise ValueError("Speaker-matrix dimensions must be positive.")
        if num_layers <= 0 or num_heads <= 0 or condition_dim % num_heads:
            raise ValueError("speaker_matrix_condition_dim must be divisible by speaker_matrix_encoder_heads.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.condition_dim = condition_dim
        hidden_dim = hidden_dim or condition_dim
        self.row_encoder = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )
        self.row_positions = nn.Parameter(torch.zeros(input_dim, condition_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=condition_dim,
            nhead=num_heads,
            dim_feedforward=4 * condition_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.pool_score = nn.Sequential(nn.LayerNorm(condition_dim), nn.Linear(condition_dim, 1))
        self.output = nn.Sequential(
            nn.LayerNorm(condition_dim + 1),
            nn.Linear(condition_dim + 1, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def forward(self, speaker_matrix: torch.Tensor) -> torch.Tensor:
        if speaker_matrix.ndim != 3:
            raise ValueError(
                "speaker_matrix must have shape (batch, matrix_input_dim, matrix_output_dim); "
                f"received {tuple(speaker_matrix.shape)}."
            )
        if tuple(speaker_matrix.shape[1:]) != (self.input_dim, self.output_dim):
            raise ValueError(
                f"Expected speaker matrices with shape (*, {self.input_dim}, {self.output_dim}), "
                f"received {tuple(speaker_matrix.shape)}."
            )
        matrix_rms = speaker_matrix.square().mean(dim=(1, 2), keepdim=True).sqrt().clamp_min(1e-8)
        normalized_matrix = speaker_matrix / matrix_rms
        row_tokens = self.row_encoder(normalized_matrix) + self.row_positions.unsqueeze(0)
        row_tokens = self.transformer(row_tokens)
        pool_weights = self.pool_score(row_tokens).squeeze(-1).softmax(dim=-1)
        pooled = torch.sum(row_tokens * pool_weights.unsqueeze(-1), dim=1)
        log_rms = matrix_rms.squeeze(-1).squeeze(-1).log()
        return self.output(torch.cat((pooled, log_rms.unsqueeze(-1)), dim=-1))


class CausalVocosBackbone(Backbone):
    """A strictly causal, stateful ConvNeXt backbone for streaming Vocos.

    The model uses no right padding. ``forward_stream`` retains the convolution
    history for each layer, so concatenating chunk outputs matches ``forward``.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
        speaker_embedding_dim: Optional[int] = None,
        speaker_conditioner_hidden_dim: Optional[int] = None,
        speaker_matrix_input_dim: Optional[int] = None,
        speaker_matrix_output_dim: Optional[int] = None,
        speaker_matrix_condition_dim: int = 256,
        speaker_matrix_encoder_layers: int = 2,
        speaker_matrix_encoder_heads: int = 8,
        speaker_matrix_conditioner_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = CausalConv1d(input_channels, dim, kernel_size=7)
        self.adanorm = adanorm_num_embeddings is not None
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_matrix_input_dim = speaker_matrix_input_dim
        self.speaker_matrix_output_dim = speaker_matrix_output_dim
        if adanorm_num_embeddings:
            self.norm = AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                CausalConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        if speaker_embedding_dim is None:
            self.speaker_conditioners = None
        else:
            if speaker_embedding_dim <= 0:
                raise ValueError("speaker_embedding_dim must be positive when provided.")
            conditioner_hidden_dim = speaker_conditioner_hidden_dim or dim
            self.speaker_conditioners = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(speaker_embedding_dim),
                        nn.Linear(speaker_embedding_dim, conditioner_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(conditioner_hidden_dim, 2 * dim),
                    )
                    for _ in range(num_layers)
                ]
            )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)
        if self.speaker_conditioners is not None:
            for conditioner in self.speaker_conditioners:
                nn.init.zeros_(conditioner[-1].weight)
                nn.init.zeros_(conditioner[-1].bias)
        if (speaker_matrix_input_dim is None) != (speaker_matrix_output_dim is None):
            raise ValueError("speaker_matrix_input_dim and speaker_matrix_output_dim must be set together.")
        if speaker_matrix_input_dim is None:
            self.speaker_matrix_encoder = None
            self.speaker_matrix_conditioners = None
        else:
            self.speaker_matrix_encoder = SpeakerMatrixEncoder(
                input_dim=speaker_matrix_input_dim,
                output_dim=speaker_matrix_output_dim,
                condition_dim=speaker_matrix_condition_dim,
                num_layers=speaker_matrix_encoder_layers,
                num_heads=speaker_matrix_encoder_heads,
            )
            conditioner_hidden_dim = speaker_matrix_conditioner_hidden_dim or dim
            self.speaker_matrix_conditioners = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(speaker_matrix_condition_dim),
                        nn.Linear(speaker_matrix_condition_dim, conditioner_hidden_dim),
                        nn.SiLU(),
                        nn.Linear(conditioner_hidden_dim, 2 * dim),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.speaker_matrix_encoder.apply(self._init_weights)
            self.speaker_matrix_conditioners.apply(self._init_weights)
            nn.init.trunc_normal_(self.speaker_matrix_encoder.row_positions, std=0.02)
            for conditioner in self.speaker_matrix_conditioners:
                nn.init.zeros_(conditioner[-1].weight)
                nn.init.zeros_(conditioner[-1].bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _post_embed(self, x: torch.Tensor, bandwidth_id: Optional[torch.Tensor]) -> torch.Tensor:
        if self.adanorm:
            assert bandwidth_id is not None
            return self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id).transpose(1, 2)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def _apply_film_condition(x: torch.Tensor, conditioner: nn.Module, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = conditioner(condition).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

    def _speaker_embedding_condition(self, x: torch.Tensor, speaker_embedding: Optional[torch.Tensor]) -> torch.Tensor:
        if speaker_embedding is None:
            raise ValueError("speaker_embedding is required by this CausalVocosBackbone.")
        if speaker_embedding.ndim != 2 or speaker_embedding.size(0) != x.size(0):
            raise ValueError(
                "speaker_embedding must have shape (batch, embedding_dim); "
                f"received {tuple(speaker_embedding.shape)} for batch size {x.size(0)}."
            )
        if speaker_embedding.size(1) != self.speaker_embedding_dim:
            raise ValueError(
                f"Expected {self.speaker_embedding_dim}-D speaker embeddings, got {speaker_embedding.size(1)}."
            )
        return F.normalize(speaker_embedding, dim=-1, eps=1e-8)

    def _speaker_matrix_condition(self, x: torch.Tensor, speaker_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        if self.speaker_matrix_encoder is None:
            raise RuntimeError("speaker_matrix conditioning is not configured for this CausalVocosBackbone.")
        if speaker_matrix is None:
            raise ValueError("speaker_matrix is required by this CausalVocosBackbone.")
        if speaker_matrix.size(0) != x.size(0):
            raise ValueError(
                f"speaker_matrix batch size {speaker_matrix.size(0)} does not match feature batch size {x.size(0)}."
            )
        return self.speaker_matrix_encoder(speaker_matrix)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get("bandwidth_id")
        speaker_embedding = kwargs.get("speaker_embedding")
        speaker_matrix = kwargs.get("speaker_matrix")
        speaker_embedding_condition = (
            self._speaker_embedding_condition(x, speaker_embedding) if self.speaker_conditioners is not None else None
        )
        speaker_matrix_condition = (
            self._speaker_matrix_condition(x, speaker_matrix) if self.speaker_matrix_conditioners is not None else None
        )
        x = self._post_embed(self.embed(x), bandwidth_id)
        for index, block in enumerate(self.convnext):
            x = block(x, cond_embedding_id=bandwidth_id)
            if self.speaker_conditioners is not None:
                x = self._apply_film_condition(x, self.speaker_conditioners[index], speaker_embedding_condition)
            if self.speaker_matrix_conditioners is not None:
                x = self._apply_film_condition(x, self.speaker_matrix_conditioners[index], speaker_matrix_condition)
        return self.final_layer_norm(x.transpose(1, 2))

    def forward_stream(
        self, x: torch.Tensor, state: Optional[Dict[str, object]] = None, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode one feature chunk and return its output plus the next state."""
        bandwidth_id = kwargs.get("bandwidth_id")
        speaker_embedding = kwargs.get("speaker_embedding")
        speaker_matrix = kwargs.get("speaker_matrix")
        speaker_embedding_condition = (
            self._speaker_embedding_condition(x, speaker_embedding) if self.speaker_conditioners is not None else None
        )
        speaker_matrix_condition = (
            self._speaker_matrix_condition(x, speaker_matrix) if self.speaker_matrix_conditioners is not None else None
        )
        state = {} if state is None else state
        x, embed_state = self.embed.forward_stream(x, state.get("embed"))
        x = self._post_embed(x, bandwidth_id)

        block_states = state.get("blocks", [None] * len(self.convnext))
        next_block_states = []
        for index, (block, block_state) in enumerate(zip(self.convnext, block_states)):
            x, block_state = block.forward_stream(x, block_state, cond_embedding_id=bandwidth_id)
            if self.speaker_conditioners is not None:
                x = self._apply_film_condition(x, self.speaker_conditioners[index], speaker_embedding_condition)
            if self.speaker_matrix_conditioners is not None:
                x = self._apply_film_condition(x, self.speaker_matrix_conditioners[index], speaker_matrix_condition)
            next_block_states.append(block_state)
        return self.final_layer_norm(x.transpose(1, 2)), {"embed": embed_state, "blocks": next_block_states}


class VocosResNetBackbone(Backbone):
    """
    Vocos backbone module built with ResBlocks.

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        num_blocks (int): Number of ResBlock1 blocks.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to None.
    """

    def __init__(
        self, input_channels, dim, num_blocks, layer_scale_init_value=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = weight_norm(nn.Conv1d(input_channels, dim, kernel_size=3, padding=1))
        layer_scale_init_value = layer_scale_init_value or 1 / num_blocks / 3
        self.resnet = nn.Sequential(
            *[ResBlock1(dim=dim, layer_scale_init_value=layer_scale_init_value) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(x)
        x = self.resnet(x)
        x = x.transpose(1, 2)
        return x
