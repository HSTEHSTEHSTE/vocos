import math
from typing import List

import torch
import torchaudio
from encodec import EncodecModel
from torch import nn

from vocos.modules import safe_log


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(self, sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, padding="center"):
        super().__init__()
        if padding not in ["center", "same", "causal"]:
            raise ValueError("Padding must be 'center', 'same', or 'causal'.")
        self.padding = padding
        self.n_fft = n_fft
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def forward(self, audio, **kwargs):
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        elif self.padding == "causal":
            # Frame t ends at sample t * hop_length. Zero-padding only on the
            # left keeps each feature frame independent of future samples.
            audio = torch.nn.functional.pad(audio, (self.n_fft - 1, 0))
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: List[float] = [1.5, 3.0, 6.0, 12.0],
        train_codebooks: bool = False,
    ):
        super().__init__()
        if encodec_model == "encodec_24khz":
            encodec = EncodecModel.encodec_model_24khz
        elif encodec_model == "encodec_48khz":
            encodec = EncodecModel.encodec_model_48khz
        else:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        self.encodec = encodec(pretrained=True)
        for param in self.encodec.parameters():
            param.requires_grad = False
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate, bandwidth=max(bandwidths)
        )
        codebook_weights = torch.cat([vq.codebook for vq in self.encodec.quantizer.vq.layers[: self.num_q]], dim=0)
        self.codebook_weights = torch.nn.Parameter(codebook_weights, requires_grad=train_codebooks)
        self.bandwidths = bandwidths

    @torch.no_grad()
    def get_encodec_codes(self, audio):
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs):
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_id])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)


class WavLMFeatures(FeatureExtractor):
    """Frozen WavLM conditioning features aligned to a Vocos synthesis hop.

    WavLM is intentionally frozen by default. Its released checkpoints use
    bidirectional Transformer attention, so this extractor is suitable for
    training and buffered inference, but not zero-lookahead live feature
    extraction. The causal backbone and head can still decode externally
    supplied feature frames with zero vocoder lookahead.
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        input_sample_rate: int = 16000,
        source_sample_rate: int = 24000,
        output_hop_length: int = 480,
        feature_layer: int = -1,
        train_wavlm: bool = False,
    ):
        super().__init__()
        try:
            from transformers import WavLMModel
        except ImportError as exc:
            raise ImportError("WavLMFeatures requires the training dependencies. Install with `pip install vocos[train]`.") from exc
        self.input_sample_rate = input_sample_rate
        self.source_sample_rate = source_sample_rate
        self.output_hop_length = output_hop_length
        self.feature_layer = feature_layer
        self.train_wavlm = train_wavlm
        self.wavlm = WavLMModel.from_pretrained(model_name)
        if not train_wavlm:
            self.wavlm.requires_grad_(False)
            self.wavlm.eval()
        self.feature_dim = self.wavlm.config.hidden_size

    @staticmethod
    def _normalize(audio: torch.Tensor) -> torch.Tensor:
        variance = audio.var(dim=-1, keepdim=True, unbiased=False)
        return (audio - audio.mean(dim=-1, keepdim=True)) / variance.add(1e-7).sqrt()

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        output_sample_rate = kwargs.get("sample_rate", self.source_sample_rate)
        if output_sample_rate != self.input_sample_rate:
            audio_16khz = torchaudio.functional.resample(audio, output_sample_rate, self.input_sample_rate)
        else:
            audio_16khz = audio
        if not self.train_wavlm:
            self.wavlm.eval()
        outputs = self.wavlm(
            self._normalize(audio_16khz),
            output_hidden_states=self.feature_layer != -1,
            return_dict=True,
        )
        if self.feature_layer == -1:
            features = outputs.last_hidden_state
        else:
            features = outputs.hidden_states[self.feature_layer]

        # The convolutional WavLM frontend has a 400-sample receptive field, so
        # it can be one frame shorter than waveform_duration / 20 ms. Repeat the
        # final feature only to align the causal decoder's output duration.
        target_frames = math.ceil(audio.size(-1) / self.output_hop_length)
        if features.size(1) < target_frames:
            features = torch.nn.functional.pad(features, (0, 0, 0, target_frames - features.size(1)), mode="replicate")
        else:
            features = features[:, :target_frames]
        return features.transpose(1, 2)


class PrecomputedWavLMFeatures(FeatureExtractor):
    """Pass through pre-extracted WavLM features with shape ``(B, C, frames)``."""

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected precomputed WavLM features with shape (B, C, frames), got {tuple(features.shape)}.")
        if features.size(1) != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} feature channels, got {features.size(1)}.")
        return features
