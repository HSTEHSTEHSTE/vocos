import math
import sys
import importlib.util
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from encodec import EncodecModel
from torch import nn

from vocos.modules import safe_log


def _load_official_wavlm_support():
    """Import the optional official-WavLM helper only when it is actually used.

    Most of the custom Vocos configurations consume precomputed features.  An
    eager import here made those configurations unusable in a lightweight
    inference delivery that intentionally did not include the sibling
    ``unilm/wavlm`` checkout.
    """

    wavlm_source_dir = Path(__file__).resolve().parents[2] / "unilm" / "wavlm"
    if not wavlm_source_dir.is_dir():
        raise ImportError(
            "WavLMFeatures requires the optional sibling unilm/wavlm checkout. "
            "Use PrecomputedWavLMFeatures when supplying extracted features."
        )
    if str(wavlm_source_dir) not in sys.path:
        sys.path.insert(0, str(wavlm_source_dir))
    try:
        from official_wavlm import OFFICIAL_WAVLM_LARGE_CHECKPOINT, load_official_wavlm_large
    except ImportError as exc:
        raise ImportError(
            "Unable to import the optional official_wavlm helper required by WavLMFeatures. "
            "Use PrecomputedWavLMFeatures when supplying extracted features."
        ) from exc
    return OFFICIAL_WAVLM_LARGE_CHECKPOINT, load_official_wavlm_large


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
        checkpoint_path: str | None = None,
        input_sample_rate: int = 16000,
        source_sample_rate: int = 24000,
        output_hop_length: int = 480,
        feature_layer: int = 6,
        train_wavlm: bool = False,
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.source_sample_rate = source_sample_rate
        self.output_hop_length = output_hop_length
        self.train_wavlm = train_wavlm
        default_checkpoint, load_official_wavlm_large = _load_official_wavlm_support()
        resolved_checkpoint_path = str(default_checkpoint) if checkpoint_path is None else checkpoint_path
        self.wavlm, config = load_official_wavlm_large(resolved_checkpoint_path)
        self.feature_layer = config.encoder_layers if feature_layer == -1 else feature_layer
        if not 1 <= self.feature_layer <= config.encoder_layers:
            raise ValueError(
                "feature_layer must be in [1, {}] (or -1 for the final layer), got {}".format(
                    config.encoder_layers, feature_layer
                )
            )
        if not train_wavlm:
            self.wavlm.requires_grad_(False)
            self.wavlm.eval()
        self.feature_dim = config.encoder_embed_dim

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        output_sample_rate = kwargs.get("sample_rate", self.source_sample_rate)
        if output_sample_rate != self.input_sample_rate:
            audio_16khz = torchaudio.functional.resample(audio, output_sample_rate, self.input_sample_rate)
        else:
            audio_16khz = audio
        if not self.train_wavlm:
            self.wavlm.eval()
        features, _ = self.wavlm.extract_features(audio_16khz, output_layer=self.feature_layer)

        # The convolutional WavLM frontend has a 400-sample receptive field, so
        # it can be one frame shorter than waveform_duration / 20 ms. Repeat the
        # final feature only to align the causal decoder's output duration.
        target_frames = math.ceil(audio.size(-1) / self.output_hop_length)
        if features.size(1) < target_frames:
            features = torch.nn.functional.pad(features, (0, 0, 0, target_frames - features.size(1)), mode="replicate")
        else:
            features = features[:, :target_frames]
        return features.transpose(1, 2)


class DistilledOneFrameWavLMContentFeatures(FeatureExtractor):
    """Frozen, one-pass causal WavLM student followed by a 1024->75 UTXSS.

    The student is the trained one-frame WavLM distillation model from the
    sibling ``one_frame_wavlm`` project.  It consumes an entire Vocos crop in
    one forward pass, emits one 1024-D frame every 20 ms, and is never part of
    the Vocos optimizer or checkpoint state.  The latter is deliberate: a
    Vocos checkpoint stays compact and rehydrates the immutable 1.9-GiB
    student from ``student_checkpoint_path`` on each worker.
    """

    def __init__(
        self,
        student_checkpoint_path: str,
        utxss_path: str,
        one_frame_source_dir: str = "/home/jhu/xli257/ARTS/one_frame_wavlm",
        source_sample_rate: int = 24000,
        student_sample_rate: int = 16000,
        output_hop_length: int = 480,
        output_layer: int = 6,
    ):
        super().__init__()
        if source_sample_rate <= 0 or student_sample_rate <= 0 or output_hop_length <= 0:
            raise ValueError("Sample rates and output_hop_length must be positive.")
        if output_layer < 1:
            raise ValueError("output_layer must be positive.")
        student_checkpoint = Path(student_checkpoint_path).expanduser().resolve()
        projection_path = Path(utxss_path).expanduser().resolve()
        source_dir = Path(one_frame_source_dir).expanduser().resolve()
        source_file = source_dir / "train.py"
        for required in (student_checkpoint, projection_path, source_file):
            if not required.is_file():
                raise FileNotFoundError(required)

        module_name = "_vocos_one_frame_wavlm_train"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, source_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to import one-frame WavLM source: {source_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        model_class = getattr(module, "WavLMLargeDistillation", None)
        if model_class is None:
            raise ImportError(f"{source_file} does not define WavLMLargeDistillation")

        checkpoint = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise ValueError(f"Invalid distilled one-frame WavLM checkpoint: {student_checkpoint}")
        metadata = checkpoint.get("args", {})
        if metadata.get("use_wavlm_large") is not True or metadata.get("wavlm_large_output_layer") != output_layer:
            raise ValueError(
                "The distilled student checkpoint does not match the requested one-frame WavLM output layer: "
                f"requested={output_layer}, checkpoint={metadata.get('wavlm_large_output_layer')!r}."
            )
        teacher_checkpoint = metadata.get("wavlm_large_checkpoint")
        if not teacher_checkpoint:
            raise ValueError(f"Distilled student checkpoint lacks wavlm_large_checkpoint metadata: {student_checkpoint}")
        student = model_class(checkpoint_path=teacher_checkpoint, warm_start=True, output_layer=output_layer)
        student.load_state_dict(checkpoint["model"], strict=True)
        student.requires_grad_(False)
        student.eval()

        projection = np.asarray(np.load(projection_path), dtype=np.float32)
        if projection.shape != (1024, 75):
            raise ValueError(f"Expected UTXSS shape (1024, 75), got {projection.shape} from {projection_path}")
        self.register_buffer("utxss", torch.from_numpy(np.ascontiguousarray(projection)), persistent=True)
        # Avoid registering the immutable 1.9-GiB student as a Vocos child
        # module: Lightning checkpoints therefore contain only trainable Vocos
        # state plus the tiny UTXSS buffer. _apply moves it with the parent.
        object.__setattr__(self, "student", student)
        self.source_sample_rate = source_sample_rate
        self.student_sample_rate = student_sample_rate
        self.output_hop_length = output_hop_length
        self.output_layer = output_layer
        self.student_checkpoint_path = str(student_checkpoint)
        self.utxss_path = str(projection_path)

    def _apply(self, fn):
        super()._apply(fn)
        self.student._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.student.eval()
        return self

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Expected audio shape (batch, samples), got {tuple(audio.shape)}")
        source_rate = int(kwargs.get("sample_rate", self.source_sample_rate))
        if source_rate != self.student_sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.student_sample_rate)
        if audio.size(-1) < 400:
            audio = torch.nn.functional.pad(audio, (0, 400 - audio.size(-1)))
        self.student.eval()
        with torch.no_grad():
            # Exactly one batched student forward per Vocos audio batch.
            features = self.student.extract_causal_features(audio)
        if features.ndim != 3 or features.size(-1) != self.utxss.size(0):
            raise RuntimeError(f"Unexpected distilled WavLM feature shape: {tuple(features.shape)}")
        content = features @ self.utxss
        target_frames = math.ceil(
            audio.size(-1) * source_rate / self.student_sample_rate / self.output_hop_length
        )
        if content.size(1) < target_frames:
            content = torch.nn.functional.pad(content, (0, 0, 0, target_frames - content.size(1)), mode="replicate")
        else:
            content = content[:, :target_frames]
        return content.transpose(1, 2)


class RandomUSCFTransformContentFeatures(FeatureExtractor):
    """Apply a sampled USCF transform and UTXSS while retaining 75-D input.

    Given precomputed one-frame content ``X`` and the same sampled Mixer6
    transform used for backbone conditioning ``S``, this emits
    ``X @ S @ UTXSS``.  The first product visits 1024-D only transiently; the
    Vocos backbone receives 75-D features and ``S`` separately as FiLM
    conditioning.
    """

    def __init__(self, utxss_path: str):
        super().__init__()
        projection_path = Path(utxss_path).expanduser().resolve()
        if not projection_path.is_file():
            raise FileNotFoundError(projection_path)
        projection = np.asarray(np.load(projection_path), dtype=np.float32)
        if projection.shape != (1024, 75):
            raise ValueError(f"Expected UTXSS shape (1024, 75), got {projection.shape} from {projection_path}")
        self.register_buffer("utxss", torch.from_numpy(np.ascontiguousarray(projection)), persistent=True)
        self.utxss_path = str(projection_path)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        matrix = kwargs.get("speaker_matrix")
        if features.ndim != 3 or features.size(1) != 75:
            raise ValueError(f"Expected 75-D features with shape (B, 75, T), got {tuple(features.shape)}")
        if matrix is None or matrix.ndim != 3 or tuple(matrix.shape[1:]) != (75, 1024):
            shape = None if matrix is None else tuple(matrix.shape)
            raise ValueError(f"Expected sampled speaker_matrix shape (B, 75, 1024), got {shape}")
        if matrix.size(0) != features.size(0):
            raise ValueError("speaker_matrix batch size must match the feature batch size.")
        content = torch.bmm(features.transpose(1, 2), matrix)
        return (content @ self.utxss).transpose(1, 2)


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
