import math
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int
    feature_root: Optional[str] = None
    feature_source_root: Optional[str] = None
    feature_hop_length: int = 480
    feature_dim: Optional[int] = None
    use_speaker_transform: bool = False
    speaker_transform_dir: Optional[str] = None
    speaker_transform_output_dim: int = 1024
    speaker_transform_cache_size: int = 64


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.feature_root = Path(cfg.feature_root) if cfg.feature_root is not None else None
        self.feature_source_root = Path(cfg.feature_source_root) if cfg.feature_source_root is not None else None
        if (self.feature_root is None) != (self.feature_source_root is None):
            raise ValueError("feature_root and feature_source_root must either both be set or both be omitted.")
        if cfg.feature_hop_length <= 0:
            raise ValueError("feature_hop_length must be positive.")
        self.feature_hop_length = cfg.feature_hop_length
        self.feature_dim = cfg.feature_dim
        self.use_speaker_transform = cfg.use_speaker_transform
        self.speaker_transform_output_dim = cfg.speaker_transform_output_dim
        self.speaker_transform_cache_size = cfg.speaker_transform_cache_size
        self._speaker_transform_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.speaker_transform_dir = Path(cfg.speaker_transform_dir) if cfg.speaker_transform_dir else None
        if self.use_speaker_transform:
            if self.feature_root is None or self.feature_source_root is None or self.feature_dim is None:
                raise ValueError("Speaker transforms require precomputed features with a declared source feature_dim.")
            if self.speaker_transform_dir is None:
                raise ValueError("speaker_transform_dir is required when use_speaker_transform is true.")
            if not self.speaker_transform_dir.is_dir():
                raise FileNotFoundError(f"Speaker-transform directory is missing: {self.speaker_transform_dir}")
            if self.speaker_transform_cache_size <= 0:
                raise ValueError("speaker_transform_cache_size must be positive.")
            self._validate_speaker_transform_provenance()

    def __len__(self) -> int:
        return len(self.filelist)

    def _feature_path(self, audio_path: Path) -> Path:
        assert self.feature_root is not None and self.feature_source_root is not None
        try:
            relative_path = audio_path.relative_to(self.feature_source_root)
        except ValueError as exc:
            raise ValueError(f"Audio path is outside feature_source_root: {audio_path}") from exc
        return (self.feature_root / relative_path).with_suffix(".pt")

    def _validate_speaker_transform_provenance(self) -> None:
        """Reject known-incompatible LinearVC projection/decoder artifacts."""
        assert self.feature_root is not None and self.speaker_transform_dir is not None
        conversion_manifest = self.feature_root / "conversion-provenance.json"
        transform_manifest = self.speaker_transform_dir / "speaker-transform-provenance.json"
        if not conversion_manifest.is_file() or not transform_manifest.is_file():
            return
        conversion_projection = json.loads(conversion_manifest.read_text()).get("projection_sha256")
        transform_projection = json.loads(transform_manifest.read_text()).get("projection_sha256")
        if conversion_projection != transform_projection:
            raise ValueError(
                "The 75-D feature conversion and speaker transforms use different content projections: "
                f"{conversion_projection!r} != {transform_projection!r}."
            )

    @staticmethod
    def _speaker_id(audio_path: Path) -> str:
        # LibriSpeech paths are <split>/<speaker>/<chapter>/<utterance>.flac.
        speaker_id = audio_path.parent.parent.name
        if not speaker_id.isdigit():
            raise ValueError(f"Cannot infer a LibriSpeech speaker ID from {audio_path}")
        return speaker_id

    def _speaker_transform_path(self, speaker_id: str) -> Path:
        assert self.speaker_transform_dir is not None
        speakers_dir = self.speaker_transform_dir / "speakers"
        return (speakers_dir if speakers_dir.is_dir() else self.speaker_transform_dir) / f"{speaker_id}.npy"

    def _load_speaker_transform(self, speaker_id: str) -> torch.Tensor:
        transform = self._speaker_transform_cache.get(speaker_id)
        if transform is not None:
            self._speaker_transform_cache.move_to_end(speaker_id)
            return transform

        transform_path = self._speaker_transform_path(speaker_id)
        if not transform_path.is_file():
            raise FileNotFoundError(f"Speaker transform is missing for LibriSpeech speaker {speaker_id}: {transform_path}")
        transform = torch.from_numpy(np.load(transform_path)).float().contiguous()
        expected_shape = (self.feature_dim, self.speaker_transform_output_dim)
        if tuple(transform.shape) != expected_shape:
            raise ValueError(
                f"Expected speaker {speaker_id} transform shape {expected_shape}, found {tuple(transform.shape)} "
                f"in {transform_path}."
            )
        self._speaker_transform_cache[speaker_id] = transform
        if len(self._speaker_transform_cache) > self.speaker_transform_cache_size:
            self._speaker_transform_cache.popitem(last=False)
        return transform

    def _load_features(self, audio_path: Path) -> torch.Tensor:
        feature_path = self._feature_path(audio_path)
        if not feature_path.is_file():
            raise FileNotFoundError(f"Precomputed WavLM feature file is missing: {feature_path}")
        try:
            features = torch.load(feature_path, map_location="cpu", weights_only=True)
        except TypeError:
            features = torch.load(feature_path, map_location="cpu")
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError(f"Expected a rank-2 tensor in {feature_path}, got {type(features)!r}.")
        if self.feature_dim is not None and features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected {self.feature_dim} WavLM channels in {feature_path}, found {features.size(1)}."
            )
        if features.size(0) == 0:
            raise ValueError(f"Precomputed WavLM feature file has no frames: {feature_path}")
        features = features.float()
        if self.use_speaker_transform:
            features = features @ self._load_speaker_transform(self._speaker_id(audio_path))
        return features

    @staticmethod
    def _repeat_to_length(values: torch.Tensor, length: int, dim: int) -> torch.Tensor:
        repeats = math.ceil(length / values.size(dim))
        repeat_shape = [1] * values.ndim
        repeat_shape[dim] = repeats
        return values.repeat(*repeat_shape).narrow(dim, 0, length)

    def _align_features_to_audio(self, features: torch.Tensor, audio_samples: int) -> torch.Tensor:
        target_frames = math.ceil(audio_samples / self.feature_hop_length)
        if features.size(0) < target_frames:
            features = torch.cat((features, features[-1:].expand(target_frames - features.size(0), -1)), dim=0)
        return features[:target_frames]

    @staticmethod
    def _peak_normalize(audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        peak = audio.abs().amax(dim=-1, keepdim=True)
        target_peak = 10 ** (gain_db / 20)
        return torch.where(peak > 0, audio * (target_peak / peak), audio)

    def __getitem__(self, index: int):
        audio_path = Path(self.filelist[index])
        y, sr = torchaudio.load(audio_path)
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        gain = np.random.uniform(-1, -6) if self.train else -3
        y = self._peak_normalize(y, float(gain))
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)

        features = self._load_features(audio_path) if self.feature_root is not None else None
        if features is not None:
            features = self._align_features_to_audio(features, y.size(-1))

        start = 0
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
            if features is not None:
                target_frames = math.ceil(y.size(-1) / self.feature_hop_length)
                features = self._repeat_to_length(features, target_frames, dim=0)
        elif self.train:
            if features is None:
                start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            else:
                required_frames = math.ceil(self.num_samples / self.feature_hop_length)
                max_start_frame = min(
                    (y.size(-1) - self.num_samples) // self.feature_hop_length,
                    features.size(0) - required_frames,
                )
                if max_start_frame < 0:
                    raise ValueError(f"Not enough precomputed feature frames for {audio_path}")
                start = np.random.randint(low=0, high=max_start_frame + 1) * self.feature_hop_length
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        if features is None:
            return y[0]

        start_frame = 0 if not self.train else start // self.feature_hop_length
        required_frames = math.ceil(self.num_samples / self.feature_hop_length)
        features = features[start_frame : start_frame + required_frames]
        return {"audio": y[0], "features": features.transpose(0, 1)}
