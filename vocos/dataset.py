import math
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

    def __len__(self) -> int:
        return len(self.filelist)

    def _feature_path(self, audio_path: Path) -> Path:
        assert self.feature_root is not None and self.feature_source_root is not None
        try:
            relative_path = audio_path.relative_to(self.feature_source_root)
        except ValueError as exc:
            raise ValueError(f"Audio path is outside feature_source_root: {audio_path}") from exc
        return (self.feature_root / relative_path).with_suffix(".pt")

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
        return features.float()

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
