import math
import json
import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

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
    # Optional corpus routing for one manifest that spans multiple audio roots.
    # Each feature mapping has ``audio_root`` and ``feature_root``.  Each
    # transform mapping has ``audio_root``, ``transform_dir``, and optionally
    # ``speaker_id_regex``.  The most specific matching audio root wins.
    feature_root_mappings: Optional[list[dict[str, str]]] = None
    speaker_transform_mappings: Optional[list[dict[str, str]]] = None
    feature_hop_length: int = 480
    feature_dim: Optional[int] = None
    use_speaker_transform: bool = False
    speaker_transform_dir: Optional[str] = None
    speaker_transform_output_dim: int = 1024
    speaker_transform_cache_size: int = 64
    use_speaker_embedding: bool = False
    speaker_embedding_dir: Optional[str] = None
    speaker_embedding_dim: Optional[int] = None
    speaker_embedding_cache_size: int = 64
    speaker_embedding_random_same_speaker: bool = False
    speaker_embedding_exclude_target: bool = True
    use_speaker_matrix: bool = False
    speaker_matrix_dir: Optional[str] = None
    speaker_matrix_input_dim: int = 75
    speaker_matrix_output_dim: int = 1024
    speaker_matrix_cache_size: int = 64
    # Choose one matrix uniformly per batch instead of deriving it from the
    # source utterance's speaker ID. Useful for randomized USCF transforms.
    speaker_matrix_random: bool = False
    # Optional path-based speaker-ID extractor.  It must contain one capture
    # group and permits non-LibriSpeech corpora such as Mixer6.
    speaker_id_regex: Optional[str] = None


@dataclass(frozen=True)
class CorpusFeatureMapping:
    audio_root: Path
    feature_root: Path
    speaker_transform_dir: Optional[Path]
    speaker_id_pattern: Optional[re.Pattern[str]]


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, train=train)
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=train,
            pin_memory=True,
            collate_fn=dataset.collate_batch if dataset.speaker_matrix_random else None,
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
        self.corpus_feature_mappings = self._parse_corpus_feature_mappings(cfg)
        if self.corpus_feature_mappings and (self.feature_root is not None or self.feature_source_root is not None):
            raise ValueError(
                "feature_root_mappings cannot be combined with feature_root or feature_source_root; "
                "use one routing mode."
            )
        if not self.corpus_feature_mappings and (self.feature_root is None) != (self.feature_source_root is None):
            raise ValueError("feature_root and feature_source_root must either both be set or both be omitted.")
        self.has_precomputed_features = bool(self.corpus_feature_mappings) or self.feature_root is not None
        if cfg.feature_hop_length <= 0:
            raise ValueError("feature_hop_length must be positive.")
        self.feature_hop_length = cfg.feature_hop_length
        self.feature_dim = cfg.feature_dim
        self.use_speaker_transform = cfg.use_speaker_transform
        self.speaker_transform_output_dim = cfg.speaker_transform_output_dim
        self.speaker_transform_cache_size = cfg.speaker_transform_cache_size
        self._speaker_transform_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.speaker_transform_dir = Path(cfg.speaker_transform_dir) if cfg.speaker_transform_dir else None
        self.use_speaker_embedding = cfg.use_speaker_embedding
        self.speaker_embedding_dim = cfg.speaker_embedding_dim
        self.speaker_embedding_cache_size = cfg.speaker_embedding_cache_size
        self._speaker_embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.speaker_embedding_dir = Path(cfg.speaker_embedding_dir) if cfg.speaker_embedding_dir else None
        self.speaker_embedding_random_same_speaker = cfg.speaker_embedding_random_same_speaker
        self.speaker_embedding_exclude_target = cfg.speaker_embedding_exclude_target
        self._speaker_embedding_candidates: dict[str, tuple[Path, ...]] = {}
        self.use_speaker_matrix = cfg.use_speaker_matrix
        self.speaker_matrix_input_dim = cfg.speaker_matrix_input_dim
        self.speaker_matrix_output_dim = cfg.speaker_matrix_output_dim
        self.speaker_matrix_cache_size = cfg.speaker_matrix_cache_size
        self._speaker_matrix_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.speaker_matrix_dir = Path(cfg.speaker_matrix_dir) if cfg.speaker_matrix_dir else None
        self.speaker_matrix_random = cfg.speaker_matrix_random
        self._random_speaker_matrix_paths: tuple[Path, ...] = ()
        self.speaker_id_pattern = None
        if cfg.speaker_id_regex is not None:
            try:
                self.speaker_id_pattern = re.compile(cfg.speaker_id_regex)
            except re.error as exc:
                raise ValueError(f"Invalid speaker_id_regex {cfg.speaker_id_regex!r}: {exc}") from exc
            if self.speaker_id_pattern.groups < 1:
                raise ValueError("speaker_id_regex must contain a capture group for the speaker ID.")
        if self.use_speaker_transform:
            if not self.has_precomputed_features or self.feature_dim is None:
                raise ValueError("Speaker transforms require precomputed features with a declared source feature_dim.")
            if not self.corpus_feature_mappings and self.speaker_transform_dir is None:
                raise ValueError("speaker_transform_dir is required when use_speaker_transform is true.")
            if self.speaker_transform_dir is not None and not self.speaker_transform_dir.is_dir():
                raise FileNotFoundError(f"Speaker-transform directory is missing: {self.speaker_transform_dir}")
            if any(mapping.speaker_transform_dir is None for mapping in self.corpus_feature_mappings):
                raise ValueError("Every feature-root mapping needs a matching speaker-transform mapping.")
            if self.speaker_transform_cache_size <= 0:
                raise ValueError("speaker_transform_cache_size must be positive.")
            self._validate_speaker_transform_provenance()
        if self.use_speaker_embedding:
            if self.speaker_embedding_dir is None:
                raise ValueError("speaker_embedding_dir is required when use_speaker_embedding is true.")
            if not self.speaker_embedding_dir.is_dir():
                raise FileNotFoundError(f"Speaker-embedding directory is missing: {self.speaker_embedding_dir}")
            if self.speaker_embedding_dim is None or self.speaker_embedding_dim <= 0:
                raise ValueError("speaker_embedding_dim must be a positive integer when use_speaker_embedding is true.")
            if self.speaker_embedding_cache_size <= 0:
                raise ValueError("speaker_embedding_cache_size must be positive.")
            if self.speaker_embedding_random_same_speaker:
                if self.feature_source_root is None:
                    raise ValueError(
                        "Random same-speaker embeddings require feature_source_root to map audio paths to embeddings."
                    )
                self._initialize_random_speaker_embedding_candidates()
        if self.use_speaker_matrix:
            if self.feature_dim is None:
                raise ValueError("Speaker matrices require precomputed features with a declared feature_dim.")
            if self.feature_dim != self.speaker_matrix_input_dim:
                raise ValueError(
                    "speaker_matrix_input_dim must equal feature_dim: "
                    f"{self.speaker_matrix_input_dim} != {self.feature_dim}."
                )
            if self.speaker_matrix_dir is None:
                raise ValueError("speaker_matrix_dir is required when use_speaker_matrix is true.")
            if not self.speaker_matrix_dir.is_dir():
                raise FileNotFoundError(f"Speaker-matrix directory is missing: {self.speaker_matrix_dir}")
            if self.speaker_matrix_cache_size <= 0:
                raise ValueError("speaker_matrix_cache_size must be positive.")
            if self.speaker_matrix_random:
                speakers_dir = self.speaker_matrix_dir / "speakers"
                matrix_root = speakers_dir if speakers_dir.is_dir() else self.speaker_matrix_dir
                paths = tuple(sorted(matrix_root.glob("*.npy")))
                if not paths:
                    raise FileNotFoundError(f"No random speaker matrices found under {matrix_root}")
                self._random_speaker_matrix_paths = paths

    @staticmethod
    def _parse_speaker_pattern(value: Optional[str], context: str) -> Optional[re.Pattern[str]]:
        if value is None:
            return None
        try:
            pattern = re.compile(value)
        except re.error as exc:
            raise ValueError(f"Invalid speaker_id_regex for {context}: {value!r}: {exc}") from exc
        if pattern.groups < 1:
            raise ValueError(f"speaker_id_regex for {context} must contain a capture group for the speaker ID.")
        return pattern

    def _parse_corpus_feature_mappings(self, cfg: DataConfig) -> tuple[CorpusFeatureMapping, ...]:
        feature_mappings = cfg.feature_root_mappings or []
        transform_mappings = cfg.speaker_transform_mappings or []
        if not feature_mappings and not transform_mappings:
            return ()
        if not feature_mappings:
            raise ValueError("speaker_transform_mappings requires feature_root_mappings.")

        transforms_by_root: dict[Path, dict[str, str]] = {}
        for mapping in transform_mappings:
            if not isinstance(mapping, dict):
                raise ValueError("Each speaker_transform_mappings entry must be a mapping.")
            try:
                audio_root = Path(mapping["audio_root"]).resolve(strict=True)
                transform_dir = Path(mapping["transform_dir"]).resolve(strict=True)
            except KeyError as exc:
                raise ValueError("speaker_transform_mappings entries require audio_root and transform_dir.") from exc
            if not transform_dir.is_dir():
                raise FileNotFoundError(f"Speaker-transform directory is missing: {transform_dir}")
            if audio_root in transforms_by_root:
                raise ValueError(f"Duplicate speaker-transform mapping for {audio_root}")
            transforms_by_root[audio_root] = {
                "transform_dir": str(transform_dir),
                "speaker_id_regex": mapping.get("speaker_id_regex"),
            }

        parsed: list[CorpusFeatureMapping] = []
        seen_roots: set[Path] = set()
        for mapping in feature_mappings:
            if not isinstance(mapping, dict):
                raise ValueError("Each feature_root_mappings entry must be a mapping.")
            try:
                audio_root = Path(mapping["audio_root"]).resolve(strict=True)
                feature_root = Path(mapping["feature_root"]).resolve(strict=True)
            except KeyError as exc:
                raise ValueError("feature_root_mappings entries require audio_root and feature_root.") from exc
            if not feature_root.is_dir():
                raise FileNotFoundError(f"Feature root is missing: {feature_root}")
            if audio_root in seen_roots:
                raise ValueError(f"Duplicate feature-root mapping for {audio_root}")
            seen_roots.add(audio_root)
            transform_mapping = transforms_by_root.pop(audio_root, None)
            transform_dir = Path(transform_mapping["transform_dir"]) if transform_mapping is not None else None
            speaker_pattern = self._parse_speaker_pattern(
                transform_mapping.get("speaker_id_regex") if transform_mapping is not None else None,
                str(audio_root),
            )
            parsed.append(CorpusFeatureMapping(audio_root, feature_root, transform_dir, speaker_pattern))
        if transforms_by_root:
            unknown_roots = ", ".join(str(path) for path in sorted(transforms_by_root))
            raise ValueError(f"speaker_transform_mappings has roots absent from feature_root_mappings: {unknown_roots}")
        return tuple(sorted(parsed, key=lambda item: len(str(item.audio_root)), reverse=True))

    def _corpus_mapping_for(self, audio_path: Path) -> Optional[CorpusFeatureMapping]:
        if not self.corpus_feature_mappings:
            return None
        resolved_audio = audio_path.resolve(strict=False)
        for mapping in self.corpus_feature_mappings:
            if resolved_audio.is_relative_to(mapping.audio_root):
                return mapping
        roots = ", ".join(str(mapping.audio_root) for mapping in self.corpus_feature_mappings)
        raise ValueError(f"Audio path is outside all feature-root mappings: {audio_path}; roots={roots}")

    def __len__(self) -> int:
        return len(self.filelist)

    def _feature_path(self, audio_path: Path) -> Path:
        mapping = self._corpus_mapping_for(audio_path)
        if mapping is not None:
            return (mapping.feature_root / audio_path.resolve(strict=False).relative_to(mapping.audio_root)).with_suffix(".pt")
        assert self.feature_root is not None and self.feature_source_root is not None
        try:
            relative_path = audio_path.relative_to(self.feature_source_root)
        except ValueError as exc:
            raise ValueError(f"Audio path is outside feature_source_root: {audio_path}") from exc
        return (self.feature_root / relative_path).with_suffix(".pt")

    def _validate_speaker_transform_provenance(self) -> None:
        """Reject known-incompatible LinearVC projection/decoder artifacts."""
        pairs: list[tuple[Path, Path]] = []
        if self.corpus_feature_mappings:
            pairs = [
                (mapping.feature_root, mapping.speaker_transform_dir)
                for mapping in self.corpus_feature_mappings
                if mapping.speaker_transform_dir is not None
            ]
        elif self.feature_root is not None and self.speaker_transform_dir is not None:
            pairs = [(self.feature_root, self.speaker_transform_dir)]
        for feature_root, transform_dir in pairs:
            conversion_manifest = feature_root / "conversion-provenance.json"
            if not conversion_manifest.is_file():
                conversion_manifest = feature_root / "extraction-metadata" / "extraction-provenance.json"
            transform_manifest = transform_dir / "speaker-transform-provenance.json"
            if not conversion_manifest.is_file() or not transform_manifest.is_file():
                continue
            conversion_projection = json.loads(conversion_manifest.read_text()).get("projection_sha256")
            transform_data = json.loads(transform_manifest.read_text())
            transform_projection = transform_data.get("projection_sha256")
            if conversion_projection == transform_projection:
                continue
            conversion_data = json.loads(conversion_manifest.read_text())
            conversion_path = conversion_data.get("projection_path")
            transform_path = transform_data.get("projection_path")
            projections_match = False
            if conversion_path and transform_path:
                conversion_array = np.load(conversion_path, mmap_mode="r")
                transform_array = np.load(transform_path, mmap_mode="r")
                projections_match = (
                    conversion_array.shape == transform_array.shape == (1024, 75)
                    and np.array_equal(conversion_array, transform_array)
                )
            elif conversion_path:
                # Some older speaker-transform provenance stores the digest of
                # the projection array rather than a path.  Compare the actual
                # feature projection against that array digest, as the mixed
                # BigVGAN manifest validator does.
                conversion_array = np.load(conversion_path, mmap_mode="r")
                conversion_array_digest = hashlib.sha256(
                    np.ascontiguousarray(conversion_array).tobytes()
                ).hexdigest()
                projections_match = (
                    conversion_array.shape == (1024, 75)
                    and conversion_array_digest == transform_projection
                )
            if not projections_match:
                raise ValueError(
                    "The 75-D feature conversion and speaker transforms use different content projections: "
                    f"{conversion_projection!r} != {transform_projection!r} for {feature_root}."
                )

    def _speaker_id(self, audio_path: Path) -> str:
        mapping = self._corpus_mapping_for(audio_path)
        pattern = mapping.speaker_id_pattern if mapping is not None else self.speaker_id_pattern
        if pattern is not None:
            match = pattern.search(str(audio_path))
            if match is None or not match.group(1):
                raise ValueError(
                    f"Cannot infer a speaker ID from {audio_path} using {pattern.pattern!r}"
                )
            return match.group(1)
        # LibriSpeech paths are <split>/<speaker>/<chapter>/<utterance>.flac.
        speaker_id = audio_path.parent.parent.name
        if not speaker_id.isdigit():
            raise ValueError(
                f"Cannot infer a LibriSpeech speaker ID from {audio_path}; set speaker_id_regex for this corpus."
            )
        return speaker_id

    def _speaker_transform_path(self, audio_path: Path, speaker_id: str) -> Path:
        mapping = self._corpus_mapping_for(audio_path)
        transform_dir = mapping.speaker_transform_dir if mapping is not None else self.speaker_transform_dir
        assert transform_dir is not None
        speakers_dir = transform_dir / "speakers"
        return (speakers_dir if speakers_dir.is_dir() else transform_dir) / f"{speaker_id}.npy"

    def _load_speaker_transform(self, audio_path: Path) -> torch.Tensor:
        speaker_id = self._speaker_id(audio_path)
        transform_path = self._speaker_transform_path(audio_path, speaker_id)
        cache_key = str(transform_path)
        transform = self._speaker_transform_cache.get(cache_key)
        if transform is not None:
            self._speaker_transform_cache.move_to_end(cache_key)
            return transform

        if not transform_path.is_file():
            raise FileNotFoundError(f"Speaker transform is missing for speaker {speaker_id}: {transform_path}")
        transform = torch.from_numpy(np.load(transform_path)).float().contiguous()
        expected_shape = (self.feature_dim, self.speaker_transform_output_dim)
        if tuple(transform.shape) != expected_shape:
            raise ValueError(
                f"Expected speaker {speaker_id} transform shape {expected_shape}, found {tuple(transform.shape)} "
                f"in {transform_path}."
            )
        self._speaker_transform_cache[cache_key] = transform
        if len(self._speaker_transform_cache) > self.speaker_transform_cache_size:
            self._speaker_transform_cache.popitem(last=False)
        return transform

    def _speaker_matrix_path(self, speaker_id: str) -> Path:
        assert self.speaker_matrix_dir is not None
        speakers_dir = self.speaker_matrix_dir / "speakers"
        return (speakers_dir if speakers_dir.is_dir() else self.speaker_matrix_dir) / f"{speaker_id}.npy"

    def _load_speaker_matrix(self, speaker_id: str) -> torch.Tensor:
        matrix = self._speaker_matrix_cache.get(speaker_id)
        if matrix is not None:
            self._speaker_matrix_cache.move_to_end(speaker_id)
            return matrix

        matrix_path = self._speaker_matrix_path(speaker_id)
        if not matrix_path.is_file():
            raise FileNotFoundError(f"Speaker matrix is missing for LibriSpeech speaker {speaker_id}: {matrix_path}")
        matrix = torch.from_numpy(np.load(matrix_path)).float().contiguous()
        expected_shape = (self.speaker_matrix_input_dim, self.speaker_matrix_output_dim)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Expected speaker {speaker_id} matrix shape {expected_shape}, found {tuple(matrix.shape)} "
                f"in {matrix_path}."
            )
        self._speaker_matrix_cache[speaker_id] = matrix
        if len(self._speaker_matrix_cache) > self.speaker_matrix_cache_size:
            self._speaker_matrix_cache.popitem(last=False)
        return matrix

    def _load_random_speaker_matrix(self) -> torch.Tensor:
        if not self._random_speaker_matrix_paths:
            raise RuntimeError("Random speaker-matrix sampling was not initialized.")
        if self.train:
            matrix_path = self._random_speaker_matrix_paths[np.random.randint(len(self._random_speaker_matrix_paths))]
        else:
            # Validation uses a fixed transform so its loss remains comparable
            # across checkpoints.
            matrix_path = self._random_speaker_matrix_paths[0]
        cache_key = f"random:{matrix_path}"
        matrix = self._speaker_matrix_cache.get(cache_key)
        if matrix is not None:
            self._speaker_matrix_cache.move_to_end(cache_key)
            return matrix
        matrix = torch.from_numpy(np.load(matrix_path)).float().contiguous()
        expected_shape = (self.speaker_matrix_input_dim, self.speaker_matrix_output_dim)
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Expected random speaker matrix shape {expected_shape}, found {tuple(matrix.shape)} in {matrix_path}."
            )
        self._speaker_matrix_cache[cache_key] = matrix
        if len(self._speaker_matrix_cache) > self.speaker_matrix_cache_size:
            self._speaker_matrix_cache.popitem(last=False)
        return matrix

    def collate_batch(self, items):
        batch = default_collate(items)
        if not isinstance(batch, dict):
            raise TypeError("Random speaker-matrix batches must contain mapping samples.")
        matrix = self._load_random_speaker_matrix()
        batch_size = batch["audio"].size(0)
        batch["speaker_matrix"] = matrix.unsqueeze(0).expand(batch_size, -1, -1).clone()
        return batch

    def _speaker_embedding_path(self, speaker_id: str) -> Path:
        assert self.speaker_embedding_dir is not None
        speakers_dir = self.speaker_embedding_dir / "speakers"
        return (speakers_dir if speakers_dir.is_dir() else self.speaker_embedding_dir) / f"{speaker_id}.npy"

    def _utterance_embedding_root(self) -> Path:
        assert self.speaker_embedding_dir is not None
        utterances_dir = self.speaker_embedding_dir / "utterances"
        return utterances_dir if utterances_dir.is_dir() else self.speaker_embedding_dir

    def _utterance_embedding_path(self, audio_path: Path) -> Path:
        assert self.feature_source_root is not None
        try:
            relative_path = audio_path.relative_to(self.feature_source_root)
        except ValueError as exc:
            raise ValueError(f"Audio path is outside feature_source_root: {audio_path}") from exc
        return (self._utterance_embedding_root() / relative_path).with_suffix(".npy")

    def _initialize_random_speaker_embedding_candidates(self) -> None:
        candidates: dict[str, list[Path]] = {}
        for file_name in self.filelist:
            audio_path = Path(file_name)
            candidates.setdefault(self._speaker_id(audio_path), []).append(audio_path)
        self._speaker_embedding_candidates = {
            speaker_id: tuple(sorted(paths)) for speaker_id, paths in candidates.items()
        }

    def _select_same_speaker_embedding_path(self, audio_path: Path) -> Path:
        speaker_id = self._speaker_id(audio_path)
        candidates = self._speaker_embedding_candidates.get(speaker_id, ())
        if self.speaker_embedding_exclude_target:
            candidates = tuple(path for path in candidates if path != audio_path)
        if not candidates:
            exclusion = "excluding the target utterance " if self.speaker_embedding_exclude_target else ""
            raise ValueError(
                f"No same-speaker ECAPA candidates found for {audio_path} ({exclusion}for speaker {speaker_id})."
            )
        if self.train:
            return candidates[np.random.randint(len(candidates))]
        # Keep validation repeatable while still using a non-target utterance.
        digest = hashlib.blake2b(str(audio_path).encode(), digest_size=8).digest()
        return candidates[int.from_bytes(digest, "little") % len(candidates)]

    def _load_speaker_embedding(self, speaker_id: str) -> torch.Tensor:
        embedding = self._speaker_embedding_cache.get(speaker_id)
        if embedding is not None:
            self._speaker_embedding_cache.move_to_end(speaker_id)
            return embedding

        embedding_path = self._speaker_embedding_path(speaker_id)
        if not embedding_path.is_file():
            raise FileNotFoundError(f"Speaker embedding is missing for LibriSpeech speaker {speaker_id}: {embedding_path}")
        embedding = torch.from_numpy(np.load(embedding_path)).float().contiguous().squeeze()
        expected_shape = (self.speaker_embedding_dim,)
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"Expected speaker {speaker_id} embedding shape {expected_shape}, found {tuple(embedding.shape)} "
                f"in {embedding_path}."
            )
        self._speaker_embedding_cache[speaker_id] = embedding
        if len(self._speaker_embedding_cache) > self.speaker_embedding_cache_size:
            self._speaker_embedding_cache.popitem(last=False)
        return embedding

    def _load_utterance_speaker_embedding(self, audio_path: Path) -> torch.Tensor:
        embedding_path = self._utterance_embedding_path(self._select_same_speaker_embedding_path(audio_path))
        cache_key = str(embedding_path)
        embedding = self._speaker_embedding_cache.get(cache_key)
        if embedding is not None:
            self._speaker_embedding_cache.move_to_end(cache_key)
            return embedding

        if not embedding_path.is_file():
            raise FileNotFoundError(f"ECAPA utterance embedding is missing: {embedding_path}")
        embedding = torch.from_numpy(np.load(embedding_path)).float().contiguous().squeeze()
        expected_shape = (self.speaker_embedding_dim,)
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"Expected utterance ECAPA embedding shape {expected_shape}, found {tuple(embedding.shape)} "
                f"in {embedding_path}."
            )
        self._speaker_embedding_cache[cache_key] = embedding
        if len(self._speaker_embedding_cache) > self.speaker_embedding_cache_size:
            self._speaker_embedding_cache.popitem(last=False)
        return embedding

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
            features = features @ self._load_speaker_transform(audio_path)
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

        features = self._load_features(audio_path) if self.has_precomputed_features else None
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
            # On-the-fly feature extractors still need any static speaker
            # conditioning. Preserve the compact tensor-only path only when
            # there is no conditioning to carry alongside the audio crop.
            if not self.use_speaker_embedding and not self.use_speaker_matrix:
                return y[0]
            batch = {"audio": y[0]}
        else:
            start_frame = 0 if not self.train else start // self.feature_hop_length
            required_frames = math.ceil(self.num_samples / self.feature_hop_length)
            features = features[start_frame : start_frame + required_frames]
            batch = {"audio": y[0], "features": features.transpose(0, 1)}
        if self.use_speaker_embedding:
            if self.speaker_embedding_random_same_speaker:
                batch["speaker_embedding"] = self._load_utterance_speaker_embedding(audio_path)
            else:
                batch["speaker_embedding"] = self._load_speaker_embedding(self._speaker_id(audio_path))
        if self.use_speaker_matrix and not self.speaker_matrix_random:
            # The matrix is a static per-speaker conditioning signal. It is
            # intentionally not applied to the 75-D features here.
            batch["speaker_matrix"] = self._load_speaker_matrix(self._speaker_id(audio_path))
        return batch
