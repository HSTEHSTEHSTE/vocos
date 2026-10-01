import torch
import torchaudio

from vocos.dataset import DataConfig, VocosDataset
from vocos.feature_extractors import PrecomputedWavLMFeatures


def _write_example_pair(tmp_path, frames=49, channels=4):
    audio_root = tmp_path / "audio"
    feature_root = tmp_path / "features"
    audio_path = audio_root / "train-clean-100" / "1" / "1-1.flac"
    audio_path.parent.mkdir(parents=True)
    torchaudio.save(audio_path, torch.zeros(1, 16_000), 16_000)
    feature_path = (feature_root / audio_path.relative_to(audio_root)).with_suffix(".pt")
    feature_path.parent.mkdir(parents=True)
    features = torch.arange(frames * channels, dtype=torch.float16).reshape(frames, channels)
    torch.save(features, feature_path)
    filelist = tmp_path / "files.txt"
    filelist.write_text(f"{audio_path}\n")
    return audio_root, feature_root, filelist, features


def test_dataset_loads_and_aligns_precomputed_wavlm_features(tmp_path):
    audio_root, feature_root, filelist, features = _write_example_pair(tmp_path)
    config = DataConfig(
        filelist_path=str(filelist),
        sampling_rate=24_000,
        num_samples=24_000,
        batch_size=1,
        num_workers=0,
        feature_root=str(feature_root),
        feature_source_root=str(audio_root),
        feature_hop_length=480,
        feature_dim=4,
    )

    batch = VocosDataset(config, train=False)[0]

    assert batch["audio"].shape == (24_000,)
    assert batch["features"].shape == (4, 50)
    torch.testing.assert_close(batch["features"][:, :-1], features.transpose(0, 1))
    torch.testing.assert_close(batch["features"][:, -1], features[-1].float())


def test_precomputed_wavlm_feature_extractor_is_identity():
    features = torch.randn(2, 4, 7)
    extractor = PrecomputedWavLMFeatures(feature_dim=4)

    output = extractor(features)

    assert output is features
