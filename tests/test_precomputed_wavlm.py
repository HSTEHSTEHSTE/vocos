import numpy as np
import torch
import torchaudio

from vocos.dataset import DataConfig, VocosDataset
from vocos.feature_extractors import PrecomputedWavLMFeatures


def _write_example_pair(tmp_path, frames=49, channels=4):
    audio_root = tmp_path / "audio"
    feature_root = tmp_path / "features"
    audio_path = audio_root / "train-clean-100" / "1" / "1" / "1-1.flac"
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


def test_dataset_lifts_75d_content_features_with_the_matching_speaker_transform(tmp_path):
    audio_root, feature_root, filelist, features = _write_example_pair(tmp_path, frames=49, channels=3)
    transform_root = tmp_path / "speaker-transforms"
    (transform_root / "speakers").mkdir(parents=True)
    transform = np.arange(15, dtype=np.float32).reshape(3, 5)
    np.save(transform_root / "speakers" / "1.npy", transform)
    config = DataConfig(
        filelist_path=str(filelist),
        sampling_rate=24_000,
        num_samples=24_000,
        batch_size=1,
        num_workers=0,
        feature_root=str(feature_root),
        feature_source_root=str(audio_root),
        feature_hop_length=480,
        feature_dim=3,
        use_speaker_transform=True,
        speaker_transform_dir=str(transform_root),
        speaker_transform_output_dim=5,
    )

    batch = VocosDataset(config, train=False)[0]
    expected = features.float() @ torch.from_numpy(transform)

    assert batch["features"].shape == (5, 50)
    torch.testing.assert_close(batch["features"][:, :-1], expected.transpose(0, 1))
    torch.testing.assert_close(batch["features"][:, -1], expected[-1])


def test_dataset_returns_the_matching_speaker_embedding(tmp_path):
    audio_root, feature_root, filelist, _ = _write_example_pair(tmp_path, frames=49, channels=3)
    embedding_root = tmp_path / "speaker-embeddings"
    (embedding_root / "speakers").mkdir(parents=True)
    embedding = np.linspace(-1, 1, 5, dtype=np.float32)
    np.save(embedding_root / "speakers" / "1.npy", embedding)
    config = DataConfig(
        filelist_path=str(filelist),
        sampling_rate=24_000,
        num_samples=24_000,
        batch_size=1,
        num_workers=0,
        feature_root=str(feature_root),
        feature_source_root=str(audio_root),
        feature_hop_length=480,
        feature_dim=3,
        use_speaker_embedding=True,
        speaker_embedding_dir=str(embedding_root),
        speaker_embedding_dim=5,
    )

    batch = VocosDataset(config, train=False)[0]

    torch.testing.assert_close(batch["speaker_embedding"], torch.from_numpy(embedding))


def test_dataset_samples_another_utterance_embedding_from_the_same_speaker(tmp_path):
    audio_root = tmp_path / "audio"
    feature_root = tmp_path / "features"
    embedding_root = tmp_path / "utterance-embeddings" / "utterances"
    audio_paths = []
    for utterance_index, embedding_value in enumerate((1.0, 2.0, 3.0)):
        audio_path = audio_root / "train-clean-100" / "1" / "1" / f"1-1-{utterance_index:04d}.flac"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(audio_path, torch.zeros(1, 16_000), 16_000)
        feature_path = (feature_root / audio_path.relative_to(audio_root)).with_suffix(".pt")
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(49, 3), feature_path)
        embedding_path = (embedding_root / audio_path.relative_to(audio_root)).with_suffix(".npy")
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embedding_path, np.full(5, embedding_value, dtype=np.float32))
        audio_paths.append(audio_path)
    filelist = tmp_path / "files.txt"
    filelist.write_text("\n".join(map(str, audio_paths)) + "\n")
    config = DataConfig(
        filelist_path=str(filelist),
        sampling_rate=24_000,
        num_samples=24_000,
        batch_size=1,
        num_workers=0,
        feature_root=str(feature_root),
        feature_source_root=str(audio_root),
        feature_hop_length=480,
        feature_dim=3,
        use_speaker_embedding=True,
        speaker_embedding_dir=str(embedding_root.parent),
        speaker_embedding_dim=5,
        speaker_embedding_random_same_speaker=True,
        speaker_embedding_exclude_target=True,
    )

    dataset = VocosDataset(config, train=False)
    for index, expected_target_value in enumerate((1.0, 2.0, 3.0)):
        embedding = dataset[index]["speaker_embedding"]
        assert embedding[0].item() in {1.0, 2.0, 3.0}
        assert embedding[0].item() != expected_target_value
