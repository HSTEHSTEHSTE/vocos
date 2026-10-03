import torch

from vocos.feature_extractors import MelSpectrogramFeatures
from vocos.heads import CausalISTFTHead
from vocos.models import CausalVocosBackbone


def _stream_backbone(backbone, features, chunk_sizes, **kwargs):
    state = None
    outputs = []
    offset = 0
    for size in chunk_sizes:
        output, state = backbone.forward_stream(features[..., offset : offset + size], state, **kwargs)
        outputs.append(output)
        offset += size
    return torch.cat(outputs, dim=1)


def test_causal_backbone_streaming_matches_full_forward():
    torch.manual_seed(0)
    backbone = CausalVocosBackbone(input_channels=3, dim=8, intermediate_dim=16, num_layers=2).eval()
    features = torch.randn(2, 3, 11)

    full_output = backbone(features)
    stream_output = _stream_backbone(backbone, features, [2, 5, 4])

    torch.testing.assert_close(stream_output, full_output, atol=1e-6, rtol=1e-6)


def test_speaker_conditioned_causal_backbone_streaming_matches_full_forward():
    torch.manual_seed(0)
    backbone = CausalVocosBackbone(
        input_channels=3, dim=8, intermediate_dim=16, num_layers=2, speaker_embedding_dim=5
    ).eval()
    with torch.no_grad():
        for conditioner in backbone.speaker_conditioners:
            conditioner[-1].weight.normal_(mean=0, std=0.02)
    features = torch.randn(2, 3, 11)
    speaker_embedding = torch.randn(2, 5)

    full_output = backbone(features, speaker_embedding=speaker_embedding)
    stream_output = _stream_backbone(
        backbone, features, [2, 5, 4], speaker_embedding=speaker_embedding
    )

    torch.testing.assert_close(stream_output, full_output, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(full_output, backbone(features, speaker_embedding=-speaker_embedding))


def test_causal_backbone_prefix_does_not_depend_on_future_features():
    torch.manual_seed(0)
    backbone = CausalVocosBackbone(input_channels=3, dim=8, intermediate_dim=16, num_layers=2).eval()
    features = torch.randn(1, 3, 10)
    changed_features = features.clone()
    changed_features[..., 6:] += 20

    original = backbone(features)
    changed = backbone(changed_features)

    torch.testing.assert_close(original[:, :6], changed[:, :6], atol=1e-6, rtol=1e-6)


def test_causal_mel_features_do_not_depend_on_future_audio():
    extractor = MelSpectrogramFeatures(sample_rate=24_000, n_fft=16, hop_length=4, n_mels=3, padding="causal")
    audio = torch.randn(1, 20)
    changed_audio = audio.clone()
    changed_audio[:, 12:] += 20

    original = extractor(audio)
    changed = extractor(changed_audio)

    assert original.shape == (1, 3, 5)
    torch.testing.assert_close(original[..., :3], changed[..., :3], atol=1e-6, rtol=1e-6)


def test_causal_mel_vocos_prefix_does_not_depend_on_future_audio():
    torch.manual_seed(0)
    extractor = MelSpectrogramFeatures(sample_rate=24_000, n_fft=16, hop_length=4, n_mels=3, padding="causal")
    backbone = CausalVocosBackbone(input_channels=3, dim=8, intermediate_dim=16, num_layers=2).eval()
    head = CausalISTFTHead(dim=8, n_fft=16, hop_length=4).eval()
    audio = torch.randn(1, 20)
    changed_audio = audio.clone()
    changed_audio[:, 12:] += 20

    original = head(backbone(extractor(audio)))
    changed = head(backbone(extractor(changed_audio)))

    torch.testing.assert_close(original[..., :12], changed[..., :12], atol=1e-6, rtol=1e-6)


def test_causal_istft_head_streaming_matches_full_forward():
    torch.manual_seed(0)
    head = CausalISTFTHead(dim=4, n_fft=16, hop_length=4).eval()
    frames = torch.randn(2, 9, 4)

    full_audio = head(frames)
    state = None
    chunks = []
    for frame_chunk in (frames[:, :3], frames[:, 3:7], frames[:, 7:]):
        audio, state = head.forward_stream(frame_chunk, state)
        chunks.append(audio)
    stream_audio = torch.cat(chunks, dim=-1)

    assert full_audio.shape == (2, 36)
    torch.testing.assert_close(stream_audio, full_audio, atol=1e-6, rtol=1e-6)


def test_causal_istft_head_prefix_does_not_depend_on_future_frames():
    torch.manual_seed(0)
    head = CausalISTFTHead(dim=4, n_fft=16, hop_length=4).eval()
    frames = torch.randn(1, 6, 4)
    changed_frames = frames.clone()
    changed_frames[:, 3:] += 20

    original = head(frames)
    changed = head(changed_frames)

    torch.testing.assert_close(original[..., :12], changed[..., :12], atol=1e-6, rtol=1e-6)
