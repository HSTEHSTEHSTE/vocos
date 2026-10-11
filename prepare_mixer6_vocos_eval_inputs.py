#!/usr/bin/env python3
"""Prepare ten short Mixer6 excerpts and CPU WavLM content features for Vocos evaluation."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence


SAMPLE_RATE = 16_000
VOCOS_SAMPLE_RATE = 24_000
HOP = 320
RECEPTIVE_FIELD = 400
SPEAKER_RE = re.compile(r"_(?:LDC|HRM)_(\d+)_CH\d+\.flac$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixer6-root", type=Path, required=True)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--wavlm-source", type=Path, required=True)
    parser.add_argument("--wavlm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--clip-seconds", type=float, default=3.0)
    parser.add_argument("--search-seconds", type=float, default=180.0)
    parser.add_argument("--cpu-threads", type=int, default=12)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.clip_seconds <= 0 or args.search_seconds <= 0 or args.cpu_threads <= 0:
        parser.error("sample count, clip duration, search duration, and CPU threads must be positive")
    return args


def matrix_path(matrix_dir: Path, speaker_id: str) -> Path:
    speakers_dir = matrix_dir / "speakers"
    return (speakers_dir if speakers_dir.is_dir() else matrix_dir) / f"{speaker_id}.npy"


def selected_files(mixer6_root: Path, matrix_dir: Path, count: int) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    selected: list[tuple[str, Path]] = []
    for path in sorted((mixer6_root / "data" / "pcm_flac").rglob("*.flac")):
        match = SPEAKER_RE.search(path.name)
        if match is None:
            continue
        speaker_id = match.group(1)
        if speaker_id in seen or not matrix_path(matrix_dir, speaker_id).is_file():
            continue
        selected.append((speaker_id, path))
        seen.add(speaker_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Found only {len(selected)} distinct Mixer6 speakers with matrices; expected {count}")
    return selected


def choose_high_energy_clip(path: Path, duration_seconds: float, search_seconds: float) -> tuple[torch.Tensor, int]:
    with sf.SoundFile(path) as source:
        clip_frames = max(1, round(duration_seconds * source.samplerate))
        search_frames = min(source.frames, max(clip_frames, round(search_seconds * source.samplerate)))
        audio = source.read(frames=search_frames, dtype="float32", always_2d=True)
        if len(audio) < clip_frames:
            raise RuntimeError(f"Mixer6 source is shorter than one requested clip: {path}")
        mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
        step = max(1, clip_frames // 4)
        cumulative_energy = np.concatenate(([0.0], np.cumsum(np.square(mono), dtype=np.float64)))
        energy = cumulative_energy[clip_frames:] - cumulative_energy[:-clip_frames]
        start = int(np.argmax(energy[::step]) * step)
        start = min(start, len(mono) - clip_frames)
        clip = torch.from_numpy(mono[start : start + clip_frames]).unsqueeze(0)
        return clip, source.samplerate


def resample(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform.contiguous()
    return torchaudio.functional.resample(waveform, source_rate, target_rate).contiguous()


def feature_frame_count(num_samples: int) -> int:
    return 0 if num_samples < RECEPTIVE_FIELD else 1 + (num_samples - RECEPTIVE_FIELD) // HOP


def streaming_chunks(waveform: torch.Tensor, chunk_frames: int = 10, history_frames: int = 100):
    chunks = []
    chunk_samples = chunk_frames * HOP
    history_samples = history_frames * HOP
    emission_samples = chunk_samples - HOP
    for current_start in range(0, waveform.shape[1], emission_samples):
        core_end = min(current_start + chunk_samples, waveform.shape[1])
        sample_start = max(0, current_start - history_samples)
        chunk = waveform[:, sample_start:core_end]
        keep_start = (current_start - sample_start) // HOP
        valid_frames = feature_frame_count(chunk.shape[1])
        keep_frames = valid_frames - keep_start if core_end >= waveform.shape[1] else emission_samples // HOP
        if keep_frames < 0:
            raise RuntimeError("Invalid streaming WavLM chunk geometry")
        chunks.append((chunk, keep_start, keep_frames))
        if core_end >= waveform.shape[1]:
            break
    return chunks


def extract_streaming(model: torch.nn.Module, waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
    pieces = []
    for chunk, keep_start, keep_frames in streaming_chunks(waveform):
        with torch.inference_mode():
            features, _ = model.extract_features(chunk.to(device), output_layer=6)
        valid = feature_frame_count(chunk.shape[1])
        if keep_start + keep_frames > valid:
            raise RuntimeError("Streaming WavLM output was shorter than expected")
        pieces.append(features[0, keep_start : keep_start + keep_frames].cpu())
    return torch.cat(pieces, dim=0)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.mixer6_root / "data" / "pcm_flac", args.projection, args.wavlm_checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    if str(args.wavlm_source) not in sys.path:
        sys.path.insert(0, str(args.wavlm_source))
    from official_wavlm import load_official_wavlm_large  # pylint: disable=import-outside-toplevel

    device = torch.device("cpu")
    model, _ = load_official_wavlm_large(args.wavlm_checkpoint, device=device)
    model.eval()
    projection = torch.from_numpy(np.load(args.projection)).float()
    if tuple(projection.shape) != (1024, 75):
        raise ValueError(f"Expected a (1024, 75) projection, got {tuple(projection.shape)}")

    rows = []
    for index, (speaker_id, source_path) in enumerate(selected_files(args.mixer6_root, args.matrix_dir, args.num_samples)):
        waveform, source_rate = choose_high_energy_clip(source_path, args.clip_seconds, args.search_seconds)
        wav16 = resample(waveform, source_rate, SAMPLE_RATE)
        wav24 = resample(waveform, source_rate, VOCOS_SAMPLE_RATE)
        tag = f"{index:02d}_{speaker_id}_{source_path.stem}"
        audio_path = args.output_dir / "audio" / f"{tag}.wav"
        full_feature_path = args.output_dir / "features_oneframe75" / f"{tag}.pt"
        streaming_feature_path = args.output_dir / "features_streaming75" / f"{tag}.pt"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        full_feature_path.parent.mkdir(parents=True, exist_ok=True)
        streaming_feature_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(audio_path, wav24, VOCOS_SAMPLE_RATE)
        with torch.inference_mode():
            full_features, _ = model.extract_features(wav16.to(device), output_layer=6)
        torch.save((full_features[0].cpu().float() @ projection).contiguous(), full_feature_path)
        torch.save((extract_streaming(model, wav16, device).float() @ projection).contiguous(), streaming_feature_path)
        rows.append((tag, speaker_id, source_path, audio_path, full_feature_path, streaming_feature_path, matrix_path(args.matrix_dir, speaker_id)))
        print(f"Prepared {tag}", flush=True)
    manifest = args.output_dir / "samples.tsv"
    manifest.write_text("".join("\t".join(map(str, row)) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
