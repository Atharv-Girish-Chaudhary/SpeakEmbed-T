import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import soxr

from pesq import pesq
from pystoi.stoi import stoi


TARGET_SR = 16000


@dataclass
class Score:
    file_ref: str
    file_deg: str
    pesq_wb: Optional[float]
    stoi_val: Optional[float]


def read_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return data.astype(np.float32, copy=False), int(sr)


def resample_if_needed(wav: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return wav
    return soxr.resample(wav, sr, target_sr)


def align_lengths(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_scores(ref_path: str, deg_path: str) -> Score:
    ref, sr_r = read_audio_mono(ref_path)
    deg, sr_d = read_audio_mono(deg_path)

    ref = resample_if_needed(ref, sr_r, TARGET_SR)
    deg = resample_if_needed(deg, sr_d, TARGET_SR)

    ref, deg = align_lengths(ref, deg)

    pesq_score = None
    stoi_score = None

    if pesq is not None and len(ref) > 0 and len(deg) > 0:
        try:
            # Wideband PESQ at 16kHz
            pesq_score = float(pesq(TARGET_SR, ref, deg, 'wb'))
        except Exception:
            pesq_score = None

    if stoi is not None and len(ref) > 0 and len(deg) > 0:
        try:
            stoi_score = float(stoi(ref, deg, TARGET_SR, extended=False))
        except Exception:
            stoi_score = None

    return Score(ref_path, deg_path, pesq_score, stoi_score)


def find_pairs(
    ref_root: str,
    deg_root: str,
    suffix_ref: str = ".wav",
    suffix_deg: str = ".wav",
    match_by_stem: bool = True,
) -> List[Tuple[str, str]]:
    ref_map = {}
    for root, _, files in os.walk(ref_root):
        for f in files:
            if f.lower().endswith(suffix_ref):
                stem = os.path.splitext(f)[0]
                ref_map[stem if match_by_stem else f] = os.path.join(root, f)

    pairs: List[Tuple[str, str]] = []
    for root, _, files in os.walk(deg_root):
        for f in files:
            if f.lower().endswith(suffix_deg):
                key = os.path.splitext(f)[0] if match_by_stem else f
                if key in ref_map:
                    pairs.append((ref_map[key], os.path.join(root, f)))
    return sorted(pairs)


def run_eval(
    ref_root: str,
    deg_root: str,
    output_json: Optional[str] = None,
    limit: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float]]:
    pairs = find_pairs(ref_root, deg_root)
    if limit is not None:
        pairs = pairs[:limit]

    scores: List[Score] = []
    for ref_p, deg_p in pairs:
        scores.append(compute_scores(ref_p, deg_p))

    pesq_vals = [s.pesq_wb for s in scores if s.pesq_wb is not None]
    stoi_vals = [s.stoi_val for s in scores if s.stoi_val is not None]

    avg_pesq = float(np.mean(pesq_vals)) if len(pesq_vals) > 0 else None
    avg_stoi = float(np.mean(stoi_vals)) if len(stoi_vals) > 0 else None

    if output_json is not None:
        os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(
                {
                    "target_sr": TARGET_SR,
                    "num_pairs": len(pairs),
                    "avg_pesq_wb": avg_pesq,
                    "avg_stoi": avg_stoi,
                    "details": [
                        {
                            "ref": s.file_ref,
                            "deg": s.file_deg,
                            "pesq_wb": s.pesq_wb,
                            "stoi": s.stoi_val,
                        }
                        for s in scores
                    ],
                },
                f,
                indent=2,
            )

    return avg_pesq, avg_stoi

__all__ = [
    "TARGET_SR",
    "Score",
    "compute_scores",
    "run_eval",
    "find_pairs",
]


