import glob
import json
import os
from typing import List, Optional, Tuple

import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import numpy as np
import soundfile as sf

from scripts.main import Main
from evaluations.voice_quality_eval import TARGET_SR, compute_scores


def load_reference_audio(flac_path: str, target_sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    wav, sr = librosa.load(flac_path, sr=target_sr, mono=True)
    return wav.astype(np.float32, copy=False), target_sr


def clone_with_main(model: Main, ref_wav: np.ndarray, text: Optional[str] = None, use_vocoder: bool = True) -> np.ndarray:
    cloned = model.clone_audio(ref_wav, use_vocoder=use_vocoder, text=text)
    return cloned.astype(np.float32, copy=False)


def evaluate_files(
    model: Main,
    flac_files: List[str],
    out_dir: Optional[str] = None,
    use_vocoder: bool = True,
) -> Tuple[Optional[float], Optional[float], List[dict]]:
    os.makedirs(out_dir or '.', exist_ok=True)

    pesq_vals: List[float] = []
    stoi_vals: List[float] = []
    details: List[dict] = []

    for i, flac in enumerate(flac_files, start=1):
        ref_wav, sr = load_reference_audio(flac)
        cloned = clone_with_main(model, ref_wav, text=None, use_vocoder=use_vocoder)

        # Save cloned audio for inspection if requested
        cloned_path = None
        if out_dir is not None:
            base = os.path.splitext(os.path.basename(flac))[0]
            cloned_path = os.path.join(out_dir, f"{base}_cloned.wav")
            sf.write(cloned_path, cloned, sr)

        # Compute objective metrics against reference (via temporary files)
        ref_tmp = os.path.join(out_dir or '.', f"__tmp_ref_{i}.wav")
        deg_tmp = os.path.join(out_dir or '.', f"__tmp_deg_{i}.wav")
        sf.write(ref_tmp, ref_wav, sr)
        sf.write(deg_tmp, cloned, sr)

        score = compute_scores(ref_tmp, deg_tmp)

        # Cleanup temps
        try:
            os.remove(ref_tmp)
            os.remove(deg_tmp)
        except Exception:
            pass

        if score.pesq_wb is not None:
            pesq_vals.append(score.pesq_wb)
        if score.stoi_val is not None:
            stoi_vals.append(score.stoi_val)

        details.append(
            {
                "flac": flac,
                "cloned_wav": cloned_path,
                "pesq_wb": score.pesq_wb,
                "stoi": score.stoi_val,
            }
        )

    avg_pesq = float(np.mean(pesq_vals)) if len(pesq_vals) > 0 else None
    avg_stoi = float(np.mean(stoi_vals)) if len(stoi_vals) > 0 else None
    return avg_pesq, avg_stoi, details


def discover_flacs(root: str) -> List[str]:
    # LibriSpeech structure has nested folders; glob recursively
    return sorted(glob.glob(os.path.join(root, '**', '*.flac'), recursive=True))


def evaluate_clone_dataset(
    num_files: int = 20,
    dataset_root: str = "datasets/LibriSpeech/train-clean-100",
    use_vocoder: bool = True,
    out_dir: Optional[str] = "data/evaluations/outputs",
    report_json: Optional[str] = "data/evaluations/dataset_eval_results.json",
) -> Tuple[Optional[float], Optional[float], dict]:
    """
    Clone num_files FLACs using Main.clone_audio and compute average PESQ/STOI.

    Returns (avg_pesq, avg_stoi, report_dict).
    """
    flacs = discover_flacs(dataset_root)
    if len(flacs) == 0:
        return None, None, {
            "dataset_root": dataset_root,
            "num_files": 0,
            "avg_pesq_wb": None,
            "avg_stoi": None,
            "details": [],
        }

    flacs = flacs[: max(0, num_files)]

    model = Main(original_encoder=False)
    avg_pesq, avg_stoi, details = evaluate_files(
        model=model,
        flac_files=flacs,
        out_dir=out_dir,
        use_vocoder=use_vocoder,
    )

    report = {
        "dataset_root": dataset_root,
        "num_files": len(flacs),
        "avg_pesq_wb": avg_pesq,
        "avg_stoi": avg_stoi,
        "details": details,
    }

    if report_json is not None:
        os.makedirs(os.path.dirname(report_json) or '.', exist_ok=True)
        with open(report_json, 'w') as f:
            json.dump(report, f, indent=2)

    return avg_pesq, avg_stoi, report


if __name__ == "__main__":
    _, _, report = evaluate_clone_dataset()
    print(json.dumps(report, indent=2))


