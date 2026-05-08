import glob
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_scripts.voice_metrics import calculate_voice_metrics

TARGET_SR = 16000
CLONED_DIR = "data/evaluations/outputs"
REPORT_JSON = "data/evaluations/voice_quality_results.json"


def evaluate_directory(cloned_dir: str = CLONED_DIR, report_json: str = REPORT_JSON) -> dict:
    wav_files = sorted(glob.glob(os.path.join(cloned_dir, "*.wav")))
    if not wav_files:
        print(f"No .wav files found in {cloned_dir}")
        return {}

    all_metrics = []
    for path in wav_files:
        audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        metrics = calculate_voice_metrics(audio, sr)
        if metrics is not None:
            all_metrics.append({"file": os.path.basename(path), **metrics})

    if not all_metrics:
        print("No metrics computed.")
        return {}

    keys = ["Timbre Richness", "Pitch Stability", "Articulation", "Speech Rhythm"]
    averages = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}

    report = {
        "num_files": len(all_metrics),
        "averages": averages,
        "details": all_metrics,
    }

    os.makedirs(os.path.dirname(report_json) or ".", exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nVoice Quality Results ({len(all_metrics)} files)")
    print("-" * 40)
    for k, v in averages.items():
        print(f"  {k:<20} {v:.1f} / 100")
    print(f"\nFull report saved to {report_json}")

    return report


if __name__ == "__main__":
    evaluate_directory()
