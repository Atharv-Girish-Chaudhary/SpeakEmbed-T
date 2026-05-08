import glob
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.embed import Embed
from scripts.speech_encoder_v2_updated import SpeechEncoderV2

TARGET_SR = 16000
REFERENCE_DIR = "datasets/LibriSpeech/train-clean-100/103/1240"
CLONED_DIR = "data/evaluations/outputs"
REPORT_JSON = "data/evaluations/speaker_similarity_results.json"


def load_encoder() -> Embed:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = SpeechEncoderV2(device, device)
    checkpoint_path = "models/speech_encoder_transformer_updated/encoder_073500_loss_0.0724.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Checkpoints may be saved as state_dict directly or nested under a key
    state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return Embed(encoder)


def get_embedding(embedder: Embed, path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    wav = wav.astype(np.float32)
    embedding, _, _ = embedder.embed_utterance(wav, return_partials=True)
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def evaluate(
    reference_dir: str = REFERENCE_DIR,
    cloned_dir: str = CLONED_DIR,
    report_json: str = REPORT_JSON,
) -> dict:
    embedder = load_encoder()

    # Match cloned files back to their reference by stem
    # cloned files are named e.g. 103-1240-0000_cloned.wav
    cloned_files = sorted(glob.glob(os.path.join(cloned_dir, "*_cloned.wav")))
    if not cloned_files:
        print(f"No cloned .wav files found in {cloned_dir}")
        return {}

    similarities = []
    details = []

    for cloned_path in cloned_files:
        stem = os.path.basename(cloned_path).replace("_cloned.wav", "")
        ref_path = os.path.join(reference_dir, f"{stem}.flac")
        if not os.path.exists(ref_path):
            print(f"  Reference not found for {stem}, skipping.")
            continue

        ref_emb = get_embedding(embedder, ref_path)
        clone_emb = get_embedding(embedder, cloned_path)
        sim = cosine_similarity(ref_emb, clone_emb)
        similarities.append(sim)
        details.append({"file": stem, "cosine_similarity": sim})

    if not similarities:
        print("No pairs evaluated.")
        return {}

    avg_sim = float(np.mean(similarities))
    report = {
        "num_pairs": len(similarities),
        "avg_cosine_similarity": avg_sim,
        "details": details,
    }

    os.makedirs(os.path.dirname(report_json) or ".", exist_ok=True)
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSpeaker Similarity Results ({len(similarities)} pairs)")
    print("-" * 40)
    print(f"  Avg cosine similarity:   {avg_sim:.4f}")
    print(f"  Min:                     {min(similarities):.4f}")
    print(f"  Max:                     {max(similarities):.4f}")
    print(f"\nFull report saved to {report_json}")

    return report


if __name__ == "__main__":
    evaluate()
