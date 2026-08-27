"""
M3-TTS Phase 2 — Data Preparation

Converts Phase 1 metadata to StyleTTS2 training format:
- Resamples audio from 16kHz to 24kHz
- Creates train_list.txt and val_list.txt
- Splits data with fixed random seed
"""

import os
import csv
import json
import random
import shutil
import subprocess
import numpy as np


def load_metadata(metadata_path):
    """Load pipe-separated metadata CSV (no header)."""
    rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[1].strip()
                language = parts[2].strip() if len(parts) > 2 else "km"
                rows.append({"audio": filename, "text": text, "language": language})
    return rows


def split_data(rows, train_ratio=0.9, val_ratio=0.05, seed=42):
    """Split data into train/val/test with fixed seed."""
    random.seed(seed)
    indices = list(range(len(rows)))
    random.shuffle(indices)

    n_train = int(len(rows) * train_ratio)
    n_val = int(len(rows) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    return train_rows, val_rows, test_rows


def resample_audio(input_path, output_path, target_sr=24000):
    """Resample audio using ffmpeg or sox."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(target_sr),
            "-ac", "1",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        # Try sox
        try:
            cmd = ["sox", input_path, "-r", str(target_sr), "-c", "1", output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False


def resample_dataset(rows, audio_source_dir, output_dir, target_sr=24000):
    """Resample all audio files and return updated rows."""
    os.makedirs(output_dir, exist_ok=True)

    SEP = chr(92)  # backslash
    file_lookup = {}
    for root, dirs, files in os.walk(audio_source_dir):
        for f in files:
            if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                actual_name = f.split(SEP)[-1].lower()
                if actual_name not in file_lookup:
                    file_lookup[actual_name] = os.path.join(root, f)

    updated_rows = []
    success_count = 0
    fail_count = 0

    for row in rows:
        audio_name = row["audio"].lower()
        source_path = file_lookup.get(audio_name)

        if not source_path or not os.path.exists(source_path):
            fail_count += 1
            continue

        out_name = os.path.splitext(row["audio"])[0] + ".wav"
        out_path = os.path.join(output_dir, out_name)

        if os.path.exists(out_path):
            success_count += 1
            updated_rows.append({
                "audio": out_name,
                "text": row["text"],
                "speaker": "0",
            })
            continue

        if resample_audio(source_path, out_path, target_sr):
            success_count += 1
            updated_rows.append({
                "audio": out_name,
                "text": row["text"],
                "speaker": "0",
            })
        else:
            fail_count += 1

    print(f"Resampled: {success_count} success, {fail_count} failed")
    return updated_rows


def write_styletts2_list(rows, output_path):
    """Write StyleTTS2 format: filename|text|speaker_id"""
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            line = f"{row['audio']}|{row['text']}|{row['speaker']}"
            f.write(line + "\n")


def prepare_dataset(
    metadata_path,
    audio_source_dir,
    output_dir,
    train_ratio=0.9,
    val_ratio=0.05,
    seed=42,
    target_sr=24000,
):
    """Full data preparation pipeline."""
    print("=" * 60)
    print("M3-TTS Data Preparation")
    print("=" * 60)

    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    rows = load_metadata(metadata_path)
    print(f"  Loaded {len(rows)} samples")

    # Split data
    print(f"\nSplitting data (train={train_ratio}, val={val_ratio}, seed={seed})...")
    train_rows, val_rows, test_rows = split_data(rows, train_ratio, val_ratio, seed)
    print(f"  Train: {len(train_rows)}")
    print(f"  Val:   {len(val_rows)}")
    print(f"  Test:  {len(test_rows)}")

    # Resample audio
    resampled_dir = os.path.join(output_dir, "resampled")
    print(f"\nResampling audio to {target_sr}Hz...")
    train_rows = resample_dataset(train_rows, audio_source_dir, resampled_dir, target_sr)
    val_rows = resample_dataset(val_rows, audio_source_dir, resampled_dir, target_sr)
    test_rows = resample_dataset(test_rows, audio_source_dir, resampled_dir, target_sr)

    # Write StyleTTS2 lists
    data_dir = os.path.join(output_dir, "styletts2")
    os.makedirs(data_dir, exist_ok=True)

    train_list_path = os.path.join(data_dir, "train_list.txt")
    val_list_path = os.path.join(data_dir, "val_list.txt")
    test_list_path = os.path.join(data_dir, "test_list.txt")

    print(f"\nWriting StyleTTS2 format...")
    write_styletts2_list(train_rows, train_list_path)
    write_styletts2_list(val_rows, val_list_path)
    write_styletts2_list(test_rows, test_list_path)

    print(f"  Train: {train_list_path} ({len(train_rows)} samples)")
    print(f"  Val:   {val_list_path} ({len(val_rows)} samples)")
    print(f"  Test:  {test_list_path} ({len(test_rows)} samples)")

    # Save split info
    split_info = {
        "total": len(rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "seed": seed,
        "target_sr": target_sr,
        "sample_rate": target_sr,
    }
    split_path = os.path.join(output_dir, "split_info.json")
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSplit info saved to {split_path}")

    # Show example line
    if train_rows:
        print(f"\nExample train line:")
        print(f"  {train_rows[0]['audio']}|{train_rows[0]['text'][:50]}...|{train_rows[0]['speaker']}")

    print("\nData preparation complete.")
    return train_rows, val_rows, test_rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python prepare_training_data.py <metadata_path> <audio_source_dir> <output_dir>")
        sys.exit(1)

    metadata_path = sys.argv[1]
    audio_source_dir = sys.argv[2]
    output_dir = sys.argv[3]

    prepare_dataset(metadata_path, audio_source_dir, output_dir)
