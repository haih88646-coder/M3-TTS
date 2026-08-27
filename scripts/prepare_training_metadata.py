#!/usr/bin/env python3
"""
M3-TTS: Convert TSV dataset to StyleTTS2 training format.

StyleTTS2 expects:
  - train_list.txt and val_list.txt
  - Format per line: audio_path|text
  - Audio files referenced by relative path

Usage:
    python prepare_training_metadata.py \
        --tsv /path/to/metadata.tsv \
        --audio-dir /path/to/audio \
        --output-dir /path/to/output \
        --audio-col audio \
        --text-col text \
        --train-split 0.9
"""

import argparse
import os
import random
import unicodedata
import re


def clean_text(text):
    """Normalize and clean Khmer text."""
    text = unicodedata.normalize("NFC", str(text).strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_tsv(tsv_path, audio_col, text_col, sep="\t"):
    """Load TSV and return list of (audio, text) tuples."""
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(sep)
        if audio_col not in header or text_col not in header:
            raise ValueError(
                f"Columns not found. Available: {header}. "
                f"Expected: {audio_col}, {text_col}"
            )
        audio_idx = header.index(audio_col)
        text_idx = header.index(text_col)

        for line_no, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            if len(parts) < max(audio_idx, text_idx) + 1:
                continue
            audio_path = parts[audio_idx].strip()
            text = parts[text_idx].strip()
            if audio_path and text:
                rows.append((audio_path, clean_text(text)))

    return rows


def write_list(filepath, lines):
    """Write lines to a text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Convert TSV to StyleTTS2 format")
    parser.add_argument("--tsv", required=True, help="Path to metadata.tsv")
    parser.add_argument("--audio-dir", default=None, help="Audio directory (for existence check)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--audio-col", default="audio", help="Audio column name")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading TSV: {args.tsv}")
    rows = load_tsv(args.tsv, args.audio_col, args.text_col)
    print(f"  Loaded {len(rows)} samples")

    if args.audio_dir:
        valid_rows = []
        for audio, text in rows:
            full_path = os.path.join(args.audio_dir, audio)
            if os.path.exists(full_path):
                valid_rows.append((audio, text))
            else:
                print(f"  [SKIP] Missing: {audio}")
        print(f"  Valid samples with existing audio: {len(valid_rows)}")
        rows = valid_rows

    # Shuffle and split
    random.seed(args.seed)
    shuffled = rows[:]
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * args.train_split)
    train_rows = shuffled[:split_idx]
    val_rows = shuffled[split_idx:]

    # Format: audio_path|text
    train_lines = [f"{audio}|{text}" for audio, text in train_rows]
    val_lines = [f"{audio}|{text}" for audio, text in val_rows]

    # Write
    train_path = os.path.join(args.output_dir, "train_list.txt")
    val_path = os.path.join(args.output_dir, "val_list.txt")
    write_list(train_path, train_lines)
    write_list(val_path, val_lines)

    print(f"\nOutput:")
    print(f"  train_list.txt: {len(train_lines)} samples -> {train_path}")
    print(f"  val_list.txt:   {len(val_lines)} samples -> {val_path}")
    print(f"\nFormat: audio_path|text")
    print(f"Example: {train_lines[0] if train_lines else 'N/A'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
