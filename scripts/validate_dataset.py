#!/usr/bin/env python3
"""
M3-TTS: Standalone dataset validation script.

Validates TSV structure, audio files, and Khmer text.
Generates reports without modifying original data.

Usage:
    python validate_dataset.py \
        --tsv /path/to/metadata.tsv \
        --audio-dir /path/to/audio \
        --reports-dir /path/to/reports \
        --audio-col audio \
        --text-col text \
        --speaker-col speaker
"""

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import soundfile as sf

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def validate_tsv(df, audio_col, text_col, speaker_col):
    """Validate TSV structure and content."""
    report = []
    report.append("TSV VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().isoformat()}")
    report.append("=" * 60)
    report.append(f"Total rows: {len(df)}")
    report.append(f"Columns: {list(df.columns)}")
    report.append("")

    # Missing values
    report.append("MISSING VALUES:")
    for col in df.columns:
        n = df[col].isnull().sum()
        report.append(f"  {col}: {n}")
    report.append("")

    # Empty transcripts
    empty_text = 0
    if text_col in df.columns:
        empty_mask = df[text_col].isnull() | (df[text_col].astype(str).str.strip() == "")
        empty_text = empty_mask.sum()
        report.append(f"Empty transcripts: {empty_text}")
    report.append("")

    # Missing audio paths
    missing_audio = 0
    if audio_col in df.columns:
        missing_mask = df[audio_col].isnull() | (df[audio_col].astype(str).str.strip() == "")
        missing_audio = missing_mask.sum()
        report.append(f"Missing audio paths: {missing_audio}")
    report.append("")

    # Duplicates
    if audio_col in df.columns:
        dup_audio = df.duplicated(subset=[audio_col], keep=False).sum()
        report.append(f"Duplicate audio paths: {dup_audio} rows")
    if text_col in df.columns:
        dup_text = df.duplicated(subset=[text_col], keep=False).sum()
        report.append(f"Duplicate transcripts: {dup_text} rows")
        dup_ratio = dup_text / len(df) if len(df) > 0 else 0
        if dup_ratio > 0.3:
            report.append(f"  WARNING: {dup_ratio:.1%} duplicate text ratio")
    report.append("")

    return "\n".join(report), empty_text, missing_audio


def validate_audio(df, audio_col, audio_dir, min_dur=0.5, max_dur=30.0):
    """Validate audio files referenced in TSV."""
    results = []
    total_duration = 0.0
    sample_rates = {}
    channels_count = {}
    valid_count = 0
    invalid_count = 0

    if audio_col not in df.columns:
        return results, 0, 0, 0.0, {}, {}

    for idx, row in df.iterrows():
        audio_path = row[audio_col]
        if pd.isna(audio_path) or str(audio_path).strip() == "":
            results.append({
                "index": idx, "audio": audio_path,
                "status": "invalid", "reason": "Empty audio path",
                "duration": None, "sample_rate": None, "channels": None
            })
            invalid_count += 1
            continue

        full_path = os.path.join(audio_dir, str(audio_path).strip())
        if not os.path.exists(full_path):
            results.append({
                "index": idx, "audio": audio_path,
                "status": "invalid", "reason": "File not found",
                "duration": None, "sample_rate": None, "channels": None
            })
            invalid_count += 1
            continue

        try:
            info = sf.info(full_path)
            duration = info.duration
            sr = info.samplerate
            ch = info.channels

            total_duration += duration
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            channels_count[ch] = channels_count.get(ch, 0) + 1

            reasons = []
            if duration < min_dur:
                reasons.append(f"Too short ({duration:.2f}s)")
            if duration > max_dur:
                reasons.append(f"Too long ({duration:.2f}s)")

            if HAS_LIBROSA:
                try:
                    y, _ = librosa.load(full_path, sr=None, mono=True)
                    if np.max(np.abs(y)) < 0.001:
                        reasons.append("Silent audio")
                except Exception:
                    reasons.append("Cannot read audio data")

            if reasons:
                results.append({
                    "index": idx, "audio": audio_path,
                    "status": "invalid", "reason": "; ".join(reasons),
                    "duration": duration, "sample_rate": sr, "channels": ch
                })
                invalid_count += 1
            else:
                results.append({
                    "index": idx, "audio": audio_path,
                    "status": "valid", "reason": "",
                    "duration": duration, "sample_rate": sr, "channels": ch
                })
                valid_count += 1
        except Exception as e:
            results.append({
                "index": idx, "audio": audio_path,
                "status": "invalid", "reason": f"Cannot open: {e}",
                "duration": None, "sample_rate": None, "channels": None
            })
            invalid_count += 1

    return results, valid_count, invalid_count, total_duration, sample_rates, channels_count


def validate_khmer_text(df, text_col):
    """Validate Khmer text content."""
    report = []
    report.append("KHMER TEXT VALIDATION REPORT")
    report.append("=" * 60)

    if text_col not in df.columns:
        report.append(f"ERROR: Column '{text_col}' not found.")
        return "\n".join(report)

    texts = df[text_col].fillna("")

    empty_count = (texts.str.strip() == "").sum()
    report.append(f"Empty transcripts: {empty_count}")

    leading_trailing = texts.str.contains(r"^\s|\s$", regex=True, na=False).sum()
    report.append(f"With leading/trailing spaces: {leading_trailing}")

    excessive_spaces = texts.str.contains(r"\s{2,}", regex=True, na=False).sum()
    report.append(f"With excessive spaces: {excessive_spaces}")

    normalized = texts.apply(lambda x: unicodedata.normalize("NFC", str(x)))
    not_normalized = (texts != normalized).sum()
    report.append(f"Not NFC-normalized: {not_normalized}")

    khmer_re = re.compile(r"[\u1780-\u17FF]")
    has_khmer = texts.apply(lambda x: bool(khmer_re.search(str(x))))
    no_khmer = (~has_khmer & (texts.str.strip() != "")).sum()
    report.append(f"Rows without Khmer characters (non-empty): {no_khmer}")

    has_english = texts.str.contains(r"[a-zA-Z]", regex=True, na=False).sum()
    report.append(f"Rows with English characters: {has_english}")

    has_numbers = texts.str.contains(r"[0-9]", regex=True, na=False).sum()
    report.append(f"Rows with numbers: {has_numbers}")

    dup_text_count = texts.duplicated().sum()
    report.append(f"Duplicate transcripts: {dup_text_count}")

    all_chars = set()
    for t in texts:
        all_chars.update(str(t))
    report.append(f"Total unique characters: {len(all_chars)}")
    khmer_chars = [c for c in all_chars if khmer_re.search(c)]
    report.append(f"Khmer characters found: {len(khmer_chars)}")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Validate Khmer TTS dataset")
    parser.add_argument("--tsv", required=True, help="Path to metadata.tsv")
    parser.add_argument("--audio-dir", required=True, help="Audio directory")
    parser.add_argument("--reports-dir", required=True, help="Reports output directory")
    parser.add_argument("--audio-col", default="audio", help="Audio column name")
    parser.add_argument("--text-col", default="text", help="Text column name")
    parser.add_argument("--speaker-col", default="speaker", help="Speaker column name")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Min audio duration (s)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Max audio duration (s)")
    args = parser.parse_args()

    os.makedirs(args.reports_dir, exist_ok=True)

    # Load TSV
    print(f"Loading TSV: {args.tsv}")
    df = pd.read_csv(args.tsv, sep="\t", dtype=str)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # TSV validation
    print("\nValidating TSV structure...")
    tsv_report, empty_text, missing_audio = validate_tsv(
        df, args.audio_col, args.text_col, args.speaker_col
    )
    with open(os.path.join(args.reports_dir, "tsv_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(tsv_report)
    print(tsv_report)

    # Audio validation
    print("\nValidating audio files...")
    audio_results, valid, invalid, duration, sr_counts, ch_counts = validate_audio(
        df, args.audio_col, args.audio_dir, args.min_duration, args.max_duration
    )
    total_hours = duration / 3600

    audio_report_lines = [
        "AUDIO VALIDATION REPORT",
        "=" * 60,
        f"Total samples:  {len(df)}",
        f"Valid samples:  {valid}",
        f"Invalid samples: {invalid}",
        f"Total duration: {total_hours:.2f} hours",
        "",
        "Sample rates:",
    ]
    for sr, count in sorted(sr_counts.items()):
        audio_report_lines.append(f"  {sr} Hz: {count}")
    audio_report_lines.extend(["", "Channels:"])
    for ch, count in sorted(ch_counts.items()):
        label = "Mono" if ch == 1 else "Stereo" if ch == 2 else f"{ch}ch"
        audio_report_lines.append(f"  {label}: {count}")

    audio_report_text = "\n".join(audio_report_lines)
    with open(os.path.join(args.reports_dir, "audio_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(audio_report_text)
    print(f"\n{audio_report_text}")

    # Save invalid audio list
    invalid_audio = [r for r in audio_results if r["status"] == "invalid"]
    if invalid_audio:
        pd.DataFrame(invalid_audio).to_csv(
            os.path.join(args.reports_dir, "invalid_audio.tsv"), sep="\t", index=False
        )

    # Save invalid samples (TSV issues)
    invalid_all = []
    if audio_col_exists := args.audio_col in df.columns:
        if missing_audio > 0:
            invalid_all.append(df[df[args.audio_col].isnull() | (df[args.audio_col].str.strip() == "")])
    if text_col_exists := args.text_col in df.columns:
        if empty_text > 0:
            invalid_all.append(df[df[args.text_col].isnull() | (df[args.text_col].str.strip() == "")])
    if invalid_all:
        combined = pd.concat(invalid_all).drop_duplicates()
        combined.to_csv(os.path.join(args.reports_dir, "invalid_samples.tsv"), sep="\t", index=False)

    # Khmer text validation
    print("\nValidating Khmer text...")
    text_report = validate_khmer_text(df, args.text_col)
    with open(os.path.join(args.reports_dir, "khmer_text_validation_report.txt"), "w", encoding="utf-8") as f:
        f.write(text_report)
    print(text_report)

    # Dataset statistics
    stats = {
        "total_samples": len(df),
        "valid_audio_samples": valid,
        "invalid_audio_samples": invalid,
        "total_duration_seconds": float(duration),
        "total_duration_hours": float(total_hours),
        "sample_rates": {str(k): v for k, v in sr_counts.items()},
        "channels": {str(k): v for k, v in ch_counts.items()},
        "columns": list(df.columns),
        "generated_at": datetime.now().isoformat(),
    }
    with open(os.path.join(args.reports_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nAll reports saved to: {args.reports_dir}")
    print("Validation complete.")


if __name__ == "__main__":
    main()
