# M3-TTS: Khmer Text-to-Speech

A Khmer Text-to-Speech project using StyleTTS2 fine-tuning.

## Project Structure

```
M3-TTS/
├── configs/
│   └── project_config.py          # Editable configuration
├── notebooks/
│   └── phase1_dataset_setup.ipynb  # Colab notebook (Phase 1)
├── scripts/
│   ├── prepare_training_metadata.py  # TSV → StyleTTS2 format
│   └── validate_dataset.py           # Standalone validation
├── reports/                         # Generated after validation
├── processed_dataset/               # Cleaned data (originals untouched)
├── docs/
│   └── phase1_report.md
└── README.md
```

## Phase 1: Dataset Setup

### Goal
Validate your Khmer TSV dataset + audio + text and verify the StyleTTS2 training framework works on Google Colab.

**No training is performed in Phase 1.**

### Setup (Google Colab)

1. Upload `khmer_tts_data.zip` to Google Drive root (`MyDrive/`)
   - The zip should contain your dataset (TSV + audio files)
2. Open `notebooks/phase1_dataset_setup.ipynb` in Google Colab
3. Edit `AUDIO_COLUMN`, `TEXT_COLUMN`, `SPEAKER_COLUMN` at the top if your TSV uses different names
4. Run all cells — the notebook auto-unzips, detects structure, validates, and generates reports

### Expected Outputs

```
reports/
├── tsv_validation_report.txt
├── audio_validation_report.txt
├── invalid_samples.tsv
├── invalid_audio.tsv
├── khmer_text_validation_report.txt
└── dataset_statistics.json

processed_dataset/
├── metadata_validated.tsv
├── styletts2_format/
│   ├── train_list.txt
│   └── val_list.txt
└── test/
    └── processed_* (test files)
```

## Why StyleTTS2?

| Framework   | Khmer Support | Training Code | License     | Colab Ready |
|------------|---------------|---------------|-------------|-------------|
| Kokoro-82M | No            | No (closed)   | Apache 2.0  | N/A         |
| F5-TTS     | No            | Yes           | CC BY-NC-SA | Yes         |
| XTTS-v2    | No            | Limited       | Coqui       | Yes         |
| **StyleTTS2** | **Yes** (community fork) | **Yes** (official) | **MIT** | **Yes** |

StyleTTS2 is chosen because:
- Official fine-tuning code available
- Existing Khmer fork (`mrrtmob/styletts2-khmer`)
- MIT license (commercially usable)
- Works on Colab T4 GPU (15GB VRAM)
- Supports single/multi-speaker training
- Only needs 1-2+ hours of clean audio

## Dataset Requirements

- WAV files at any sample rate (will be resampled to 22050 Hz)
- Mono or stereo (will be converted to mono)
- TSV with at minimum: audio path column + text column
- UTF-8 encoded Khmer text

## Phase 2 (Future)

- Fine-tune Stage 1 (diagnostic model)
- Fine-tune Stage 2 (prosody predictor)
- SLM adversarial training
- Khmer G2P (if character-level training is insufficient)
- Inference pipeline
- Deployment

## License

Project code: MIT
TTS Framework: StyleTTS2 (MIT)
