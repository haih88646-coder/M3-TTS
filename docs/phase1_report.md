# Phase 1 Report: Khmer TTS Dataset Setup

## 1. TTS Framework Research

### Kokoro-82M
- **Repository**: https://github.com/hexgrad/kokoro
- **License**: Apache 2.0 (weights), MIT (inference code)
- **Architecture**: StyleTTS 2 + ISTFTNet, decoder-only
- **Parameters**: 82M
- **Supported Languages**: English, French, Korean, Japanese, Chinese
- **Training Code**: Not available officially
- **Khmer Support**: No
- **Verdict**: Not practical. No official way to add new languages. Community attempts (kikiri-tts) are incomplete.

### F5-TTS
- **Repository**: https://github.com/SWivid/F5-TTS
- **License**: CC BY-NC-SA 4.0 (non-commercial)
- **Architecture**: Flow matching with DiT transformer
- **Training Code**: Available, requires custom vocab creation
- **Khmer Support**: No official support. Community fine-tunes exist for Japanese, Korean, etc.
- **Verdict**: Possible but non-commercial license. Requires building custom Khmer tokenizer.

### Coqui TTS / XTTS-v2
- **Repository**: https://github.com/coqui-ai/TTS
- **License**: coqui-public-model-license
- **Supported Languages**: 17 languages (no Khmer)
- **Training Code**: Available
- **Verdict**: No Khmer support. Limited fine-tuning options for new languages.

### StyleTTS2 (SELECTED)
- **Repository**: https://github.com/yl4579/StyleTTS2
- **License**: MIT
- **Architecture**: Style diffusion + adversarial training + SLM
- **Training Code**: Official fine-tuning code + Colab notebook
- **Khmer Support**: Community fork exists: https://github.com/mrrtmob/styletts2-khmer
- **Colab Demo**: https://colab.research.google.com/github/yl4579/StyleTTS2/blob/main/Colab/StyleTTS2_Finetune_Demo.ipynb
- **GPU Requirements**: NVIDIA T4 (15GB) minimum; better with A100/V100
- **Data Requirements**: 1+ hours minimum, 10+ hours recommended
- **Verdict**: Best choice for Khmer TTS. Has existing Khmer fork, official training code, MIT license.

## 2. Selected Framework: StyleTTS2

### Why StyleTTS2?

1. **Existing Khmer fork** - `mrrtmob/styletts2-khmer` by a Cambodian developer (Blizzer, Phnom Penh)
2. **Official training code** - Fine-tuning scripts for both Stage 1 and Stage 2
3. **MIT License** - Commercially usable
4. **Colab compatible** - Works on free T4 GPU
5. **Low data requirement** - 22k samples (~20-25 hours) provides excellent quality
6. **Proven results** - Outperforms larger models (XTTS-v2, MetaVoice) in TTS benchmarks

### Architecture

```
Text Input
    ↓
Text Encoder (PL-BERT based)
    ↓
Style Diffusion Module
    ↓
Duration Predictor
    ↓
mel-spectrogram Generation (DiT)
    ↓
iSTFTNet Vocoder
    ↓
Waveform Output (22050 Hz)
```

### Training Stages

- **Stage 1**: Diagnostic model - learns text-to-mel mapping
- **Stage 2**: Prosody predictor - adds naturalness and expressiveness
- **Stage 3** (optional): SLM adversarial training for quality improvement

### Pre-trained Components

- ASR text aligner (pre-trained on English, Japanese, Chinese)
- JDC pitch extractor (language-independent)
- PL-BERT text encoder (pre-trained on English Wikipedia)

## 3. Khmer G2P Analysis

### Current State

No reliable, pip-installable Khmer G2P system exists.

### Options Evaluated

1. **eSpeak-NG** - Has `kh` language code but produces limited/empty output for Khmer
2. **Custom rule-based G2P** - Requires linguistics expertise, error-prone
3. **MMS Phonemizer** - Meta's multilingual model, Khmer support uncertain
4. **Character-level training** - StyleTTS2 can learn Khmer characters directly

### Recommendation

**Character-level training** for Phase 2. The text encoder in StyleTTS2 can learn Khmer Unicode character representations directly from the paired audio-text data. This avoids the need for a separate G2P system.

If quality is insufficient, build a custom Khmer G2P in a future phase.

## 4. Dataset Compatibility

### StyleTTS2 Input Format

```
audio_path|text
```

Example:
```
000001.wav|សួស្តីអ្នកទាំងអស់គ្នា។
000002.wav|ថ្ងៃនេះយើងនឹងនិយាយអំពីទំនាក់ទំនង។
```

### Conversion

The `scripts/prepare_training_metadata.py` script automatically converts:
- TSV with configurable column names → StyleTTS2 `train_list.txt` / `val_list.txt`
- Splits 90/10 train/validation

## 5. Phase 2 Roadmap

1. Fine-tune StyleTTS2 Stage 1 on Khmer data
2. Evaluate Stage 1 output quality
3. Fine-tune Stage 2 (prosody predictor)
4. Optional: SLM adversarial training
5. MOS (Mean Opinion Score) evaluation
6. Build Khmer G2P if character-level is insufficient
7. Create inference pipeline
8. Package model for deployment

## 6. Success Criteria (Phase 1)

- [x] Google Colab GPU works
- [x] Google Drive mountable
- [x] TSV loads correctly
- [x] Khmer text displays correctly
- [x] Audio files can be read
- [x] Dataset statistics generated
- [x] Invalid samples identified
- [x] Original TSV not modified
- [x] Original audio not modified
- [x] Sample audio preprocessing works
- [x] StyleTTS2 framework verified
- [x] Training format conversion automated
- [x] No full training performed
