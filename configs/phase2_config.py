"""
M3-TTS Phase 2 — Training Configuration

StyleTTS2 Fine-tuning for Khmer TTS
"""

# =============================================================================
# MODEL
# =============================================================================

MODEL_NAME = "StyleTTS2"
MODEL_VERSION = "v1.0-khmer"
FRAMEWORK = "StyleTTS2"
LICENSE = "MIT"

# Pre-trained checkpoint (LJSpeech)
PRETRAINED_CHECKPOINT = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LJSpeech.pth"
PRETRAINED_CONFIG = "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LJSpeech_config.yml"

# =============================================================================
# PATHS (Google Drive)
# =============================================================================

GDRIVE_MOUNT = "/content/drive"
PROJECT_ROOT = f"{GDRIVE_MOUNT}/MyDrive/khmer_tts"

# Phase 1 outputs
PHASE1_DATA = "/content/khmer_tts_data"
PHASE1_METADATA = "/content/khmer_tts_data/data/processed_rom/train/metadata.csv"
PHASE1_AUDIO_DIR = "/content/khmer_tts_data"

# Phase 2 working directories
PROCESSED_DIR = f"{PROJECT_ROOT}/processed_dataset"
RESAMPLED_DIR = f"{PROJECT_ROOT}/resampled_audio"
STYLETTS2_DATA_DIR = f"{PROJECT_ROOT}/styletts2_data"
CHECKPOINTS_DIR = f"{PROJECT_ROOT}/checkpoints"
SAMPLES_DIR = f"{PROJECT_ROOT}/samples"
REPORTS_DIR = f"{PROJECT_ROOT}/reports"
EXPORT_DIR = f"{PROJECT_ROOT}/export"
LOG_DIR = f"{PROJECT_ROOT}/logs"

# StyleTTS2 repo (cloned in Colab)
STYLETTS2_REPO = "https://github.com/yl4579/StyleTTS2.git"
STYLETTS2_DIR = "/content/StyleTTS2"

# =============================================================================
# AUDIO SETTINGS
# =============================================================================

SAMPLE_RATE = 24000  # StyleTTS2 requires 24kHz
CHANNELS = 1
HOP_LENGTH = 300     # 24000 / 300 = 80 fps
N_FFT = 2048
WIN_LENGTH = 1200

# =============================================================================
# TEXT SETTINGS
# =============================================================================

# Column names in the metadata CSV (pipe-separated, no header)
AUDIO_COLUMN = "audio"
TEXT_COLUMN = "text"
SPEAKER_COLUMN = "language"  # All "en" but we use speaker_id=0

# Text processing
# Our text is romanized Khmer (Latin characters).
# StyleTTS2 TextCleaner vocab includes A-Z, a-z, IPA chars.
# Romanized Khmer uses only Latin chars -> NO G2P needed.
TEXT_ENCODING = "romanized"  # Direct Latin character encoding

# =============================================================================
# TRAINING SETTINGS
# =============================================================================

# Fine-tuning from LJSpeech
TRAINING_MODE = "finetune"  # "finetune" or "scratch"

# Training hyperparameters (for T4 14.6GB VRAM)
BATCH_SIZE = 4              # Reduced for T4 (original: 8)
GRADIENT_ACCUMULATION = 2   # Effective batch size = 8
MAX_LEN = 300               # Max audio frames (~3.75 sec at 80fps)
EPOCHS = 50
LEARNING_RATE = 1e-4
BERT_LR = 1e-5
FT_LR = 1e-4
WEIGHT_DECAY = 0.01

# Training phases (epoch thresholds)
DIFF_EPOCH = 10             # When diffusion starts
JOINT_EPOCH = 30            # When joint training starts

# Checkpointing
SAVE_INTERVAL = 5           # Save every N epochs
EVAL_INTERVAL = 5           # Evaluate every N epochs
LOG_INTERVAL = 10           # Log every N steps

# Resume training
RESUME_FROM_CHECKPOINT = None  # Set to checkpoint path to resume

# Pilot training
PILOT_STEPS = 200           # Steps for pilot training
PILOT_BATCH_SIZE = 2        # Smaller batch for pilot
PILOT_SUBSET_SIZE = 100     # Number of samples for pilot

# =============================================================================
# SPLIT SETTINGS
# =============================================================================

RANDOM_SEED = 42
TRAIN_RATIO = 0.9
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# =============================================================================
# EVALUATION SENTENCES (Romanized Khmer)
# =============================================================================

EVAL_SENTENCES = {
    "normal": [
        "sou sdei nak teang os knongea.",
        "kyun thngai nih yerng nung niek arp mean teang nak teang os knongea.",
    ],
    "question": [
        "het avey ban chea kak min chhlouy teb sar robos knhom?",
        "avey ban chea meak min niek robos nak teang?",
    ],
    "long": [
        "peal dael yerng char teuk pteah yuol mean arp mean ar romnerh robos meak mnous mihn teang yerng nung char teuk pteah yuol tha yerng nung jong yuol tha romnerh mean teang nak teang os knongea.",
    ],
    "relationship": [
        "romnerh robos nak teang os knongea mean teang meak pteah tiav knong chivit.",
        "nak teang os knongea dochchea ban niek robos nak teang min chhlouy teb sar.",
        "peal meak mnous min niek knear robos nak teang min te chem chem te tae srolanh mean teang robos.",
    ],
    "numbers": [
        "meak nih mean prambuon mek muleah.",
        "pho mok robos knhom mean pi prambuon mek.",
    ],
    "english_khmer": [
        "YouTube ke chea veticah dav pininh muleah somrobos nak ponheaktoh matega.",
    ],
}

# =============================================================================
# EXPORT SETTINGS
# =============================================================================

EXPORT_MODEL_NAME = "khmer_tts_v1"
EXPORT_VERSION = "1.0.0"

# =============================================================================
# GPU CONFIGURATION
# =============================================================================

# Auto-detect and configure based on VRAM
GPU_CONFIGS = {
    "low": {"vram_gb": 8, "batch_size": 2, "max_len": 200, "grad_accum": 4},
    "standard": {"vram_gb": 14, "batch_size": 4, "max_len": 300, "grad_accum": 2},
    "high": {"vram_gb": 24, "batch_size": 8, "max_len": 400, "grad_accum": 1},
    "very_high": {"vram_gb": 40, "batch_size": 16, "max_len": 500, "grad_accum": 1},
}


def detect_gpu_config(vram_gb):
    """Select training config based on available VRAM."""
    if vram_gb >= 40:
        cfg = GPU_CONFIGS["very_high"]
    elif vram_gb >= 24:
        cfg = GPU_CONFIGS["high"]
    elif vram_gb >= 14:
        cfg = GPU_CONFIGS["standard"]
    else:
        cfg = GPU_CONFIGS["low"]
    return cfg


def print_config():
    """Print current Phase 2 configuration."""
    print("=" * 60)
    print("M3-TTS Phase 2 Configuration")
    print("=" * 60)
    print(f"  Model:           {MODEL_NAME} {MODEL_VERSION}")
    print(f"  Framework:       {FRAMEWORK}")
    print(f"  Training mode:   {TRAINING_MODE}")
    print(f"  Sample rate:     {SAMPLE_RATE} Hz")
    print(f"  Text encoding:   {TEXT_ENCODING}")
    print(f"  Batch size:      {BATCH_SIZE}")
    print(f"  Gradient accum:  {GRADIENT_ACCUMULATION}")
    print(f"  Effective batch: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Max len:         {MAX_LEN} frames")
    print(f"  Epochs:          {EPOCHS}")
    print(f"  Learning rate:   {LEARNING_RATE}")
    print(f"  Diff epoch:      {DIFF_EPOCH}")
    print(f"  Joint epoch:     {JOINT_EPOCH}")
    print(f"  Checkpoint dir:  {CHECKPOINTS_DIR}")
    print(f"  Samples dir:     {SAMPLES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
