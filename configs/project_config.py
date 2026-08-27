"""
M3-TTS Project Configuration
Phase 1 — Khmer TTS Dataset Setup

Edit these values to match your dataset and Google Drive structure.
"""

# =============================================================================
# TSV COLUMN MAPPING
# =============================================================================
# Adjust these to match your TSV header column names.
# The notebook will inspect your TSV and display column names automatically.

AUDIO_COLUMN = "audio"
TEXT_COLUMN = "text"
SPEAKER_COLUMN = "speaker"

# =============================================================================
# PATHS (Google Drive)
# =============================================================================

# Google Drive mount point (default for Colab)
GDRIVE_MOUNT = "/content/drive"

# ZIP file location on Google Drive
ZIP_FILENAME = "khmer_tts_data.zip"
ZIP_PATH = f"{GDRIVE_MOUNT}/MyDrive/{ZIP_FILENAME}"

# Working directories
EXTRACT_DIR = "/content/khmer_tts_data"  # Where zip is extracted (Colab local)
PROJECT_ROOT = f"{GDRIVE_MOUNT}/MyDrive/khmer_tts"

# Dataset location (auto-detected after unzip)
DATASET_DIR = None  # Set automatically
AUDIO_DIR = None     # Set automatically
METADATA_TSV = None  # Set automatically

# Output directories
PROCESSED_DIR = f"{PROJECT_ROOT}/processed_dataset"
REPORTS_DIR = f"{PROJECT_ROOT}/reports"
CHECKPOINTS_DIR = f"{PROJECT_ROOT}/checkpoints"
SAMPLES_DIR = f"{PROJECT_ROOT}/samples"
TEST_DIR = f"{PROCESSED_DIR}/test"

# =============================================================================
# AUDIO SETTINGS
# =============================================================================

TARGET_SAMPLE_RATE = 22050  # StyleTTS2 default
TARGET_CHANNELS = 1         # Mono
MIN_DURATION_SEC = 0.5
MAX_DURATION_SEC = 30.0
SILENCE_THRESHOLD_DB = -40  # dB below peak for silence detection
SILENCE_MIN_DURATION = 0.1  # seconds

# =============================================================================
# STYLETTS2 CONFIGURATION
# =============================================================================

TTS_FRAMEWORK = "StyleTTS2"
TTS_REPO = "https://github.com/yl4579/StyleTTS2"
TTS_KHMER_REPO = "https://github.com/mrrtmob/styletts2-khmer"
TTS_LICENSE = "MIT"

# StyleTTS2 model URLs
STYLETTS2_LJSPEECH_CHECKPOINT = (
    "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LJSpeech/"
)
STYLETTS2_LIBRITTS_CHECKPOINT = (
    "https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/"
)

# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

MIN_AUDIO_SAMPLES = 100        # Minimum audio files required
MAX_DUPLICATE_TEXT_RATIO = 0.3 # Warn if >30% text duplicates
MIN_UNIQUE_CHARACTERS = 50     # Minimum unique chars expected for Khmer

# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("M3-TTS Project Configuration")
    print("=" * 60)
    print(f"  TTS Framework:    {TTS_FRAMEWORK}")
    print(f"  TTS License:      {TTS_LICENSE}")
    print(f"  Target Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  Target Channels:  {TARGET_CHANNELS}")
    print(f"  Audio Column:     {AUDIO_COLUMN}")
    print(f"  Text Column:      {TEXT_COLUMN}")
    print(f"  Speaker Column:   {SPEAKER_COLUMN}")
    print(f"  Dataset Dir:      {DATASET_DIR}")
    print(f"  Metadata TSV:     {METADATA_TSV}")
    print(f"  Reports Dir:      {REPORTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
