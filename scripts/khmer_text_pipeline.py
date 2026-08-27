"""
M3-TTS Phase 2 — Khmer Text Pipeline

Romanized Khmer text processing for StyleTTS2.
Our text is already romanized (Latin characters), so no G2P is needed.
The StyleTTS2 TextCleaner vocabulary includes A-Z, a-z which covers our text.
"""

import re
import unicodedata


# Romanized Khmer characters used in our dataset
ROMANIZED_KHMER_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?-'\"()"
)

# Unicode Khmer characters (for reference/validation)
KHMER_CONSONANTS = set("\u1780-\u17A2")  # ក-អ
KHMER_VOWELS = set("\u17A3-\u17B3")      # អិ-អុ
KHMER_DIACRITICS = set("\u17B6-\u17D3")  # ា-៓
KHMER_INDIC_DIGITS = set("\u17E0-\u17E9")  # ០-៩
KHMER_LETTERS = KHMER_CONSONANTS | KHMER_VOWELS | KHMER_DIACRITICS


def normalize_romanized(text):
    """
    Normalize romanized Khmer text for StyleTTS2.

    Steps:
    1. Strip leading/trailing whitespace
    2. Collapse multiple spaces
    3. Remove non-ASCII characters (keep Latin + digits + basic punctuation)
    4. Lowercase (StyleTTS2 TextCleaner is case-sensitive, keep original case)
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    # Keep only characters in the StyleTTS2 vocabulary
    # StyleTTS2 vocab: pad, punctuation, A-Z, a-z, IPA chars
    # Our romanized Khmer uses: A-Z, a-z, digits, basic punctuation
    cleaned = []
    for ch in text:
        if ch in ROMANIZED_KHMER_CHARS:
            cleaned.append(ch)
        elif ch in "[]{}<>":
            cleaned.append(ch)
        else:
            # Skip unknown chars
            pass

    return "".join(cleaned)


def has_khmer_script(text):
    """Check if text contains Unicode Khmer script characters."""
    if not text:
        return False
    return any(ch in KHMER_LETTERS for ch in text)


def detect_text_type(text):
    """Detect whether text is romanized Khmer or Unicode Khmer."""
    if not text:
        return "empty"
    if has_khmer_script(text):
        return "unicode_khmer"
    # Check if it looks like romanized Khmer
    if any(ch in "bcdfghjklmnpqrstvwxyz" for ch in text.lower()):
        return "romanized_khmer"
    return "unknown"


def validate_romanized_text(text):
    """
    Validate romanized Khmer text.

    Returns: (is_valid, issues)
    """
    issues = []
    if not text or not isinstance(text, str):
        return False, ["Empty text"]

    text = text.strip()

    if len(text) < 2:
        issues.append("Text too short")

    # Check for Khmer script (should not be present in romanized)
    if has_khmer_script(text):
        issues.append("Contains Unicode Khmer characters")

    # Check for excessive spaces
    if "  " in text:
        issues.append("Multiple consecutive spaces")

    # Check for empty text after cleaning
    cleaned = normalize_romanized(text)
    if not cleaned:
        issues.append("Text is empty after normalization")

    return len(issues) == 0, issues


def prepare_for_styletts2(text):
    """
    Prepare romanized Khmer text for StyleTTS2 training.

    StyleTTS2 TextCleaner processes:
    - pad ($)
    - punctuation (;:,.!?¡¿—…"«»"" ')
    - letters (A-Z, a-z)
    - IPA characters

    Our romanized text uses only Latin chars, digits, and basic punctuation.
    All of these are in the TextCleaner vocabulary.
    """
    text = normalize_romanized(text)
    # Add pad tokens (StyleTTS2 adds these automatically, but we can add them for verification)
    return text


def test_pipeline():
    """Test the text pipeline with sample inputs."""
    test_cases = [
        # Normal romanized Khmer
        "sou sdei nak teang os knongea.",
        "kyun thngai nih yerng nung niek arp mean teang nak teang os knongea.",
        # With numbers
        "meak nih mean prambuon mek muleah.",
        # With English
        "YouTube ke chea veticah dav pininh muleah.",
        # With punctuation
        "het avey ban chea kak min chhlouy teb sar robos knhom?",
        # Long sentence
        "peal dael yerng char teuk pteah yuol mean arp mean ar romnerh robos meak mnous mihn teang yerng nung char teuk pteah yuol tha yerng nung jong yuol tha romnerh mean teang nak teang os knongea.",
    ]

    print("=" * 60)
    print("Khmer Text Pipeline Test")
    print("=" * 60)

    for i, text in enumerate(test_cases):
        text_type = detect_text_type(text)
        is_valid, issues = validate_romanized_text(text)
        processed = prepare_for_styletts2(text)

        print(f"\nTest {i+1}:")
        print(f"  Input:    {text[:80]}...")
        print(f"  Type:     {text_type}")
        print(f"  Valid:    {is_valid}")
        if issues:
            print(f"  Issues:   {issues}")
        print(f"  Output:   {processed[:80]}...")

    print("\n" + "=" * 60)
    print("Pipeline test complete.")


if __name__ == "__main__":
    test_pipeline()
