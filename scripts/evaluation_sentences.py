"""
M3-TTS Phase 2 — Evaluation Sentences

Fixed Khmer evaluation set for consistent model comparison.
All sentences are in romanized Khmer (Latin characters).
"""

# =============================================================================
# EVALUATION SENTENCES BY CATEGORY
# =============================================================================

EVAL_SENTENCES = {
    "normal": [
        "sou sdei nak teang os knongea.",
        "kyun thngai nih yerng nung niek arp mean teang nak teang os knongea.",
        "knhom som sdei robos meak teang.",
        "meak chheu knongea mean teang meak pteah tiav knong chivit.",
        "yerng lerng teuk mean teang meak mnous te chem chem.",
    ],
    "question": [
        "het avey ban chea kak min chhlouy teb sar robos knhom?",
        "avey ban chea meak min niek robos nak teang?",
        "knhom chham mok sdei robos meak nih?",
        "meak nih mean arp mean avey?",
        "avey ban chea meak mnous min te chem robos nak teang?",
    ],
    "long": [
        "peal dael yerng char teuk pteah yuol mean arp mean ar romnerh robos meak mnous mihn teang yerng nung char teuk pteah yuol tha yerng nung jong yuol tha romnerh mean teang nak teang os knongea.",
        "meak mnous min niek knear robos nak teang min te chem chem te tae srolanh mean teang robos os knongea knong chivit meak mnous.",
        "yerng mok lueb teuk yerng char teuk pteah yuol mean arp mean ar romnerh robos meak mnous mihn teang yerng nung char teuk pteah yuol tha yerng nung jong yuol tha romnerh mean teang nak teang os knongea knong chivit meak mnous.",
    ],
    "relationship": [
        "romnerh robos nak teang os knongea mean teang meak pteah tiav knong chivit.",
        "nak teang os knongea dochchea ban niek robos nak teang min chhlouy teb sar.",
        "peal meak mnous min niek knear robos nak teang min te chem chem te tae srolanh mean teang robos.",
        "meak chheu knongea mean teang meak pteah tiav knong chivit robos nak teang os knongea.",
        "romnerh robos nak teang os knongea mean teang meak pteah tiav knong chivit meak mnous.",
    ],
    "psychology": [
        "meak mnous min niek knear robos nak teang min te chem chem.",
        "romnerh robos nak teang os knongea mean teang meak pteah tiav knong chivit meak mnous.",
        "peal meak mnous min niek knear robos nak teang min te chem chem te tae srolanh mean teang robos.",
    ],
    "self_improvement": [
        "yerng mok lueb teuk yerng char teuk pteah yuol mean arp mean ar romnerh robos meak mnous.",
        "knhom som sdei robos meak teang os knongea mean teang meak pteah tiav knong chivit.",
        "meak chheu knongea mean teang meak pteah tiav knong chivit robos nak teang os knongea.",
    ],
    "numbers": [
        "meak nih mean prambuon mek muleah.",
        "pho mok robos knhom mean pi prambuon mek.",
        "meak nih mean prammeas muleah.",
        "pho mok robos knhom mean prambuon mek.",
    ],
    "english_khmer": [
        "YouTube ke chea veticah dav pininh muleah somrobos nak ponheaktoh matega.",
        "Google ke chea meak mnous robos yerng te chem chem.",
        "Facebook ke chea veticah robos nak teang os knongea.",
    ],
}

# =============================================================================
# FLAT LIST FOR TRAINING SAMPLE GENERATION
# =============================================================================

ALL_EVAL_SENTENCES = []
for category, sentences in EVAL_SENTENCES.items():
    for sent in sentences:
        ALL_EVAL_SENTENCES.append({"category": category, "text": sent})

# =============================================================================
# TEST SENTENCES FOR PILOT EVALUATION
# =============================================================================

PILOT_TEST_SENTENCES = [
    "sou sdei nak teang os knongea.",
    "kyun thngai nih yerng nung niek arp mean teang nak teang os knongea.",
    "het avey ban chea kak min chhlouy teb sar robos knhom?",
    "meak nih mean prambuon mek muleah.",
    "YouTube ke chea veticah dav pininh muleah somrobos nak ponheaktoh matega.",
]


def get_all_sentences():
    """Get all evaluation sentences as a flat list."""
    return [s["text"] for s in ALL_EVAL_SENTENCES]


def get_sentences_by_category(category):
    """Get sentences for a specific category."""
    return [s["text"] for s in ALL_EVAL_SENTENCES if s["category"] == category]


def print_eval_sentences():
    """Print all evaluation sentences."""
    print("=" * 60)
    print("M3-TTS Evaluation Sentences")
    print("=" * 60)
    for category, sentences in EVAL_SENTENCES.items():
        print(f"\n{category.upper()} ({len(sentences)} sentences):")
        for i, sent in enumerate(sentences, 1):
            print(f"  {i}. {sent}")
    print(f"\nTotal: {len(get_all_sentences())} sentences")
    print("=" * 60)


if __name__ == "__main__":
    print_eval_sentences()
