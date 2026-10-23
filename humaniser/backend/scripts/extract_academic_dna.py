"""
extract_academic_dna.py — Process downloaded PDFs, extract academic n-grams
and transition phrases, compute sentence length distributions, and build a
high-performance Pickle file with a pre-compiled FlashText automaton.
"""

import fitz  # PyMuPDF
import os
import re
import pickle
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from flashtext import KeywordProcessor

# Ensure NLTK data is ready
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Paths — use relative resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
INPUT_DIR = os.path.join(PROJECT_ROOT, "research papers")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "app", "core", "academic_dna.pkl")

# Stop words for filtering n-grams
STOP_WORDS = {
    "the", "and", "a", "of", "to", "in", "is", "that", "it", "this",
    "as", "for", "with", "by", "on", "are", "be", "an", "at", "or",
    "was", "were", "has", "have", "had", "not", "but", "from",
}

# Common PDF junk patterns to filter out
PDF_JUNK_PATTERNS = [
    r'^\d+$',                       # Bare page numbers
    r'^[A-Z]\.\s*$',               # Single letter sections
    r'^\[?\d+\]',                   # Citation references
    r'arXiv:\d+',                   # arXiv IDs
    r'https?://',                   # URLs
    r'^(fig|figure|table)\s*\.?\s*\d', # Figure/table captions
    r'^\w\.\s*\w\.\s*\w\.',        # Author initials
    r'[=∈∑∏∫≤≥±×÷√∞∂∇]+',        # Math symbols
    r'\{.*\}',                      # LaTeX braces
    r'\\[a-z]+\{',                  # LaTeX commands
    r'^\s*\d+\s*\.\s*\d+',        # Version numbers / decimal numbers alone
]
PDF_JUNK_RE = [re.compile(p, re.IGNORECASE) for p in PDF_JUNK_PATTERNS]

# AI phrases that we want to map TO academic equivalents
AI_TO_ACADEMIC_MAP = {
    "furthermore": "moreover",
    "additionally": "further",
    "in conclusion": "in summation",
    "it is worth noting": "it merits attention",
    "it is important to note": "it is critical to observe",
    "research has shown": "empirical evidence suggests",
    "in today's world": "in the contemporary landscape",
    "it is clear that": "the evidence indicates that",
    "a wide range of": "a broad spectrum of",
    "plays a crucial role": "serves as a cornerstone",
    "has been shown": "has been empirically demonstrated",
    "it is essential": "it remains imperative",
    "in order to": "so as to",
    "due to the fact that": "owing to the circumstance that",
    "in light of": "in consideration of",
    "with regard to": "with reference to",
    "in terms of": "from the standpoint of",
    "as previously mentioned": "as earlier articulated",
    "this demonstrates": "this substantiates",
    "this indicates": "this intimates",
    "this suggests": "this connotes",
    "it is crucial": "it is of paramount importance",
    "delve into": "undertake a thorough examination of",
    "shed light on": "elucidate",
    "it is imperative": "it is of critical necessity",
    "plays an important role": "constitutes a pivotal element",
    "there is no doubt": "it is incontrovertible",
    "needless to say": "it is self-evident",
    "it goes without saying": "it is axiomatic",
    "a variety of": "a heterogeneous array of",
    "when it comes to": "with respect to",
    "the fact that": "the observation that",
    "has become increasingly": "has progressively assumed",
    "significant challenges": "formidable obstacles",
    "significant implications": "far-reaching ramifications",
    "has resulted in": "has precipitated",
    "the development of": "the formulation of",
    "the use of": "the employment of",
    "the impact of": "the ramifications of",
}


def is_junk_sentence(sent: str) -> bool:
    """Filter out PDF artifacts, math, citations, and non-prose fragments."""
    sent = sent.strip()
    if len(sent) < 20:
        return True
    if len(sent) > 1000:
        return True
    for pattern in PDF_JUNK_RE:
        if pattern.search(sent):
            return True
    # Too many numbers or special chars = probably math/data
    alpha_chars = sum(1 for c in sent if c.isalpha())
    if alpha_chars / max(len(sent), 1) < 0.5:
        return True
    # All caps = probably a heading
    words = sent.split()
    if len(words) > 2 and all(w.isupper() for w in words[:3]):
        return True
    return False


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF, skipping first/last pages (usually title/refs)."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        num_pages = len(doc)
        # Skip first page (title/abstract header) and last 2 pages (references)
        start_page = 1 if num_pages > 3 else 0
        end_page = max(start_page + 1, num_pages - 2)
        for page_num in range(start_page, end_page):
            text += doc[page_num].get_text()
        return text
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
        return ""


def analyze_corpus():
    """Process all PDFs and build the Academic DNA pickle file."""
    print("🧠 Analyzing Linguistic DNA from prestigious papers...")

    all_sentences = []
    all_lengths = []
    all_ngrams = Counter()
    all_words = Counter()

    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"❌ No PDF files found in {INPUT_DIR}")
        return

    print(f"📚 Found {len(pdf_files)} PDFs to process.")

    for filename in pdf_files:
        pdf_path = os.path.join(INPUT_DIR, filename)
        file_size = os.path.getsize(pdf_path)
        # Skip suspiciously small files (likely corrupt)
        if file_size < 5000:
            print(f"⏭️  Skipping {filename} (too small: {file_size} bytes)")
            continue

        print(f"📖 Processing: {filename}...")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue

        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'-\s+', '', text)  # Fix hyphenated line breaks
        sentences = sent_tokenize(text)

        good_sentences = 0
        for sent in sentences:
            sent = sent.strip()
            if is_junk_sentence(sent):
                continue

            words = word_tokenize(sent.lower())
            word_count = len(words)
            all_lengths.append(word_count)
            all_sentences.append(sent)
            good_sentences += 1

            # Count vocabulary
            for w in words:
                if w.isalpha() and len(w) > 3 and w not in STOP_WORDS:
                    all_words[w] += 1

            # Extract Trigrams and Quadgrams for academic transitions
            for n in range(3, 5):
                for i in range(len(words) - n + 1):
                    ngram_words = words[i:i + n]
                    ngram = " ".join(ngram_words)
                    # Ensure it has substance (at least one non-stop word)
                    if any(w not in STOP_WORDS and w.isalpha() for w in ngram_words):
                        # Filter out n-grams with numbers/symbols
                        if all(w.isalpha() for w in ngram_words):
                            all_ngrams[ngram] += 1

        print(f"   → Extracted {good_sentences} quality sentences")

    if not all_sentences:
        print("❌ No valid sentences extracted from any PDF.")
        return

    print(f"\n📊 Corpus Summary:")
    print(f"   Total sentences: {len(all_sentences)}")
    print(f"   Mean sentence length: {np.mean(all_lengths):.1f} words")
    print(f"   Std dev: {np.std(all_lengths):.1f}")
    print(f"   Unique vocab (>3 chars): {len(all_words)}")

    # Filter most common academic n-grams (exclude noise)
    common_ngrams = [
        p for p, count in all_ngrams.most_common(2000)
        if count >= 3  # Seen at least 3 times across corpus
    ]
    print(f"   Academic n-grams (freq >= 3): {len(common_ngrams)}")

    # Build Aho-Corasick automaton with AI → Academic replacement map
    automaton = KeywordProcessor()
    for ai_phrase, academic_phrase in AI_TO_ACADEMIC_MAP.items():
        automaton.add_keyword(ai_phrase, academic_phrase)
    print(f"   Automaton keywords loaded: {len(AI_TO_ACADEMIC_MAP)}")

    # Build top vocabulary list
    top_vocab = [
        w for w, c in all_words.most_common(3000)
        if w.isalpha() and len(w) > 3
    ]

    dna_data = {
        "sentence_stats": {
            "mean": float(np.mean(all_lengths)),
            "std": float(np.std(all_lengths)),
            "hist": np.histogram(all_lengths, bins=20)[0].tolist(),
            "lengths": all_lengths,
        },
        "automaton": automaton,
        "top_vocab": top_vocab,
        "common_ngrams": common_ngrams[:1000],
        "sample_sentences": all_sentences[:300],
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(dna_data, f)

    print(f"\n✅ Elite Academic DNA saved to {OUTPUT_FILE}")
    print(f"   Pickle size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")


if __name__ == "__main__":
    analyze_corpus()
