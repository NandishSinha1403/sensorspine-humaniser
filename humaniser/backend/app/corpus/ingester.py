import fitz  # PyMuPDF
import re
import os
import numpy as np
from typing import List, Dict, Any
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from app.corpus.style_profile import save_profile, load_profile

TRANSITION_PHRASES = ["however", "although", "nevertheless", "furthermore", "consequently", "moreover", "therefore", "instead", "conversely"]
HEDGING_PHRASES = ["it appears that", "data suggest", "one might argue", "it is possible that", "this indicates", "suggests", "seems to", "potentially"]

def clean_text(text: str) -> str:
    # Strip reference sections (common pattern)
    text = re.split(r'\n(?:References|Bibliography)\n', text, flags=re.IGNORECASE)[0]
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Strip page numbers and short lines (likely headers/footers)
        if re.match(r'^\d+$', line):
            continue
        if len(line) < 20:
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def extract_features(text: str) -> Dict[str, Any]:
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    content_words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # Sentence stats
    sent_lengths = [len(word_tokenize(s)) for s in sentences]
    stats = {
        "mean": float(np.mean(sent_lengths)),
        "std": float(np.std(sent_lengths)),
        "median": float(np.median(sent_lengths)),
        "distribution": {
            "short (0-15)": len([l for l in sent_lengths if l <= 15]),
            "medium (16-30)": len([l for l in sent_lengths if 15 < l <= 30]),
            "long (31-45)": len([l for l in sent_lengths if 30 < l <= 45]),
            "extra_long (45+)": len([l for l in sent_lengths if l > 45])
        }
    }
    
    # Paragraph openers
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    openers = []
    opening_sentences = []
    for p in paragraphs:
        p_sents = sent_tokenize(p)
        if p_sents:
            opening_sentences.append(p_sents[0])
            p_words = word_tokenize(p_sents[0])
            if p_words:
                first_word = p_words[0].lower()
                if first_word.isalnum() and first_word not in stop_words:
                    openers.append(first_word)
                    
    top_openers = [w for w, count in Counter(openers).most_common(30)]
    
    # Template extraction
    templates = []
    for s in sentences[:100]:
        # Simple template: replace numbers and capitalized words (approx proper nouns)
        # In a real app, use spaCy NER for this
        t = re.sub(r'\d+', '<NUM>', s)
        # Heuristic for proper nouns: capitalized words not at start of sentence
        words_in_s = word_tokenize(t)
        new_words = []
        for i, w in enumerate(words_in_s):
            if i > 0 and w[0].isupper() and w.isalpha():
                new_words.append("<NAME>")
            else:
                new_words.append(w)
        templates.append(" ".join(new_words).replace(" .", ".").replace(" ,", ","))

    # Phrases
    found_transitions = [p for p in TRANSITION_PHRASES if p in text.lower()]
    found_hedging = [p for p in HEDGING_PHRASES if p in text.lower()]
    
    # Vocab and TTR
    unique_words = set(content_words)
    ttr = len(unique_words) / len(words) if words else 0
    top_vocab = [w for w, count in Counter(content_words).most_common(100)]
    
    # Punctuation
    punctuation_counts = Counter(re.findall(r'[,;—\-]', text))
    total_chars = len(text) if text else 1
    punct_profile = {
        "comma_rate": punctuation_counts.get(',', 0) / total_chars * 1000,
        "semicolon_rate": punctuation_counts.get(';', 0) / total_chars * 1000,
        "dash_rate": (punctuation_counts.get('—', 0) + punctuation_counts.get('-', 0)) / total_chars * 1000
    }
    
    return {
        "sentence_stats": stats,
        "paragraph_openers": top_openers,
        "opening_patterns": opening_sentences[:50],
        "transition_phrases": found_transitions,
        "hedging_phrases": found_hedging,
        "top_vocab": top_vocab,
        "ttr": float(ttr),
        "punctuation_profile": punct_profile,
        "sample_sentences": templates
    }

def merge_profiles(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    def union_and_cap(list1, list2, cap):
        return list(dict.fromkeys(list1 + list2))[:cap]

    old_count = old.get("document_count", 1)
    new_count = 1
    total = old_count + new_count
    w_old = old_count / total
    w_new = new_count / total

    merged = {
        "document_count": total,
        "sentence_stats": {
            "mean": old["sentence_stats"]["mean"] * w_old + new["sentence_stats"]["mean"] * w_new,
            "std": old["sentence_stats"]["std"] * w_old + new["sentence_stats"]["std"] * w_new,
            "median": old["sentence_stats"]["median"] * w_old + new["sentence_stats"]["median"] * w_new,
            "distribution": {
                k: old["sentence_stats"]["distribution"].get(k, 0) + new["sentence_stats"]["distribution"].get(k, 0)
                for k in new["sentence_stats"]["distribution"]
            },
        },
        "paragraph_openers": union_and_cap(old["paragraph_openers"], new["paragraph_openers"], 80),
        "opening_patterns": union_and_cap(old.get("opening_patterns", []), new.get("opening_patterns", []), 100),
        "transition_phrases": union_and_cap(old["transition_phrases"], new["transition_phrases"], 80),
        "hedging_phrases": union_and_cap(old["hedging_phrases"], new["hedging_phrases"], 80),
        "top_vocab": union_and_cap(old["top_vocab"], new["top_vocab"], 200),
        "ttr": old["ttr"] * w_old + new["ttr"] * w_new,
        "punctuation_profile": {
            k: old["punctuation_profile"].get(k, 0) * w_old + new["punctuation_profile"].get(k, 0) * w_new
            for k in new["punctuation_profile"]
        },
        "sample_sentences": union_and_cap(old["sample_sentences"], new["sample_sentences"], 150),
    }
    return merged


def ingest_pdf(file_path: str, field: str) -> Dict[str, Any]:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    pages = len(doc)
    doc.close()

    cleaned = clean_text(text)
    new_profile = extract_features(cleaned)
    new_profile["document_count"] = 1

    existing = load_profile(field)
    if existing:
        final_profile = merge_profiles(existing, new_profile)
    else:
        final_profile = new_profile

    save_profile(field, final_profile)
    return {"status": "success", "field": field, "pages_processed": pages, "features": new_profile}
