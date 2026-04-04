import arxiv
import fitz # PyMuPDF
import os
import re
import json
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from typing import List, Dict, Any

# Ensure NLTK data is ready
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

OUTPUT_FILE = "/Users/am_nandish/Documents/sensorspine-humaniser/humaniser/backend/app/core/academic_dna.json"
TEMP_DIR = "/Users/am_nandish/Documents/sensorspine-humaniser/humaniser/backend/scripts/temp_papers"

CATEGORIES = ["cs.AI", "cs.LG", "stat.ML"]
TOTAL_PAPERS = 8 # Reduced for reliability

def harvest_papers():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    
    print(f"🚀 Searching arXiv for top papers in {CATEGORIES}...")
    client = arxiv.Client(
        page_size=TOTAL_PAPERS,
        delay_seconds=3,
        num_retries=3
    )
    
    search = arxiv.Search(
        query=f"({' OR '.join(['cat:'+c for c in CATEGORIES])})",
        max_results=TOTAL_PAPERS,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    paper_paths = []
    for result in client.results(search):
        filename = f"{result.get_short_id()}.pdf"
        filepath = os.path.join(TEMP_DIR, filename)
        if not os.path.exists(filepath):
            print(f"📥 Downloading: {result.title[:50]}...")
            result.download_pdf(dirpath=TEMP_DIR, filename=filename)
        paper_paths.append(filepath)
    
    return paper_paths

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"⚠️ Error reading {pdf_path}: {e}")
        return ""

def analyze_corpus(texts: List[str]):
    print("🧠 Analyzing Linguistic DNA...")
    all_sentences = []
    all_lengths = []
    all_ngrams = Counter()
    
    stop_words = {"the", "and", "a", "of", "to", "in", "is", "that", "it", "this", "as", "for", "with", "by", "on", "are", "be"}
    
    for text in texts:
        if not text: continue
        
        # Clean text (remove common PDF artifacts)
        text = re.sub(r'\s+', ' ', text)
        sentences = sent_tokenize(text)
        
        for sent in sentences:
            # Basic cleanup for sentence extraction
            sent = sent.strip()
            if len(sent) < 15: continue # Skip fragments
            
            words = word_tokenize(sent.lower())
            word_count = len(words)
            all_lengths.append(word_count)
            all_sentences.append(sent)
            
            # Extract Bigrams/Trigrams for transitions
            for n in range(2, 4):
                for i in range(len(words)-n+1):
                    ngram = " ".join(words[i:i+n])
                    # Check if any part of the ngram is a stop word (optional, but keep complex ones)
                    all_ngrams[ngram] += 1

    # Filter most common research-y phrases
    common_phrases = [p for p, count in all_ngrams.most_common(500) if any(w not in stop_words for w in p.split())]
    
    dna = {
        "sentence_stats": {
            "mean_length": float(np.mean(all_lengths)),
            "std_dev_length": float(np.std(all_lengths)),
            "max_length": int(np.max(all_lengths)) if all_lengths else 0,
            "min_length": int(np.min(all_lengths)) if all_lengths else 0
        },
        "top_phrases": common_phrases[:300],
        "sample_sentences": all_sentences[:100] # For template injection
    }
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dna, f, indent=2)
    
    print(f"✅ Academic DNA saved to {OUTPUT_FILE}")

def main():
    paper_paths = harvest_papers()
    texts = [extract_text_from_pdf(p) for p in paper_paths]
    analyze_corpus(texts)
    
    # Cleanup
    # print("🧹 Cleaning up temporary files...")
    # for p in paper_paths: os.remove(p)

if __name__ == "__main__":
    main()
