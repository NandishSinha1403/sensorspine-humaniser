"""
generate_brown_data.py — Generate Brown corpus frequency data for the
perplexity proxy detector. Outputs to app/core/brown_frequencies.json.
"""

import json
import os
import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist


def generate_frequencies():
    print("Generating Brown corpus frequencies...")
    try:
        nltk.download('brown', quiet=True)
        words = [w.lower() for w in brown.words() if w.isalnum()]

        # Unigram frequencies
        word_freq = dict(FreqDist(words))

        # Bigram frequencies
        bigram_freq = {}
        for i in range(len(words) - 1):
            pair = f"{words[i]}|{words[i + 1]}"
            bigram_freq[pair] = bigram_freq.get(pair, 0) + 1

        data = {
            "unigrams": word_freq,
            "bigrams": bigram_freq,
            "total_words": len(words),
        }

        # Resolve output path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", "app", "core", "brown_frequencies.json")
        output_path = os.path.abspath(output_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f)

        print(f"✅ Successfully generated frequencies at {output_path}")
        print(f"   Unigrams: {len(word_freq)}, Bigrams: {len(bigram_freq)}")
    except Exception as e:
        print(f"❌ Error generating frequencies: {e}")


if __name__ == "__main__":
    generate_frequencies()
