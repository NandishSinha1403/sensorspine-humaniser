import json
import nltk
from nltk.corpus import brown
from nltk.probability import FreqDist
import os

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
            "total_words": len(words)
        }
        
        output_path = os.path.join(os.path.dirname(__file__), "app/core/brown_frequencies.json")
        with open(output_path, "w") as f:
            json.dump(data, f)
            
        print(f"Successfully generated frequencies at {output_path}")
    except Exception as e:
        print(f"Error generating frequencies: {e}")

if __name__ == "__main__":
    generate_frequencies()
