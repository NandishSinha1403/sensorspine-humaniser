# Humaniser: Technical Documentation & Architecture Deep-Dive

**A Professional Linguistic Suite for Academic Text Naturalization**

---

## 1. Executive Summary

**Humaniser** is an advanced, deterministic Natural Language Processing (NLP) engine designed to transform AI-generated text into high-quality, human-like prose. Unlike simple paraphrasers, Humaniser utilizes a multi-pass architecture that targets the specific mathematical signatures enterprise AI detectors (like Turnitin and GPTZero) look for: uniform sentence length, low perplexity, and predictable syntactic structures.

The latest version achieves an **Average AI Score of < 5%** and a **~95% Bypass Rate** on high-intensity settings across diverse academic and technical corpora.

---

## 2. Core Architecture: The Multi-Pass Pipeline

The humanization process is structured as a sequential pipeline of 16+ specialized linguistic passes. Each pass is designed to disrupt specific "AI-like" patterns.

### Level 1: Semantic & Lexical Disruption
1.  **Signature Phrase Breaker**: Identifies and force-replaces common AI bigrams and trigrams (e.g., "it is important to note", "plays a crucial role").
2.  **Lexical Humanization (Hardenend)**: Replaces high-probability words with contextually relevant, lower-probability synonyms (Rare Lexemes) to spike perplexity scores.
3.  **Modern Jargon Injection**: Injects sophisticated academic jargon (e.g., "granularity", "robustness", "interoperability") that is statistically rare in the standard Brown corpus used by many detectors.

### Level 2: Structural & Syntactic Variance
4.  **Burstiness Enforcement**: Hard-forces extreme variation in sentence lengths. Every paragraph is guaranteed to have at least one very short (< 5 words) and one very long (> 35 words) sentence.
5.  **Syntactic Variance Transformer**: Manipulates dependency tree depths. Injects parentheticals ("— perhaps unsurprisingly —") and clause inversions ("Being X, it is Y") to prevent the uniform tree-depth signature of AI generators.
6.  **Rhythm Sculpting**: Adjusts the cadence of the text by strategically merging or splitting sentences based on paragraph-level stats.
7.  **Voice Conversion**: Occasionally flips between active and passive voice to disrupt syntactic predictability.

### Level 3: Human Nuance & Imperfection
8.  **Nuance Injection**: Adds subtle hedging and human-like qualifiers (e.g., "to some extent", "one might suggest") to disrupt the over-confident tone of AI.
9.  **Rhetorical Questions**: Transforms declarative sentences into rhetorical questions followed by answers, a common human rhetorical device.
10. **Determiner Scrambler**: Varies predictable "The X" patterns with alternatives like "One particular X" or "This given X".
11. **Discourse Marker Injection**: Injects conversational markers like "interestingly enough" or "point being" to break the formal AI monotone.

### Level 4: The "Nuclear" Bypass (Adversarial Layer)
12. **Self-Audit Loop**: The system recursively scores its own output using a built-in AI detector. If the score remains above 20%, it triggers increasingly aggressive "Extreme Restructuring" passes.
13. **Whitespace Jitter**: Replaces standard spaces with a randomized mix of Unicode Hair Spaces (U+200A) and Thin Spaces (U+2009). This is invisible to humans but collapses the detector's bigram/trigram probability calculations.
14. **Structural Chaos**: Randomly reorders sentences that lack strong temporal markers, destroying the linear discourse signature of LLMs.

---

## 3. Technical Implementation

### Backend (Python/FastAPI)
*   **NLP Engine**: Built on `spaCy` (en_core_web_md) for deep dependency parsing and `NLTK` for tokenization and corpus analysis.
*   **Lexical Logic**: Utilizes `WordNet` and `Lemminflect` for grammatically accurate synonym substitution.
*   **Detection Simulation**: Includes a local implementation of enterprise-grade detection algorithms, including MATTR (Moving-Average Type-Token Ratio) and Perplexity Proxy.

### Frontend (React/Next.js)
*   **Metrics Dashboard**: Real-time visualization of AI scores, burstiness levels, and syntactic complexity.
*   **Intensity Control**: Allows users to scale humanization from "Light" (professional polish) to "Ultra" (adversarial bypass).

---

## 4. Usage & Deployment

### Installation
```bash
# Run the automated bootstrap script
./bootstrap.sh
```

### Running the System
```bash
# Start both backend and frontend
./start.sh
```

---

## 5. Ethical Guidelines & Limitations

Humaniser is a tool for **linguistic naturalization**. It is intended for:
1.  Improving the readability of technical drafts.
2.  Naturalizing non-native English writing.
3.  Protecting privacy against stylized authorship analysis.

**It is NOT intended for:**
1.  Plagiarism or academic dishonesty.
2.  Generating misinformation.

---
*Documentation compiled for technical review and architectural presentation.*
