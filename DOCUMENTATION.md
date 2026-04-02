# Humaniser: Technical Documentation & Architecture Deep-Dive

**A Professional Linguistic Suite for Academic Text Naturalization**

---

## 1. Executive Summary

**Humaniser** is an advanced, deterministic Natural Language Processing (NLP) suite designed to transform AI-generated academic text into natural, human-like scholarly writing. 

Unlike conventional "AI Humanizers" that simply wrap generative Large Language Models (LLMs) with a prompt like "rewrite this to sound human," this project takes a **rule-based, mathematical approach**. It works by reverse-engineering the exact statistical signals that enterprise AI detectors (like Turnitin and GPTZero) search for—such as low burstiness, uniform sentence structure, and predictable token sequences—and systematically disrupts them while preserving 100% of the author's original facts and citations.

This documentation is designed to provide a comprehensive understanding of the project's architecture, tools, and algorithms, serving as a complete guide for technical presentations and future development.

---

## 2. System Architecture

The project is divided into two decoupled layers: a high-performance Python backend for NLP processing, and a Next.js React frontend for data visualization and user interaction.

```text
[ User Input (Next.js Dashboard) ] 
       │
       ▼ (REST API: /api/humanize)
[ FastAPI Backend ]
       │
       ├──► 1. Semantic Router (Zero-shot domain classification)
       ├──► 2. Voice Fingerprinting (Extract author's linguistic DNA)
       ├──► 3. Detector Engine (Score 250-word segments)
       │
       ▼
[ The 12-Pass NLP Pipeline ] ◄──► [ NLP Models: spaCy, WordNet, NLTK ]
       │                                          ▲
       ▼                                          │
[ Self-Audit Loop ] ──────────────────────────────┘
       │
       ▼
[ Voice Drift Correction ] (Re-apply author's DNA)
       │
       ▼ (JSON Response: Text, Metrics, Diff Data)
[ Dashboard Render (AI Heatmap, LCS Diff Viewer, Stats) ]
```

---

## 3. Core Technologies & Justification

### Backend (Python)
*   **FastAPI:** Chosen for its asynchronous capabilities and high performance. It handles the heavy computational load of the NLP pipeline without blocking, providing rapid API responses.
*   **spaCy (`en_core_web_md`):** The heavy-lifter of the project. Used for **Dependency Parsing** and **Part-of-Speech (POS) tagging**. spaCy allows the engine to understand *grammar*—it knows exactly where a subject ends and a verb begins, which is required for splitting sentences safely without hallucinating.
*   **NLTK (Natural Language Toolkit):** Used for fast sentence and word tokenization.
*   **WordNet (via NLTK):** An offline lexical database used to find synonyms. By pairing WordNet with spaCy's word vectors, the engine ensures that substituted words make contextual sense.
*   **lemminflect:** Used alongside WordNet to ensure that when a verb is replaced, its tense (past, present participle, etc.) is perfectly preserved.
*   **PyMuPDF (`fitz`):** Used in the Training Server to extract text from academic PDFs to build custom style profiles.

### Frontend (TypeScript / Node.js)
*   **Next.js 14 (App Router):** Provides a robust, server-rendered React framework.
*   **Tailwind CSS:** Used for the highly customized, responsive, and premium UI design.
*   **React Hooks (`useMemo`, `useEffect`):** Used extensively to calculate real-time linguistic metrics (Flesch-Kincaid complexity, Type-Token Ratio) on the client side without lagging the UI.

---

## 4. The AI Detection Engine: How Detectors Work

To defeat AI detectors, the engine must first act like one. The internal `detector.py` module scores text based on the following heavily weighted metrics:

1.  **Burstiness (40% Weight):** AI models generate sentences of highly uniform length (usually 18-22 words). The detector measures the **Standard Deviation** of sentence lengths. A high standard deviation means high burstiness (Human).
2.  **AI Signature Phrases (35% Weight):** AI models are trained to be "helpful," leading to an overuse of transitional clichés like *"Furthermore,"* *"It is important to note,"* and *"Delve into."*
3.  **MATTR - Moving Average Type-Token Ratio (15% Weight):** A measure of vocabulary diversity calculated over a sliding 50-word window to detect repetitive word usage.
4.  **Perplexity Proxy (7% Weight):** Measures how "predictable" the next word is.
5.  **Punctuation Uniformity (3% Weight):** Measures the rhythmic, unnatural consistency of comma placements.

**Segment-Based Scoring:** Detectors do not score a 5,000-word document as a single block. They chunk it into **250-word segments**. Humaniser perfectly mimics this behavior, calculating risk on a per-segment basis.

---

## 5. The Secret Sauce: The 12-Pass Linguistic Pipeline

When text is submitted, it does not just get "rewritten." It is passed through a sequence of mathematical and grammatical transformations (`humanizer.py`).

### Pass 1: Citation Guard
Uses precise Regular Expressions to identify academic citations (`[1, 2, 3]` or `(Smith et al., 2020)`) and math formulas. These are replaced with impenetrable placeholder tokens (e.g., `XCIT0REF`) so the NLP engine ignores them. They are restored at the very end.

### Pass 2: Confidence Gradient
The 250-word segments are scored. 
*   **Low Risk (<25% AI):** Receives a "Light Pass" (basic synonym swapping).
*   **High Risk (>45% AI):** Triggers the "Full Pipeline." This preserves coherence in safe text while aggressively targeting robotic text.

### Pass 3: AI Phrase Replacement
Targets the highest-weighted detection signal. Replaces known AI signatures with punchy, human alternatives (e.g., *"Furthermore, it is important to note"* $\rightarrow$ *"Notably"*).

### Pass 4: Sentence Restructuring (Dependency Parsing)
Uses **spaCy** to map the grammatical tree of a sentence. It finds coordinating conjunctions (`CCONJ` like "and", "but") that connect two independent clauses and physically splits the long AI sentence into two shorter, human-like sentences.

### Pass 5: Paragraph Rhythm Fix
AI models often fall into repetitive structures (e.g., starting three sentences in a row with "The" or "This"). This pass detects those patterns and uses subject inversion to break the monotony.

### Pass 6: Rhythm Sculpting
Calculates the current Standard Deviation of sentence lengths. If it is too low (too uniform), the engine actively targets the longest sentences for splitting, and the shortest sentences for comma-spliced qualifiers.

### Pass 7: Burstiness Injection
Injects domain-specific "template" sentences (e.g., *"The protocol held."*) at mathematical intervals to drastically alter the sentence length variance and spike the text's "burstiness" score.

### Pass 8: Lexical Humanization (Vector Mathematics)
Iterates through adjectives and adverbs. Queries **WordNet** for synonyms. To ensure the synonym makes sense, it calculates the **Cosine Similarity** between the original word's spaCy vector and the synonym's vector. If the similarity is $>0.65$, it replaces the word and inflects the tense using `lemminflect`.

### Pass 9: Style Overlay
Pre-2010 human researchers were "epistemically cautious." AI writes with false confidence. This pass injects academic hedging learned from the Corpus Training Server (e.g., *"The data seem to suggest"*).

### Pass 10: Voice Conversion
Actively converts random Active Voice sentences into Passive Voice (and vice-versa). This structurally alters the token sequence, confusing the LLM predictors used by Turnitin.

### Pass 11: Clause Reordering
Finds subordinate clauses and flips them. (*"Because the test failed, we stopped"* $\rightarrow$ *"We stopped because the test failed."*)

### Pass 12: Discourse Markers
Injects human-like transitional markers (*"Interestingly enough,"*, *"Broadly speaking,"*) to break token predictability.

---

## 6. Voice Fingerprinting & Drift Correction

When you run text through 12 mathematical transformations, it can lose the author's original tone. This is known as **Style Drift**.

**The Solution (`voice.py`):**
Before the pipeline starts, the engine extracts the author's "Linguistic DNA":
*   Average sentence length.
*   Formality ratio (density of words with 4+ syllables).
*   Connector style preference (Simple "and/but" vs Complex "therefore/consequently").
*   Person orientation (1st Person "I/We" vs 3rd Person "The study").

**Pass 13 (Drift Correction):** After humanization, the engine checks the new text against the original DNA. If the pipeline made the text too formal, or injected too many complex connectors, this pass actively trims them back, ensuring the output still sounds like the original author.

---

## 7. The Corpus Training System

AI detectors are trained on millions of documents. Humaniser allows you to fight back by building your own **Style Profiles**.

1.  **Ingestion (`trainer.py` & `ingester.py`):** Users upload PDFs of human-written academic papers from their specific field (e.g., Computer Science, Medicine).
2.  **Extraction:** PyMuPDF reads the text. The engine calculates the average sentence length, extracts common transition phrases, and identifies field-specific vocabulary.
3.  **JSON Profile:** This data is saved as a `[field].json` file.
4.  **Application:** When a user selects "Computer Science" in the dashboard, the pipeline uses the math and vocabulary from the CS JSON profile to guide the transformations.

---

## 8. Dashboard & Frontend Mechanics

The Next.js dashboard is built for absolute transparency. 

*   **Real-Time Metrics:** As the user types, React `useMemo` hooks calculate word count, reading time, Flesch-Kincaid Complexity, and Type-Token Ratio (TTR) instantly.
*   **AI Heatmap:** Maps the per-segment AI scores returned by the backend to background opacity values, creating a visual heat map of exactly which sentences look like AI.
*   **LCS Diff Algorithm:** The frontend uses a custom Longest Common Subsequence (LCS) algorithm to map the original text against the humanized text, rendering a beautiful "Git-style" word-level diff viewer so the user sees exactly what was changed.

---

## 9. Running and Showcasing the Project

### Setup
Ensure you have Python 3.10+ and Node.js 18+ installed.

1.  **Bootstrap the Environment:**
    ```bash
    ./bootstrap.sh
    ```
    *This installs pip/npm dependencies and downloads the required spaCy/NLTK models.*

2.  **Launch the Suite:**
    ```bash
    ./start.sh
    ```
    *This boots three asynchronous processes:*
    *   `localhost:8000` - Core NLP FastAPI Server
    *   `localhost:8001` - Corpus Training Server
    *   `localhost:3000` - Next.js User Dashboard

### Presentation Flow (Showcase Strategy)
1.  **Paste AI Text:** Copy a heavily AI-generated paragraph from ChatGPT into the dashboard.
2.  **Click Analyze:** Show the AI Probability score spiking to 90%+. Switch to the **AI Heatmap** to show exactly which sentences triggered the detector.
3.  **Click Humanize:** Watch the terminal/UI log the pipeline steps. Show the score drop to $<20\%$.
4.  **Show the Diff:** Open the "Show Diff" view to prove that no facts or citations were altered, only syntax and vocabulary.
5.  **Explain the Math:** Use the "Linguistic DNA" panel on the right to show how the "Sentence Variance" mathematically increased to bypass the detector.

---

## 10. Future Scaling & Known Limitations

If this project scales to an enterprise level, the following architectural upgrades are recommended:

1.  **spaCy Memory Footprint:** The `en_core_web_md` model uses ~500MB of RAM. If deployed via multi-worker Gunicorn, memory usage scales linearly. **Solution:** Decouple spaCy into a dedicated microservice (e.g., using Celery/Redis) that handles the dependency parsing workload.
2.  **Regex Compilation:** `humanizer.py` currently compiles regex statements per request. **Solution:** Pre-compile regex dictionaries at the module level during API boot.
3.  **React Diff Algorithm:** The LCS diff viewer runs in $O(N \times M)$ time on the React main thread. For texts exceeding 1,000 words, this can cause UI stutter. **Solution:** Offload the diff calculation to a Web Worker, or use an optimized library like `diff-match-patch`.
4.  **Math/LaTeX Constraints:** While standard citations are protected, heavy inline LaTeX math formulas can occasionally confuse the sentence tokenizer.

---
*Documentation compiled for technical review and architectural presentation.*