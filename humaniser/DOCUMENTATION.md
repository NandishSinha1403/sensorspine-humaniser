# Humaniser: Professional Linguistic Suite
## A Comprehensive Guide to Academic Writing Naturalization

---

### CHAPTER 1 — What is Humaniser?

**The Problem**
In the modern academic landscape, even original human work is under threat. Tools like Grammarly or basic spell-checkers often suggest "statistically clean" corrections. When a human writer follows these suggestions, their text becomes more predictable. AI detectors—like those used by Turnitin or GPTZero—flag this predictability as machine-generated. This leads to "false positives," where a researcher or student is accused of using AI for work they wrote entirely themselves.

**The Solution**
Humaniser is a specialized tool that preserves your meaning and hard work while naturalizing the linguistic patterns that detectors misread. It acts as a "stylistic filter," taking text that is structurally uniform (a hallmark of AI) and introducing the messy, varied, and nuanced patterns found in genuine human scholarship.

**What it does NOT do**
Humaniser is not a "writer." It does not generate new ideas, perform research, or write paragraphs from scratch. It is a refinement engine for existing drafts.

**Who it is for**
- **Researchers:** Ensuring that papers intended for top-tier journals aren't caught in automated filters.
- **Students:** Protecting original essays from being falsely flagged by university detection systems.
- **Non-Native English Writers:** Helping writers whose extremely formal or "textbook" English might be mistaken for AI output.

---

### CHAPTER 2 — How AI Detection Works

To beat a detector, you must first understand the signals it is listening for.

#### BURSTINESS
**Definition:** The variation in sentence length across a piece of writing.
Human writing is "bursty." We naturally mix short, punchy sentences with long, complex ones. AI writing is uniform—every sentence tends to be 18–22 words long.
- **The Metric:** We measure this using **Standard Deviation (Std Dev)**. 
- **The Numbers:** If every sentence is exactly 20 words, your Std Dev is 0. If your sentences range from 5 to 40 words, your Std Dev is likely 12 or higher. 
- **The Human Benchmark:** Academic papers from the early 2000s typically show an Std Dev of 8–15. AI text usually hovers between 2 and 5.

#### PERPLEXITY
**Definition:** How surprising or unpredictable the next word is in a sequence.
AI models are trained to pick the most likely next word. This results in "low perplexity." Humans are more creative and unpredictable, resulting in "high perplexity."
- **The Analogy:** A GPS gives perfect, predictable directions (*low perplexity*). A human giving directions says, "Turn left at the big tree, you'll see a weird blue house" (*high perplexity*).

#### AI SIGNATURE PHRASES
Language models are trained to be helpful, structured, and polite. This leads to a high density of "signature phrases" that real humans rarely use in such high concentrations.
- **Examples:** "Furthermore," "it is important to note," "in conclusion," "plays a crucial role," "delve into."
- **The Human Reality:** Humans writing under deadline or focused on data use shorter, rougher connectives like "also," "and," "but," or "so."

#### MATTR (Moving Average Type-Token Ratio)
**Definition:** Vocabulary diversity measured in a sliding window.
- **Type:** Unique words.
- **Token:** Total words.
If you write 100 words and use 60 unique words, your TTR is 0.60. However, longer texts naturally have lower TTRs because common words (the, and, of) repeat more. **MATTR** fixes this by measuring TTR in small 50-word windows and averaging them. AI text usually clusters tightly around MATTR 0.65–0.75.

#### PUNCTUATION UNIFORMITY
AI places commas in very rhythmic, predictable patterns. Human punctuation is "messy"—one sentence might have no commas at all, while the next has four. We measure the Std Dev of comma counts per sentence to detect this machine-like rhythm.

---

### CHAPTER 3 — The 7-Pass Humanization Pipeline

#### Pass 1 — AI Phrase Replacement
**What:** Scans for known AI signature phrases and replaces them with natural human alternatives.
**Why:** These phrases are the single strongest signal detectors use.
- **Example:** *"Furthermore, it is important to note that"* → *"And notably,"*

#### Pass 2 — Sentence Restructuring
**What:** Uses **spaCy** dependency parsing to find unnaturally long sentences to split, or short consecutive ones to join.
**Why:** AI often produces repetitive "Subject-Verb-Object" structures. Restructuring breaks this machine-like cadence.
- **What is spaCy?** An open-source X-ray for grammar. It maps out the grammatical role of every word—identifying exactly where a clause ends or a subject begins.

#### Pass 2.5 — Paragraph Rhythm Fix
**What:** Detects repetitive sentence openings (e.g., three sentences in a row starting with "The" or "This") and rewrites them using subject inversion.
**Why:** AI falls into paragraph-level rhythms that detectors catch even if the individual words are changed.

#### Pass 3 — Burstiness Injection
**What:** Deliberately varies sentence lengths to match human writing distributions.
**How:** It inserts short "punchy" sentences and adds qualifying clauses to medium ones.
**Why:** This is the most effective way to lower an AI score, as detectors weight sentence length variance very heavily.

#### Pass 3.5 — Rhythm Sculpting
**What:** Compares your text’s length distribution against your trained corpus of 2000s papers and actively adjusts your text to match those specific human stats.
**Why:** Generic randomness isn't enough; we want your text to specifically mimic the "heartbeat" of a real research paper.

#### Pass 4 — Lexical Humanization
**What:** Replaces common adjectives and adverbs with less frequent synonyms using **WordNet**.
**What is WordNet?** A massive lexical database from Princeton University that groups words by meaning.
**Why:** AI always picks the "safest," most common word. Humans often reach for the second or third best option to add flavor.

#### Pass 5 — Corpus Style Overlay
**What:** Injects **hedging phrases** learned from your specific field's research papers.
**Why:** Pre-2010 researchers were trained to be **epistemically cautious** (careful about claiming to "know" vs. merely "observing"). AI writes with a false, unearned confidence.
- **Hedging Example:** *"The data seem to suggest"* or *"One might reasonably conclude."*

#### Pass 6 — Self Audit
**What:** Runs the detector on the internal output. If any **segment** (a 250-word chunk, the same unit used by Turnitin) scores above 45, it reruns the entire pipeline on that segment.

#### Pass 7 — Voice Preservation
**What:** Analyzes the author's original "voice fingerprint" (preferred length, formality, person preference) and ensures the final output hasn't drifted too far into "corpus territory."
**Why:** We want the result to sound like *you*, not a generic 2005 journal article.

---

### CHAPTER 4 — The Corpus Training System

**What is a Corpus?**
A corpus is a collection of texts used to study language patterns. In Humaniser, we use a corpus of papers from the early 2000s because they were written before AI writing tools existed. These patterns are "guaranteed human."

**The Style Profile**
Each time you upload a PDF, the **Ingester** extracts:
- Sentence length statistics (Mean/Std Dev)
- Common opening word patterns
- Specific hedging and transition phrases used in that field
- Vocabulary frequency and punctuation habits

This is saved as a **Style Profile** (a JSON file). When you humanize text, the engine loads this profile to use as a "template" for your changes.

**Quality Reporting**
The Training Server includes a **Quality Report** that runs the detector on your own corpus. If your corpus sentences score under 35, the papers are safe to train on. If they score higher, you may have accidentally uploaded AI-influenced work.

---

### CHAPTER 5 — Field-Specific Profiles

Different academic fields have their own "accents." Humaniser allows you to target 8 specific styles:

1. **Computer Science:** Precise, passive-voice heavy, algorithm-centric, and uses "we demonstrate."
2. **Social Sciences:** Heavy hedging, qualitative language, and long literature reviews.
3. **Life Sciences / Medicine:** Methodology-heavy, passive voice, and strict statistical reporting (*p < 0.05*).
4. **Engineering:** Direct, results-focused, referencing figures and tables constantly.
5. **Law:** Extremely formal, Latin-heavy, and full of complex subordinate clauses.
6. **Economics:** Mathematical prose, often using "we find" and "our results suggest."
7. **Business:** Strategy-focused, professional, yet punchy.
8. **Humanities:** Narrative-driven, contextual, and interpretive.

---

### CHAPTER 6 — Tech Stack Explained

- **FastAPI:** A modern, high-performance Python framework used for our backend. It acts as the "brain" that processes your text.
- **Next.js 14:** The React framework used for the Dashboard. It provides a fast, responsive interface.
- **spaCy:** Our industrial-strength NLP library. We use its `en_core_web_sm` model to "understand" the grammar of your sentences.
- **NLTK (Natural Language Toolkit):** Used for tokenization (splitting text into units) and accessing WordNet.
- **WordNet:** Princeton's offline database that allows us to find nuanced synonyms without an internet connection.
- **PyMuPDF (fitz):** The engine that reads your PDFs to build the style profiles.
- **NumPy / SciPy:** The "math engines" that calculate the complex statistics (Standard Deviation, distributions) required to trick detectors.

---

### CHAPTER 7 — Glossary

- **AI Detection:** Identifying text generated by large language models.
- **Burstiness:** The variance in sentence length.
- **Dependency Parsing:** Mapping the grammatical relationships between words.
- **Epistemically Cautious:** Being careful with claims of knowledge (hedging).
- **Humanization:** Matching text to natural human writing patterns.
- **MATTR:** A measure of how diverse your vocabulary is over time.
- **NLP:** Natural Language Processing—how computers read and write.
- **Perplexity:** How unpredictable or "surprising" a sentence is.
- **Profile:** A statistical "fingerprint" of a writing style.
- **Segment:** A 250-word chunk of text (analysis unit).
- **Signature Phrases:** Cliches that AI uses too often.
- **Stylometry:** The mathematical study of linguistic style.
- **Tokenization:** Breaking text into individual words or sentences.
- **WordNet:** An offline English dictionary of meanings and synonyms.

---

### CHAPTER 8 — The Detection Engine in Detail

**The 5 Signals and Their Weights**
The Humaniser detector calculates a composite score based on five key linguistic markers, weighted by their reliability in distinguishing AI from human prose:
1.  **Burstiness (40%):** The variance in sentence length. This is the highest-weighted signal because it is the hardest for LLMs to consistently fake and the most stable metric across different detector implementations.
2.  **Signature Phrases (35%):** The density of "AI-isms" (e.g., "In conclusion," "it is important to note").
3.  **MATTR (15%):** Moving Average Type-Token Ratio, measuring vocabulary diversity.
4.  **Perplexity Proxy (7%):** A calculated measure of word predictability based on frequency and sequence.
5.  **Punctuation Uniformity (3%):** The rhythmic consistency of comma and stop placement.

**The 250-Word Segment Approach**
Detectors rarely analyze a 10,000-word thesis as a single block. Instead, they break text into **250-word segments**. Humaniser mimics this:
-   Scores are calculated for each segment individually.
-   The final score is a weighted average of these segments.
-   If one segment spikes (e.g., a list or a very technical section), the engine identifies it as a "hotspot" for humanization.

**Why burstiness gets highest weight** — hardest signal to fake, most consistent across all detector implementations.

**The Calibration Step**
Raw statistical scores are often "flat." To match the aggressive scoring of commercial detectors, Humaniser applies a calibration curve:
-   **High-Risk Boost:** Scores above 60 are multiplied by **1.3** (max 99).
-   **Low-Risk Reduction:** Scores below 25 are multiplied by **0.7**.
This ensures that "mostly human" text looks very human, and "mostly AI" text is flagged clearly.

**MATTR Window Size 50**
Simple Type-Token Ratio (TTR) fails on long texts because common words eventually repeat, dragging the score down regardless of quality. Humaniser uses a **sliding window of 50 words**. This captures the "local" variety of vocabulary, which is where AI tends to be most repetitive. Simple TTR fails on longer texts because the denominator grows faster than the numerator as vocabulary saturates.

**Example calculation walkthrough for each signal** (Conceptual):
-   **Burstiness:** Sentence lengths [12, 45, 8, 22] -> Std Dev 14.2 -> Human score high.
-   **Signature Phrases:** 3 "Furthermore" in 200 words -> AI score high.
-   **MATTR:** Sliding window of 50 words consistently finds 35+ unique words -> Human score high.

---

### CHAPTER 9 — Author Voice Fingerprinting in Detail

The `voice.py` module extracts a "fingerprint" of the original text before any changes are made. This ensures the final output still sounds like the original author.

**Every feature extracted by voice.py:**
-   **preferred_sentence_length:** The median length the author naturally gravitates toward.
-   **length_variance:** How much the author typically varies their sentence length.
-   **favorite_openers:** A list of the top 3 ways the author starts sentences (e.g., "The", "However", "In").
-   **connector_style:** The ratio of "Simple" (and, but, so) vs. "Complex" (however, therefore, consequently) connectors.
-   **punctuation_habits:** The frequency of semi-colons, dashes, and parentheticals.
-   **formality_score:** Calculated as the ratio of latinate words (4+ syllables) to the total word count.
-   **personal_markers:** Detection of first-person ("I/We") vs. third-person ("The study") orientation.

**How formality_score is calculated:** Ratio of latinate words (4+ syllables) to total words.
**How connector_style ratio works:** counts 'and/but/so' vs 'however/therefore/consequently'.

**Pass 7 and Drift Correction**
"Drift" occurs when the humanization passes (1–6) push the text so far toward the "academic corpus" style that it loses the author's original intent or tone. 
-   **The Fix:** Pass 7 compares the transformed text's fingerprint against the original. If the formality score has jumped by more than 20%, or if the "connector_style" has shifted from simple to complex, Pass 7 "trims" the changes back to align with the author's natural habits.
-   **Concrete example:** informal author with short sentences gets corpus overlay injecting long hedging phrases, Pass 7 detects mismatch and trims them back.

---

### CHAPTER 10 — The Training Server in Detail

Every `trainer.py` endpoint explained:
-   `POST /train/upload`: Upload a single PDF to a specific field profile.
-   `POST /train/upload-batch`: Upload multiple PDFs at once.
-   `GET /train/profiles`: List all active field profiles and their document counts.
-   `GET /train/profile/{field}`: Retrieve the raw statistical data for a specific field.
-   `DELETE /train/profile/{field}`: Reset a profile back to factory defaults.
-   `GET /train/quality-report`: Returns the "Human Probability" score for the trained data.

**The merge algorithm** — how new PDFs merge without losing old data. When a new PDF is uploaded, the engine merges the statistics. Mean sentence lengths are recalculated as weighted averages, and new phrases are added to the existing "phrase bank."

**Why phrase lists are capped at 80 and vocabulary at 200** — To keep the engine fast and prevent "statistical noise" or over-fitting to outliers.

**Quality report scoring** — why under 35 means good training data. If a profile scores higher, it indicates the uploaded PDFs may have been AI-generated or heavily AI-assisted.
**What happens if an AI-written paper is accidentally uploaded** — it injects AI patterns into profile, quality report catches this.

---

### CHAPTER 11 — The 8 Field Profiles Explained

For each of these 8 fields explain typical sentence length range, common hedging patterns, vocabulary characteristics, and why it differs from other fields:

1.  **General:** Baseline. Length 15–25. Standard academic hedging ("it appears").
2.  **Computer Science:** Length 12-20. Direct, active. Uses "We demonstrate". Differs by higher technical noun density.
3.  **Social Sciences:** Length 25–40. Heavy hedging ("One might posit"). Differs by focus on qualitative nuance.
4.  **Life Sciences:** Length 18-28. Passive voice ("Samples were analyzed"). Differs by strict statistical reporting.
5.  **Engineering:** Length 15-22. Results-focused. "Table 1 shows". Differs by practical, industrial vocabulary.
6.  **Medicine:** Length 20-30. Extremely formal, latinate. "Correlation indicates". Differs by high evidentiary threshold in hedging.
7.  **Law:** Length 40-60+. "Notwithstanding", "Hereinafter". Differs by extreme complexity and subordinate clauses.
8.  **Economics:** Length 20-35. Mathematical prose. "ceteris paribus". Differs by blending technical precision with theoretical hedging.

---

### CHAPTER 12 — Complete Workflow Walkthrough

Step by step of exactly what happens internally when a user submits text:
1.  **User pastes text into dashboard.**
2.  **POST /api/detect called** — detector runs 5 signals on 250-word segments.
3.  **User clicks Humanize** — POST /api/humanize called.
4.  **Voice fingerprint extracted** from original text via `voice.py`.
5.  **Pass 1: phrase replacement** — each AI phrase found and swapped for human alternatives.
6.  **Pass 2: spaCy loads**, dependency tree built, split/join decisions made.
7.  **Pass 2.5: paragraph rhythm analyzed**, repetitive openers rewritten.
8.  **Pass 3: burstiness measured**, short sentences and clauses injected.
9.  **Pass 3.5: corpus profile loaded**, length distribution sculpted to match human targets.
10. **Pass 4: WordNet queried**, adjectives/adverbs substituted with low-probability synonyms.
11. **Pass 5: hedging phrases from corpus profile injected** (field-specific).
12. **Pass 6: self audit** — segments re-scored, iterations if needed (Pass 1-5).
13. **Pass 7: voice fingerprint applied**, drift corrected to match original author style.
14. **Response returned** — frontend shows scores, changes, voice card.

---

### CHAPTER 13 — Performance and Limitations

-   **What works well:** 200-3000 words, academic prose, English.
-   **Known limitations:** under 50 words gets reduced pipeline, non-English preserved but not transformed, math-heavy text has limited transformable content.
-   **Why scores are not guaranteed:** detection algorithms change and update, no offline system can guarantee against a live updating detector.
-   **Recommended pattern:** humanize in 500-1000 word sections.
-   **Why more corpus papers helps:** larger phrase banks, more accurate rhythm targets, better hedging variety.

---

### CHAPTER 14 — Complete File Reference

One line description for every file:
-   `humaniser/start.sh`: Bootstrap script for backend and frontend.
-   `humaniser/.env.example`: Environment variable template.
-   `humaniser/DOCUMENTATION.md`: Technical guide for the suite.
-   `humaniser/README.md`: Project overview and setup.
-   `humaniser/README_TRAINING.md`: Guide for the training system.
-   `humaniser/backend/requirements.txt`: Python dependencies.
-   `humaniser/backend/build_corpus.py`: CLI tool for corpus ingestion.
-   `humaniser/backend/test_train.py`: Trainer validation script.
-   `humaniser/backend/app/main.py`: FastAPI application entry point.
-   `humaniser/backend/app/trainer.py`: Logic for training endpoints.
-   `humaniser/backend/app/api/routes.py`: API route definitions.
-   `humaniser/backend/app/core/detector.py`: AI detection logic.
-   `humaniser/backend/app/core/humanizer.py`: Multi-pass humanization engine.
-   `humaniser/backend/app/core/voice.py`: Voice analysis and drift correction.
-   `humaniser/backend/app/corpus/ingester.py`: PDF extraction and cleaning.
-   `humaniser/backend/app/corpus/style_profile.py`: Profile management logic.
-   `humaniser/backend/app/corpus/profiles/`: Storage for field JSON profiles.
-   `humaniser/frontend/src/app/dashboard/page.tsx`: Dashboard UI.
-   `humaniser/frontend/src/app/dashboard/api.ts`: API client for the dashboard.
-   `humaniser/frontend/src/app/layout.tsx`: Next.js root layout.
-   `humaniser/training_ui/index.html`: Corpus management interface.
