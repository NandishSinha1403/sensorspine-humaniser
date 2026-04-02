<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Next.js_14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/spaCy-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-54_Passing-22c55e?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" />
</p>

# 🧠 Humaniser: Professional Linguistic Suite

**Academic Writing Naturalization Engine** — A professional-grade suite that transforms AI-generated text into nuanced, human-like academic writing. It systematically models and disrupts the statistical signals (predictable sequences, uniform structure, signature phrasing) that AI detection systems rely on.

> **Core Idea**: Instead of simple pattern-matching or synonym swapping, Humaniser understands *why* AI text is detectable and breaks those patterns while meticulously preserving the author's original meaning and academic rigor.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **12-Pass Linguistic Pipeline** | Multi-stage transformation: Phrase replacement → sentence restructuring → paragraph rhythm → rhythm sculpting → burstiness injection → lexical substitution → style overlay → voice conversion → clause reordering → discourse markers |
| **250-Word Segment Scoring** | Mirrors how detection systems (like Turnitin and GPTZero) actually process documents — scoring each segment independently to identify "hotspots." |
| **Confidence-Gradient Processing** | Intelligently routes text: segments scoring low get light processing, while high-risk segments trigger the full pipeline to maximize naturalization while preserving coherence. |
| **Voice Fingerprinting & Drift Correction** | Extracts the author's unique "writing DNA" (sentence length patterns, connector preferences, formality level) before processing, and restores it afterwards. |
| **Domain-Aware Style Profiles** | Specialized optimization for 10+ academic fields including Computer Science, Law, Medicine, and Social Sciences. |
| **Citation & Formula Protection** | Academic citations (`[1,2,3]`, `(Smith et al., 2019)`) and mathematical notations are tokenized and protected during the transformation process. |
| **Interactive AI Heatmap** | Real-time dashboard with word-level diff view, live linguistic DNA metrics, and a visual heatmap of AI detection signals. |

---

## 🏗️ Project Architecture

```text
.
├── bootstrap.sh              # Environment setup & NLP model downloader
├── start.sh                  # Single-command production launcher
├── stop.sh                   # Graceful shutdown script
├── humaniser/
│   ├── backend/              # FastAPI Core Engine
│   │   ├── app/
│   │   │   ├── core/         # AI engines (Humanizer, Detector, Voice)
│   │   │   ├── corpus/       # PDF Ingester & Style Profile generator
│   │   │   └── trainer.py    # Corpus training server
│   │   └── tests/            # 54 Automated tests (pytest)
│   ├── frontend/             # Next.js Dashboard UI
│   └── training_ui/          # Standalone Corpus Management UI
└── DOCUMENTATION.md          # Technical Deep-Dive & Algorithm details
```

---

## 🚀 Quick Start

### 1. Bootstrap the Environment
This script installs all Python/Node dependencies and downloads required NLP models (`spaCy`, `WordNet`, `NLTK` corpora).
```bash
./bootstrap.sh
```

### 2. Launch the Suite
Starts the API (8000), Trainer (8001), and Dashboard (3000) in the background.
```bash
./start.sh
```

### 3. Access the Tools
- **Dashboard:** [http://localhost:3000](http://localhost:3000)
- **Training UI:** [http://localhost:8001](http://localhost:8001)
- **API Health:** [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧠 How the Pipeline Works

The Humaniser applies a **12-pass sophisticated pipeline** to every submission:
1. **Citation Guard:** Protects references and formulas.
2. **Confidence Gradient:** Routes text based on initial AI risk.
3. **AI Phrase Replacement:** Swaps machine-learned clichés for natural connectives.
4. **Sentence Restructuring:** Breaks rhythmic patterns using dependency parsing.
5. **Paragraph Rhythm Fix:** Corrects repetitive sentence openings.
6. **Rhythm Sculpting:** Matches sentence length distribution to human corpora.
7. **Burstiness Injection:** Manually varies sentence lengths for high "burstiness."
8. **Lexical Humanization:** Introduces vocabulary diversity via WordNet and spaCy vectors.
9. **Style Overlay:** Injects field-specific hedging and transitional habits.
10. **Voice Conversion:** Fine-tunes active/passive voice orientation.
11. **Clause Reordering:** Enhances flow via structural inversion.
12. **Voice Drift Correction:** Re-aligns the output with the author's original fingerprint.

---

## 🧬 Field-Specific Optimization

Humaniser allows you to target 10 specific academic "accents":
- **Computer Science:** Precise, active-voice, results-centric.
- **Law:** Formal, Latin-heavy, complex subordinate clauses.
- **Social Sciences:** Qualitative nuance, heavy hedging.
- **Life Sciences / Medicine:** Methodology-focused, passive-voice, strict evidentiary thresholds.
- **Engineering:** Direct, data-driven, referencing-heavy.
- **Economics:** Blending theoretical hedging with mathematical precision.
- **And more:** Business, Humanities, General Academic.

---

## 🧪 Testing & Validation

The suite includes **54 automated tests** covering every aspect of the engine:
```bash
cd humaniser/backend
python3 -m pytest tests/ -v
```
Tests cover detector accuracy, all pipeline passes, voice correction stability, citation protection, and end-to-end processing benchmarks.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/NandishSinha1403">Nandish Sinha</a> — SensorSpine</sub>
</p>
