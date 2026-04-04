# Humaniser: Professional AI Text Naturalizer

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

**Humaniser** is an advanced linguistic suite designed to transform AI-generated text into high-quality, human-like prose that consistently bypasses enterprise-grade AI detectors (Turnitin, GPTZero, etc.).

---

## 🚀 Key Features

*   **16+ specialized NLP Passes**: Targeting everything from lexical probability to dependency tree depth.
*   **Hard-Enforced Burstiness**: Guarantees human-like sentence length variation in every paragraph.
*   **Adversarial Bypass Layer**: Includes Unicode whitespace jitter and structural chaos for high-end detector evasion.
*   **Self-Audit System**: Recursive humanization loop that stops only when the AI score is below target.
*   **Modern Jargon Injection**: Disrupts predictability by using sophisticated, context-aware terminology.
*   **Professional UI**: Real-time metrics dashboard built with Next.js and Tailwind CSS.

## 📊 Benchmarks (v2.0)

| Metric | AI Original | Humaniser (Ultra) |
| :--- | :--- | :--- |
| **Average AI Score** | 98.4% | **0.75%** |
| **Bypass Rate (<20%)** | 0% | **94% - 100%** |
| **Perplexity (Proxy)** | Low (Predictable) | **High (Human-like)** |
| **Burstiness (Std Dev)** | 2.1 | **14.8+** |

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js 18+
*   pip & npm

### One-Click Setup
```bash
chmod +x bootstrap.sh start.sh stop.sh
./bootstrap.sh
```

### Launch
```bash
./start.sh
```
The application will be available at:
*   **Frontend**: `http://localhost:3000`
*   **Backend API**: `http://localhost:8000`

---

## 📂 Project Structure

```
humaniser/
├── backend/            # FastAPI, spaCy, NLTK, Linguistic Passes
│   └── app/core/       # The "Brain": humanizer.py & detector.py
└── frontend/           # Next.js, Tailwind, Metrics Dashboard
```

## 📜 Documentation

For a deep dive into the architecture and linguistic passes, see [DOCUMENTATION.md](./DOCUMENTATION.md).

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/NandishSinha1403">Nandish Sinha</a> — SensorSpine</sub>
</p>
