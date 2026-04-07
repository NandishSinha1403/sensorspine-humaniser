import sys
import os
import time
import random
import logging
import numpy as np
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(os.path.join(os.getcwd(), "humaniser/backend"))

from app.core.humanizer import humanize_text
from app.core.detector import detect_ai_score

# Silence logs
logging.getLogger("humaniser.nlp").setLevel(logging.ERROR)

DOMAINS = {
    "Medical": [
        "The patient presents with acute myocardial infarction. Thrombolytic therapy was initiated immediately.",
        "Pharmacological interventions in pediatric oncology require precise dosage calculations based on BSA.",
        "The efficacy of CRISPR-Cas9 in targeting neurodegenerative markers remains a focal point of clinical trials."
    ],
    "Legal": [
        "Pursuant to Section 4(b) of the non-disclosure agreement, the party of the first part shall indemnify the second part.",
        "The doctrine of stare decisis dictates that lower courts must follow the precedents established by higher courts.",
        "Force majeure clauses generally excuse performance in the event of unpredictable cataclysmic occurrences."
    ],
    "Computer Science": [
        "In a distributed system, achieving consensus requires protocols like Paxos or Raft to handle network partitions.",
        "The O(n log n) complexity of the sorting algorithm is optimal for large datasets stored in non-volatile memory.",
        "Asynchronous I/O operations allow the event loop to process other requests while waiting for socket readiness."
    ],
    "Philosophy/Humanities": [
        "The existentialist perspective posits that existence precedes essence, placing the burden of meaning on the individual.",
        "Deconstructionist readings of the text reveal inherent contradictions in the binary opposition of self and other.",
        "The socio-economic stratification of late-capitalist societies facilitates the commodification of digital identity."
    ],
    "Physics/Math": [
        "The wave-particle duality of light is evidenced by the double-slit experiment under non-vacuum conditions.",
        "Solving the Schrödinger equation for a hydrogen atom yields the probability density functions of electron orbitals.",
        "The Riemann Hypothesis concerns the distribution of non-trivial zeros of the zeta function in the complex plane."
    ]
}

EDGE_CASES = [
    ("Citation Minefield", "As noted by [1, 2], and corroborated by Smith et al. (2024), the 'system' is efficient (Jones, 2023; Wang, 2022)."),
    ("LaTeX Math", "The value of $\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$ is crucial for the normalization of the $\Psi$ function."),
    ("Code Mix", "To implement the function `def solve(x): return x * 2`, one must first initialize the local environment variables."),
    ("Ultra-Short", "AI is good. People like it. It works well."),
    ("Ultra-Long Sent", "The integration of advanced linguistic strategies within a framework of adversarial natural language processing facilitates the transformation of machine-generated outputs into a state that is statistically indistinguishable from the nuanced and varied prose produced by experienced human authors across multiple academic and technical disciplines."),
    ("Invisible Chars", "This text has a\u200Bhidden\u200Czero\u200Dwidth characters already."),
    ("List Heavy", "1. First point. 2. Second point. 3. Third point. 4. Fourth point. 5. Fifth point. 6. Sixth point."),
]

def generate_sample(id):
    domain = random.choice(list(DOMAINS.keys()))
    base_texts = DOMAINS[domain]
    # Combine random domain sentences to make a paragraph
    text = " ".join(random.sample(base_texts, k=min(len(base_texts), random.randint(2, 3))))
    return text, domain

print("="*60)
print("🚀 MEGA STRESS TEST: 100 SAMPLES (DOMAINS + EDGE CASES)")
print("="*60)

results = []
start_total = time.time()

# 1. Test Domains (80 samples)
print(f"Testing 80 Domain Samples...")
for i in range(80):
    text, domain = generate_sample(i)
    result = humanize_text(text, intensity=1.0)
    results.append({
        "type": f"Domain: {domain}",
        "original_score": result["original_score"],
        "final_score": result["humanized_score"]
    })
    if (i+1) % 20 == 0:
        print(f"  Processed {i+1}/80...")

# 2. Test Edge Cases (20 samples)
print(f"Testing 20 Edge Case Samples...")
for i in range(20):
    name, text = random.choice(EDGE_CASES)
    result = humanize_text(text, intensity=1.0)
    results.append({
        "type": f"Edge Case: {name}",
        "original_score": result["original_score"],
        "final_score": result["humanized_score"]
    })
    if (i+1) % 10 == 0:
        print(f"  Processed {i+1}/20...")

end_total = time.time()

# --- ANALYZE ---
domain_scores = [r["final_score"] for r in results if "Domain" in r["type"]]
edge_scores = [r["final_score"] for r in results if "Edge Case" in r["type"]]
all_scores = [r["final_score"] for r in results]

print("\n" + "="*60)
print("📊 FINAL RESULTS: 100 SAMPLES")
print("="*60)
print(f"Average AI Score (Overall)  : {np.mean(all_scores):.2f}%")
print(f"Average AI Score (Domains)  : {np.mean(domain_scores):.2f}%")
print(f"Average AI Score (Edge Cases): {np.mean(edge_scores):.2f}%")
print(f"Bypass Rate (<20%)          : {(sum(1 for s in all_scores if s < 20.0)/len(all_scores))*100:.1f}%")
print(f"Bypass Rate (<10%)          : {(sum(1 for s in all_scores if s < 10.0)/len(all_scores))*100:.1f}%")
print(f"Perfect Bypass (0.0%)       : {(sum(1 for s in all_scores if s == 0.0)/len(all_scores))*100:.1f}%")
print(f"Max Score Detected          : {np.max(all_scores):.2f}%")
print(f"Total Time                  : {end_total - start_total:.2f}s")
print("="*60)

# Extrapolated for 10,000
print(f"\nExtrapolated for 10,000 samples:")
print(f"- Expected Avg Score: {np.mean(all_scores):.2f}%")
print(f"- Confidence Interval (95%): ±{1.96 * (np.std(all_scores)/np.sqrt(100)):.2f}%")
print(f"- Estimated Execution Time: {(end_total - start_total) * 100 / 3600:.2f} hours")
print("="*60)
