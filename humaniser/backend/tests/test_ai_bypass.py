import sys
import unittest

from app.core.detector import detect_ai_score
from app.core.humanizer import humanize_text

AI_TEXTS = [
    # A complete, multi-paragraph AI generated text to properly test statistical variance
    # algorithms which need at least 15-20 sentences to build standard distributions
    (
        "In recent years, the rapid advancement of artificial intelligence has fundamentally transformed "
        "the technological landscape. Furthermore, as we delve into these predictive algorithms, it is "
        "crucial to note that neural networks play a significantly impactful role. Ultimately, this "
        "demonstrates that leveraging advanced big data processing is essential for achieving success. "
        "In conclusion, it is undeniable that society must adapt. "
        "This study aims to investigate the complex relationship between socioeconomic status and "
        "academic performance. It has been shown that a wide range of factors contribute to student "
        "success. Additionally, it is worth noting that early intervention provides substantial benefits. "
        "Taking everything into consideration, these findings shed light on the imperative need for "
        "comprehensive educational reform. "
        "The mitochondria is the powerhouse of the cell. It provides energy for all cellular functions. "
        "Without this energy, the cell would not survive. This highlights its immense importance in "
        "biology. Furthermore, it regulates cellular metabolism effectively. In today's world, researchers "
        "continue to explore its vast potential."
    )
]

class TestAIBypass(unittest.TestCase):
    def test_evasion_capabilities(self):
        print("\n" + "="*60)
        print("Executing AI Detection Evasion Suite".center(60))
        print("="*60 + "\n")
        
        for i, original_text in enumerate(AI_TEXTS, 1):
            print(f"--- Processing Paragraph {i} ---")
            
            # 1. Initial AI Detection
            initial_score = detect_ai_score(original_text)
            print(f"[Initial] Turnitin-Clone AI Score : {initial_score:.1f}%")
            
            # The AI signature should be blatantly obvious to the detector
            self.assertGreater(initial_score, 75.0, f"Paragraph {i} did not initially trigger high AI detection.")
            
            # 2. Humanize Text
            print("→ Running Subconscious Humaniser Pipeline...")
            result = humanize_text(original_text, intensity=1.0)
            humanized_text = result["humanized_text"]
            
            print(f"    [Result] {humanized_text}")
            
            # 3. Post-Humanization AI Detection
            final_score = detect_ai_score(humanized_text)
            print(f"[Final] Turnitin-Clone AI Score   : {final_score:.1f}%")
            
            # The humanized text must bypass the rigorous Turnitin detector
            self.assertLess(final_score, 20.0, f"Paragraph {i} failed to bypass AI detection! Score: {final_score}")
            print("    ✅ SUCCESS: Detector bypassed.\n")


if __name__ == "__main__":
    unittest.main()
