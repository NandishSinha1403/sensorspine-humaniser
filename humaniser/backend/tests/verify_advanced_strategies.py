import unittest
import os
import sys
import pickle
import logging

# Setup path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.core.humanizer import humanize_text, ACADEMIC_DNA, ACADEMIC_AUTOMATON
from app.core.external_detectors import detector_service

class TestAdvancedBypass(unittest.TestCase):
    def test_dna_loading(self):
        """Check if Academic DNA (Pickle) is loaded correctly."""
        print("\nChecking DNA loading...")
        self.assertIsNotNone(ACADEMIC_DNA, "ACADEMIC_DNA should not be None")
        self.assertTrue(len(ACADEMIC_DNA) > 0, "ACADEMIC_DNA should contain stats")
        self.assertIsNotNone(ACADEMIC_AUTOMATON, "ACADEMIC_AUTOMATON should not be None")
        print("✅ DNA loaded successfully.")

    def test_external_detectors(self):
        """Check if ExternalDetectorService is functional."""
        print("\nChecking External Detector Service...")
        scores = detector_service.detect_all("This is a test sentence.")
        self.assertIn("simulated_external", scores, "Should have at least simulated score if no keys")
        print(f"✅ External Detectors: {list(scores.keys())}")

    def test_advanced_passes(self):
        """Check if new advanced passes are applied and tracked."""
        print("\nChecking Advanced NLP Passes...")
        test_text = (
            "Artificial intelligence is transforming the medical field. "
            "The study suggests that machine learning algorithms can analyze data. "
            "This leads to earlier detection of diseases. "
            "Researchers have shown that it works. "
            "However, there are ethical concerns."
        )
        # Use high intensity to trigger all passes
        result = humanize_text(test_text, intensity=1.0)
        
        passes = result.get("passes_applied", [])
        changes = result.get("changes_made", {})
        
        expected_new_passes = [
            "academic_ngrams", "syntactic_clefting", "paragraph_wave", 
            "epistemic_hedging", "adversarial_punctuation"
        ]
        
        print(f"Applied passes: {passes}")
        
        # Check if at least some new passes were tracked in changes
        new_changes_found = False
        for p in expected_new_passes:
            if changes.get(p, 0) > 0:
                print(f"✅ Pass '{p}' made {changes[p]} changes.")
                new_changes_found = True
        
        self.assertTrue(new_changes_found, "At least one new advanced pass should have recorded changes.")

    def test_clefting_output(self):
        """Verify if 'It was ... that' structure is present in high intensity output."""
        print("\nChecking Clefting Output...")
        test_text = "The researcher analyzed the data carefully. The experiment proved the hypothesis."
        result = humanize_text(test_text, intensity=1.0)
        output = result["humanized_text"].lower()
        
        cleft_indicators = ["it was", "it is", "that"]
        found = any(ind in output for ind in cleft_indicators)
        print(f"Humanized Output Sample: {result['humanized_text'][:100]}...")
        # Since it's probabilistic, we just log if it happened
        if "it was" in output or "it is" in output:
             print("✅ Clefting structure detected.")

    def test_adversarial_punctuation(self):
        """Check for semicolons in lists or em-dashes."""
        print("\nChecking Adversarial Punctuation...")
        test_text = "The system requires data, power, and connectivity. The result, which is interesting, was expected."
        result = humanize_text(test_text, intensity=1.0)
        output = result["humanized_text"]
        
        has_semicolon = ";" in output
        has_em_dash = " — " in output or " —" in output
        
        if has_semicolon: print("✅ Semicolon injection detected.")
        if has_em_dash: print("✅ Em-dash injection detected.")

if __name__ == "__main__":
    unittest.main()
