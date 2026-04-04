"""Debug: Which signals are causing post-humanization scores to remain high?"""
import logging
logging.getLogger("humaniser.nlp").setLevel(logging.ERROR)

from app.core.humanizer import humanize_text
from app.core.detector import (
    calculate_burstiness, calculate_syntactic_variance,
    calculate_phrase_score, calculate_perplexity_proxy,
    calculate_mattr, calculate_punctuation_uniformity,
    _calibrate
)
from nltk.tokenize import sent_tokenize, word_tokenize

# Pick samples that scored 100% in the stress test: samples 9, 17, 21, 29, 30, 32, 34, 43, 44
FAILING_SAMPLES = [
    # Sample 9 (Cybersecurity)
    "Cybersecurity is a critical concern in our increasingly digital world. With the rise of cyberattacks, organizations and individuals need to implement robust security measures to protect their data and systems. Common cybersecurity threats include malware, phishing, and ransomware. The use of artificial intelligence and machine learning can help in detecting and responding to cyberattacks in real-time. However, cybercriminals are also using AI to develop more sophisticated and targeted attacks.",
    # Sample 21 (Social media)
    "The impact of social media on mental health is a subject of growing concern among researchers and policymakers. Studies have shown a correlation between social media use and increased rates of anxiety, depression, and loneliness, particularly among young people. The pressure to present a curated and idealized version of one's life, as well as the prevalence of cyberbullying and online harassment, are some of the factors contributing to these negative effects. It is crucial to develop strategies for promoting healthy social media habits.",
]

for i, text in enumerate(FAILING_SAMPLES):
    print(f"\n{'='*60}")
    print(f"SAMPLE {i+1} - BEFORE HUMANIZATION")
    print(f"{'='*60}")
    sents = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    s1 = calculate_burstiness(sents)
    s2 = calculate_syntactic_variance(text)
    s3 = calculate_phrase_score(text)
    s4 = calculate_perplexity_proxy(words)
    s5 = calculate_mattr(words)
    s6 = calculate_punctuation_uniformity(sents)
    
    raw = s1*0.10 + s2*0.30 + s3*0.25 + s4*0.30 + s5*0.03 + s6*0.02
    
    print(f"  Burstiness (0.10):     {s1:6.2f} -> weighted {s1*0.10:.2f}")
    print(f"  Syntactic Var (0.30):  {s2:6.2f} -> weighted {s2*0.30:.2f}")
    print(f"  Phrase Score (0.25):   {s3:6.2f} -> weighted {s3*0.25:.2f}")
    print(f"  Perplexity (0.30):     {s4:6.2f} -> weighted {s4*0.30:.2f}")
    print(f"  MATTR (0.03):          {s5:6.2f} -> weighted {s5*0.03:.2f}")
    print(f"  Punctuation (0.02):    {s6:6.2f} -> weighted {s6*0.02:.2f}")
    print(f"  RAW Total:             {raw:6.2f}")
    print(f"  Calibrated:            {_calibrate(raw):6.2f}%")
    
    # Now humanize
    result = humanize_text(text, intensity=1.0)
    htext = result["humanized_text"]
    
    print(f"\n{'='*60}")
    print(f"SAMPLE {i+1} - AFTER HUMANIZATION")
    print(f"{'='*60}")
    print(f"  Changes: {result['changes_made']}")
    
    sents2 = sent_tokenize(htext)
    words2 = word_tokenize(htext.lower())
    
    s1b = calculate_burstiness(sents2)
    s2b = calculate_syntactic_variance(htext)
    s3b = calculate_phrase_score(htext)
    s4b = calculate_perplexity_proxy(words2)
    s5b = calculate_mattr(words2)
    s6b = calculate_punctuation_uniformity(sents2)
    
    raw2 = s1b*0.10 + s2b*0.30 + s3b*0.25 + s4b*0.30 + s5b*0.03 + s6b*0.02
    
    print(f"  Burstiness (0.10):     {s1b:6.2f} -> weighted {s1b*0.10:.2f}  (was {s1:.1f})")
    print(f"  Syntactic Var (0.30):  {s2b:6.2f} -> weighted {s2b*0.30:.2f}  (was {s2:.1f})")
    print(f"  Phrase Score (0.25):   {s3b:6.2f} -> weighted {s3b*0.25:.2f}  (was {s3:.1f})")
    print(f"  Perplexity (0.30):     {s4b:6.2f} -> weighted {s4b*0.30:.2f}  (was {s4:.1f})")
    print(f"  MATTR (0.03):          {s5b:6.2f} -> weighted {s5b*0.03:.2f}  (was {s5:.1f})")
    print(f"  Punctuation (0.02):    {s6b:6.2f} -> weighted {s6b*0.02:.2f}  (was {s6:.1f})")
    print(f"  RAW Total:             {raw2:6.2f}  (was {raw:.1f})")
    print(f"  Calibrated:            {_calibrate(raw2):6.2f}%  (was {_calibrate(raw):.1f}%)")
    print(f"\n  HUMANIZED TEXT (first 300 chars):")
    print(f"  {htext[:300]}...")
