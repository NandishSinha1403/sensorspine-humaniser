"""
Cleanup passes — final enforcement, final cleanup, and human error injection.
"""

import re
from typing import Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from app.core.passes.utils import get_rng, load_spacy, is_valid_sentence
from app.core.detector import get_tree_depth


def pass_final_enforcement(text: str) -> Tuple[str, int]:
    """Hard-enforce burstiness and syntactic variance requirements."""
    sp = load_spacy()
    if not sp:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs, total_changes = [], 0

    for paragraph in paragraphs:
        doc = sp(paragraph)
        sentences = list(doc.sents)
        if not sentences:
            processed_paragraphs.append(paragraph)
            continue

        p_word_count = len(word_tokenize(paragraph))
        lengths = [len(word_tokenize(s.text)) for s in sentences]
        depths = [
            get_tree_depth(next((t for t in s if t.dep_ == "ROOT"), None)) or 1
            for s in sentences
        ]

        has_very_short = any(l < 5 for l in lengths)
        has_very_long = any(l > 35 for l in lengths)
        has_flat = any(d < 4 for d in depths)
        has_deep = any(d > 12 for d in depths)
        new_sentences = [s.text.strip() for s in sentences]
        changes = 0

        if p_word_count > 60:
            # Force a short sentence
            if not has_very_short:
                for i, s_text in enumerate(new_sentences):
                    s_doc = sp(s_text)
                    if len(s_doc) > 10:
                        split_idx = -1
                        for token in s_doc:
                            if token.pos_ in ("VERB", "AUX") and 3 < token.i < len(s_doc) - 4:
                                split_idx = token.i
                                break

                        if split_idx != -1:
                            p1_tokens = s_doc[:split_idx + 1]
                            p2_tokens = s_doc[split_idx + 1:]
                            if is_valid_sentence(p1_tokens) and is_valid_sentence(p2_tokens):
                                p1 = "".join([t.text_with_ws for t in p1_tokens]).strip().rstrip(",; ") + "."
                                p2 = "".join([t.text_with_ws for t in p2_tokens]).strip()
                                if p2:
                                    new_sentences[i] = p1
                                    new_sentences.insert(i + 1, p2[0].upper() + p2[1:])
                                    changes += 1
                                    has_very_short = True
                                    break

            # Force a long sentence
            if not has_very_long and len(new_sentences) >= 2:
                for i in range(len(new_sentences) - 1):
                    s1, s2 = new_sentences[i], new_sentences[i + 1]
                    if len(s1.split()) + len(s2.split()) > 20:
                        merged = s1.rstrip(".!?") + ", and furthermore, " + s2[0].lower() + s2[1:]
                        if len(merged.split()) > 35:
                            new_sentences[i] = merged
                            new_sentences.pop(i + 1)
                            changes += 1
                            has_very_long = True
                            break

        # Force a flat sentence (depth < 4) — insert a short coda, never truncate
        if not has_flat and len(new_sentences) > 1:
            short_codas = [
                "This matters.",
                "The evidence is clear.",
                "The data confirm this.",
                "This cannot be overlooked.",
            ]
            insert_idx = min(2, len(new_sentences))
            new_sentences.insert(insert_idx, get_rng().choice(short_codas))
            changes += 1
            has_flat = True

        # Force a deep sentence (depth > 12)
        if not has_deep:
            deepening_phrases = [
                ", precisely because the underlying modality, though complex, remains relevant.",
                ", a finding that suggests a much deeper level of systemic interaction.",
                ", which effectively underscores the nuanced complexity inherent in the system.",
                ", potentially indicating that the primary factors are more interconnected than previously thought.",
            ]
            for i, s_text in enumerate(new_sentences):
                if len(s_text.split()) > 10:
                    qualifier = get_rng().choice(deepening_phrases)
                    if qualifier.split(",")[1].strip()[:10].lower() not in s_text.lower():
                        new_sentences[i] = s_text.rstrip(".!?") + qualifier
                        changes += 1
                        has_deep = True
                        break

        processed_paragraphs.append(" ".join(new_sentences))
        total_changes += changes

    return "\n\n".join(processed_paragraphs), total_changes


def pass_final_cleanup(text: str) -> Tuple[str, int]:
    """Removes artifacts and orphaned verb sentences. Also fixes common pipeline artifacts."""
    sp = load_spacy()
    if not sp:
        return text, 0

    original_text = text

    # Fix double articles and prepositions
    text = re.sub(r'\b(a|an|the)\s+(a|an|the)\s+', r'\1 ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(of|in|on|at|to|for|with|by)\s+(of|in|on|at|to|for|with|by)\s+', r'\1 ', text, flags=re.IGNORECASE)

    # Fix spacing and duplicate punctuation
    text = re.sub(r'[\s,;>]+([.!?])', r'\1', text)
    text = re.sub(r'\s+([.,])\s+', r'\1 ', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r',,+', ',', text)
    text = re.sub(r';\s*;', ';', text)
    text = re.sub(r'—\s*—', '—', text)

    # Fix orphaned sentence fragments: ". Word." where Word is alone
    text = re.sub(r'\.\s+([A-Z][a-z]+)\.\s+', r'. \1 ', text)

    doc = sp(text)
    new_sentences, removed_count = [], 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Remove very short fragments (< 4 words, likely pipeline artifacts)
        words = [t for t in sent if not t.is_punct and not t.is_space]
        if len(words) < 3 and len(new_sentences) > 0:
            removed_count += 1
            continue

        first_token = next((t for t in sent if not t.is_punct and not t.is_space), None)
        if first_token and first_token.pos_ in ("VERB", "AUX"):
            if first_token.tag_ == "VBG":
                new_sentences.append(sent_text)
                continue
            if not any(
                t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl")
                for t in sent
            ):
                removed_count += 1
                continue
        new_sentences.append(sent_text)

    final_text = " ".join(new_sentences)

    # Final pass: fix capitalization after sentence breaks
    final_text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), final_text)

    return final_text, (1 if final_text != original_text else 0)


def pass_human_errors(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Occasionally introduce a minor human-like punctuation choice."""
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0
    for sent in sentences:
        if ":" in sent and get_rng().random() < 0.1 * intensity:
            sent = sent.replace(":", " —", 1)
            changes += 1
        new_sentences.append(sent)
    return " ".join(new_sentences), changes
