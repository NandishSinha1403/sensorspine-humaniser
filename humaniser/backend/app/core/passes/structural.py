"""
Structural passes — restructuring, structural reset, structural chaos,
morphological shifting, and appositive injection.
"""

import re
from typing import Tuple, List, Any, Optional, Dict
from nltk.tokenize import sent_tokenize, word_tokenize
from app.core.passes.utils import get_rng, load_spacy, is_valid_sentence


def pass_morphological_shifting(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Identify high-frequency verbs and convert them to noun phrases."""
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_sentences, changes = [], 0

    verb_to_noun = {
        "analyzed": "conducted an analysis of", "analyze": "perform an analysis of",
        "examined": "carried out an examination of", "examine": "conduct an examination of",
        "investigated": "undertook an investigation of", "investigate": "carry out an investigation of",
        "reviewed": "performed a review of", "review": "conduct a review of",
        "discussed": "provided a discussion of", "discuss": "offer a discussion of",
        "concluded": "reached a conclusion regarding", "conclude": "formulate a conclusion about",
        "evaluated": "made an evaluation of", "evaluate": "perform an evaluation of",
        "identified": "achieved the identification of", "identify": "complete the identification of",
    }

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if get_rng().random() < intensity:
            for v_past, n_phrase in verb_to_noun.items():
                pattern = re.compile(rf"\b{v_past}\b", re.IGNORECASE)
                if pattern.search(sent_text):
                    sent_text = pattern.sub(n_phrase, sent_text)
                    changes += 1
        new_sentences.append(sent_text)
    return " ".join(new_sentences), changes


def pass_appositive_injection(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Insert descriptive comma-clauses into the middle of sentences."""
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_sentences, changes = [], 0
    appositives = [
        ", a point that bears repeating,", ", which is often overlooked,",
        ", as many scholars have noted,", ", a detail of significant importance,",
        ", which complicates matters slightly,", ", in a manner of speaking,",
    ]

    for sent in doc.sents:
        sent_text = sent.text.strip()
        chance = 1.0 if intensity >= 1.0 else (0.25 * intensity)
        if len(sent) > 10 and get_rng().random() < chance:
            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            if root:
                subj = next((t for t in root.children if "subj" in t.dep_), None)
                if subj:
                    idx = subj.i - sent[0].i + 1
                    if idx < len(sent) - 2:
                        part1 = "".join([t.text_with_ws for t in sent[:idx]]).strip()
                        part2 = "".join([t.text_with_ws for t in sent[idx:]]).strip()
                        sent_text = f"{part1}{get_rng().choice(appositives)} {part2}"
                        changes += 1
        new_sentences.append(sent_text)
    return " ".join(new_sentences), changes


def pass_restructuring(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Split long sentences at conjunctions or merge very short ones."""
    sp = load_spacy()
    if not sp:
        return text, 0

    doc = sp(text)
    new_sentences = []
    changes = 0
    sentences = list(doc.sents)
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_text = sent.text.strip()
        word_count = len(sent)

        if word_count > 22 and get_rng().random() < intensity:
            split_token = None
            for token in sent:
                if token.pos_ == "CCONJ" and token.head.dep_ == "ROOT":
                    # Guard: don't split compound noun phrases
                    # Check if the conjunction joins two nouns/NPs rather than two clauses
                    prev_token = sent[token.i - sent.start - 1] if token.i > sent.start else None
                    next_token = sent[token.i - sent.start + 1] if token.i - sent.start + 1 < len(sent) else None
                    if prev_token and next_token:
                        if (prev_token.pos_ in ("NOUN", "PROPN", "ADJ") and
                            next_token.pos_ in ("NOUN", "PROPN", "ADJ", "DET")):
                            continue  # This is a compound NP, skip

                    is_list = False
                    for prev in reversed(sent[:token.i - sent.start]):
                        if prev.text.lower() in ("such", "including", "like"):
                            is_list = True
                            break
                        if prev.pos_ == "PUNCT" or prev.i < token.i - 5:
                            break
                    if is_list:
                        continue
                    split_token = token
                    break

            if split_token:
                part1_tokens = [t for t in sent if t.i < split_token.i]
                part2_tokens = [t for t in sent if t.i >= split_token.i]

                if is_valid_sentence(part1_tokens) and is_valid_sentence(part2_tokens):
                    part1 = "".join([t.text_with_ws for t in part1_tokens]).strip().rstrip(",; ")
                    part2 = "".join([t.text_with_ws for t in part2_tokens]).strip()
                    first_word_match = re.match(r'^(\w+)', part2)
                    if first_word_match:
                        first_word = first_word_match.group(1).lower()
                        if first_word in ('and', 'but', 'or', 'yet', 'so'):
                            part2 = part2[len(first_word):].strip()
                    part2 = part2[0].upper() + part2[1:] if part2 else ""
                    new_sentences.append(part1 + ".")
                    new_sentences.append(part2)
                    changes += 1
                    i += 1
                    continue

        if i < len(sentences) - 1 and word_count < 8:
            next_sent = sentences[i + 1]
            next_sent_text = next_sent.text.strip()
            if len(next_sent) < 8 and sent_text and next_sent_text:
                connector = get_rng().choice([", and ", "; ", ", but "])
                combined = sent_text.rstrip('.,; ') + connector + next_sent_text[0].lower() + next_sent_text[1:]
                new_sentences.append(combined)
                changes += 1
                i += 2
                continue

        new_sentences.append(sent_text)
        i += 1

    return " ".join(new_sentences), changes


def pass_structural_reset(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Aggressive structural deconstruction with list-avoidance."""
    sp = load_spacy()
    if not sp or intensity < 0.3:
        return text, 0

    doc = sp(text)
    sentences = list(doc.sents)
    new_sentences = []
    changes = 0
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_text = sent.text.strip()
        s_len = len(sent)

        # Skip first sentence to avoid fragments at paragraph starts
        if i == 0:
            new_sentences.append(sent_text)
            i += 1
            continue

        if s_len > 18 and get_rng().random() < (0.7 * intensity):
            split_done = False
            for token in sent:
                if (token.text.lower() in ("which", "who", "where") and
                    token.dep_ in ("relcl", "nsubj", "nsubjpass", "advmod") and
                    8 < (token.i - sent.start) < s_len - 5):

                    if token.i > sent.start and sent[token.i - sent.start - 1].pos_ in ("VERB", "AUX", "ADP", "DET"):
                        continue

                    is_list = False
                    if token.i > sent.start and sent[token.i - sent.start - 1].text.lower() in ("such", "including", "like", "as"):
                        is_list = True
                    if sent_text.count(",") > 2:
                        is_list = True
                    if is_list:
                        continue

                    idx_off = token.i - sent.start
                    part1_t = [t for t in sent[:idx_off]]
                    part2_t = [t for t in sent[idx_off:]]
                    if is_valid_sentence(part1_t) and is_valid_sentence(part2_t):
                        p1 = "".join([t.text_with_ws for t in part1_t]).strip().rstrip(",; ") + "."
                        p2 = "".join([t.text_with_ws for t in part2_t]).strip()
                        p2 = re.sub(r'^[,;\s]+', '', p2)
                        if p2:
                            p2_lower = p2.lower()
                            if p2_lower.startswith("which "):
                                p2 = "This " + p2[6:]
                            elif p2_lower.startswith("who "):
                                p2 = "They " + p2[4:]
                            elif p2_lower.startswith("where "):
                                p2 = "There, " + p2[6:]
                            p2 = p2[0].upper() + p2[1:]
                            p2 = p2.rstrip(".") + "."
                            new_sentences.append(p1)
                            new_sentences.append(p2)
                            changes += 1
                            split_done = True
                            break

            if not split_done:
                for token in sent:
                    if (token.pos_ == "CCONJ" and token.text.lower() in ("and", "but", "yet", "so") and
                        7 < (token.i - sent.start) < s_len - 5):

                        if token.i > sent.start and sent[token.i - sent.start - 1].pos_ in ("VERB", "AUX"):
                            continue

                        is_list = False
                        for prev in reversed(sent[:token.i - sent.start]):
                            if prev.text.lower() in ("such", "including", "like", "as"):
                                is_list = True
                                break
                            if prev.pos_ == "PUNCT" or prev.i < token.i - 5:
                                break
                        if sent_text.count(",") > 2:
                            is_list = True
                        if is_list:
                            continue

                        idx_off = token.i - sent.start
                        part1_t = [t for t in sent[:idx_off]]
                        part2_t = [t for t in sent[idx_off + 1:]]
                        if is_valid_sentence(part1_t) and is_valid_sentence(part2_t):
                            p1 = "".join([t.text_with_ws for t in part1_t]).strip().rstrip(",; ") + "."
                            p2 = "".join([t.text_with_ws for t in part2_t]).strip()
                            p2 = re.sub(r'^[,;\s]+', '', p2)
                            if p2 and len(p1.split()) > 4 and len(p2.split()) > 4:
                                p2 = p2[0].upper() + p2[1:]
                                p2 = re.sub(r'^(and|but|or|yet|so)\s+', '', p2, flags=re.IGNORECASE)
                                p2 = p2[0].upper() + p2[1:]
                                p2 = p2.rstrip(".") + "."
                                new_sentences.append(p1)
                                new_sentences.append(p2)
                                changes += 1
                                split_done = True
                                break

            if split_done:
                i += 1
                continue

        if s_len < 10 and i < len(sentences) - 1:
            next_sent = sentences[i + 1]
            next_sent_text = next_sent.text.strip()
            if len(next_sent) < 12 and sent_text and next_sent_text and get_rng().random() < 0.6 * intensity:
                connector = get_rng().choice([", and ", " — ", "; in fact, ", ", which means "])
                merged = sent_text.rstrip('.!?') + connector + next_sent_text[0].lower() + next_sent_text[1:]
                new_sentences.append(merged)
                changes += 1
                i += 2
                continue

        new_sentences.append(sent_text)
        i += 1

    return " ".join(new_sentences), changes


def pass_structural_chaos(text: str) -> Tuple[str, int]:
    """Randomly reorders sentences that don't have strong temporal markers."""
    sp = load_spacy()
    if not sp:
        return text, 0
    temporal_markers = {
        "first", "second", "third", "then", "finally", "next", "lastly",
        "initially", "after", "before", "meanwhile", "simultaneously",
        "subsequently", "eventually", "later", "afterward", "to start",
        "to conclude", "moreover", "furthermore", "however", "nevertheless",
    }
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs, total_changes = [], 0
    for p in paragraphs:
        doc = sp(p)
        sentences = list(doc.sents)
        if len(sentences) < 3:
            new_paragraphs.append(p)
            continue
        moveable_indices = [
            i for i, sent in enumerate(sentences)
            if (sent.text.strip().lower().split()[0].rstrip(",;:") if sent.text.strip() else "") not in temporal_markers
        ]
        if len(moveable_indices) > 2:
            shuffled_indices = moveable_indices.copy()
            get_rng().shuffle(shuffled_indices)
            new_sent_list = [s.text.strip() for s in sentences]
            for i, original_idx in enumerate(moveable_indices):
                new_sent_list[original_idx] = sentences[shuffled_indices[i]].text.strip()
            new_paragraphs.append(" ".join(new_sent_list))
            total_changes += 1
        else:
            new_paragraphs.append(p)
    return "\n\n".join(new_paragraphs), total_changes
