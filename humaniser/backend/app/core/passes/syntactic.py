"""
Syntactic passes — clefting, syntactic variance enforcement, clause reorder.
(Yoda inversion removed per user request.)
"""

import re
from typing import Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from app.core.passes.utils import get_rng, load_spacy, is_valid_sentence, DEEPENING_PHRASES
from app.core.detector import get_tree_depth


def pass_syntactic_clefting(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Convert SVO sentences into It-Clefts and Noun Clause Fronting.

    Rule 1 (It-Cleft):  "X did Y" → "It was X that did Y"
    Rule 2 (Noun Clause Fronting): "X is important" → "What is important is X"
    """
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_sentences, changes = [], 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if get_rng().random() < (0.15 * intensity) and len(sent) > 10:
            subj = next((t for t in sent if t.dep_ == "nsubj"), None)
            verb = next((t for t in sent if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)

            if subj and verb:
                subj_text = "".join([t.text_with_ws for t in subj.subtree]).strip()
                rest_text = "".join([t.text_with_ws for t in sent if t.i > subj.i]).strip()

                if subj_text and rest_text:
                    if get_rng().random() < 0.5:
                        # Rule 1: It-Cleft
                        tense = "is" if verb.tag_ in ("VBP", "VBZ") else "was"
                        new_sent = f"It {tense} {subj_text.lower()} that {rest_text}"
                    else:
                        # Rule 2: Noun Clause Fronting
                        attr = next((t for t in verb.children if t.dep_ in ("attr", "acomp", "dobj")), None)
                        if attr:
                            attr_text = "".join([t.text_with_ws for t in attr.subtree]).strip()
                            new_sent = f"What {rest_text.rstrip('.')} is {subj_text.lower()}."
                        else:
                            tense = "is" if verb.tag_ in ("VBP", "VBZ") else "was"
                            new_sent = f"It {tense} {subj_text.lower()} that {rest_text}"

                    sent_text = new_sent
                    changes += 1
        new_sentences.append(sent_text)
    return " ".join(new_sentences), changes


def pass_clause_reorder(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Swap subordinate clause position (front ↔ back)."""
    subordinators = {
        "because": "because", "since": "since", "although": "although",
        "while": "while", "whereas": "whereas", "even though": "even though",
    }
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0

    for sent in sentences:
        if get_rng().random() > 0.25 * intensity:
            new_sentences.append(sent)
            continue

        reordered = False
        sent_lower = sent.lower().strip()
        for sub in subordinators:
            if sent_lower.startswith(sub + " ") or sent_lower.startswith(sub + ","):
                comma_idx = sent.find(",")
                if 0 < comma_idx < len(sent) - 5:
                    clause1 = sent[:comma_idx].strip()
                    clause2 = sent[comma_idx + 1:].strip()
                    if clause2:
                        new_sent = (
                            clause2[0].upper() + clause2[1:].rstrip('.') +
                            " " + clause1[0].lower() + clause1[1:] + "."
                        )
                        new_sentences.append(new_sent)
                        changes += 1
                        reordered = True
                        break
        if not reordered:
            new_sentences.append(sent)

    return " ".join(new_sentences), changes


def pass_syntactic_variance(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Enforce depth extremes in every paragraph: one < 4, one > 12.
    Targets the detector's 'Syntactic Variance' metric.
    """
    sp = load_spacy()
    if not sp:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs = []
    total_changes = 0

    # Strictly academic parentheticals — no colloquialisms
    parentheticals = [
        " — a distinction that bears emphasis —",
        " — as the evidence suggests —",
        " — and this warrants attention —",
        " — a finding that merits scrutiny —",
        " — in this particular context —",
        " — as documented in the literature —",
        " — to a notable degree —",
        " — an observation that recurs frequently —",
        " — from an empirical standpoint —",
        " — given the available data —",
    ]

    for paragraph in paragraphs:
        doc = sp(paragraph)
        sentences = list(doc.sents)
        if not sentences:
            processed_paragraphs.append(paragraph)
            continue

        depths = []
        for sent in sentences:
            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            depths.append(get_tree_depth(root) if root else 1)

        has_flat = any(d < 4 for d in depths)
        has_deep = any(d > 12 for d in depths)
        new_sentences = [s.text.strip() for s in sentences]
        changes = 0

        # Force a flat sentence — but NEVER truncate mid-thought.
        # Instead, ADD a short simple sentence rather than destroying an existing one.
        if not has_flat and len(new_sentences) > 1:
            short_codas = [
                "This matters.",
                "The evidence is clear.",
                "The data confirm this.",
                "This cannot be overlooked.",
                "Such findings are telling.",
                "This pattern recurs.",
            ]
            # Insert a short sentence after the 2nd sentence
            insert_idx = min(2, len(new_sentences))
            new_sentences.insert(insert_idx, get_rng().choice(short_codas))
            changes += 1
            has_flat = True

        if not has_deep:
            for i, s_text in enumerate(new_sentences):
                if len(s_text.split()) > 10:
                    s_doc = sp(s_text)
                    root = next((t for t in s_doc if t.dep_ == "ROOT"), None)
                    if root:
                        extra = get_rng().choice(DEEPENING_PHRASES[:3])
                        new_sentences[i] = s_text.rstrip(".!?") + extra + "."
                        changes += 1
                        has_deep = True
                        break

        # Participial inversion and parenthetical injection
        final_paragraph_sentences = []
        for s_text in new_sentences:
            s_doc = sp(s_text)
            sent_text = s_text

            # Participial inversion: "X is Y" → "Being Y, X is ..."
            # Only for mid-length sentences with clear subject-predicate structure
            if 14 < len(s_doc) < 25 and get_rng().random() < (0.10 * intensity):
                root = next((t for t in s_doc if t.dep_ == "ROOT" and t.lemma_ == "be"), None)
                if root:
                    nsubj = next((t for t in root.children if t.dep_ == "nsubj"), None)
                    attr = next((t for t in root.children if t.dep_ in ("attr", "acomp")), None)
                    if nsubj and attr:
                        subj_text = "".join([t.text_with_ws for t in nsubj.subtree]).strip()
                        attr_text = "".join([t.text_with_ws for t in attr.subtree]).strip()
                        remaining_tokens = [t for t in s_doc if t.i > max(nsubj.i, attr.i, root.i)]
                        remaining = "".join([t.text_with_ws for t in remaining_tokens]).strip()
                        if remaining and len(remaining.split()) > 4:
                            new_sent = f"Being {attr_text.lower()}, {subj_text.lower()} {root.text} {remaining}"
                            new_sent = new_sent.strip().rstrip(".,; ") + "."
                            sent_text = new_sent[0].upper() + new_sent[1:]
                            changes += 1

            # Parenthetical injection — rare, academic-only
            if len(s_doc) > 15 and get_rng().random() < (0.12 * intensity):
                root = next((t for t in s_doc if t.dep_ == "ROOT"), None)
                if root and root.pos_ == "VERB":
                    subj = next((t for t in root.children if "subj" in t.dep_), None)
                    if subj:
                        p_text = get_rng().choice(parentheticals)
                        subj_end_idx = subj.i - s_doc[0].i + 1
                        if 2 < subj_end_idx < len(s_doc) - 3:
                            part1 = "".join([t.text_with_ws for t in s_doc[:subj_end_idx]]).strip()
                            part2 = "".join([t.text_with_ws for t in s_doc[subj_end_idx:]]).strip()
                            if len(part2.split()) > 5:  # Ensure remaining part is substantial
                                sent_text = f"{part1}{p_text} {part2}"
                                changes += 1

            final_paragraph_sentences.append(sent_text)

        processed_paragraphs.append(" ".join(final_paragraph_sentences))
        total_changes += changes

    return "\n\n".join(processed_paragraphs), total_changes

