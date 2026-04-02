import random
import hashlib
import re
import numpy as np
import nltk
import logging
from typing import List, Dict, Any, Optional, Tuple
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from lemminflect import getInflection, getLemma
from app.corpus.style_profile import load_profile
from app.core.detector import (
    detect_ai_score, score_sentences, calculate_burstiness,
    calculate_perplexity_proxy, _split_into_segments, score_segment
)

# Configure Logging
logger = logging.getLogger("humaniser.nlp")
logger.setLevel(logging.DEBUG)

# Global spaCy model variable for lazy loading
nlp = None

# ===========================================================================
# AI Phrase Replacement Map
# ===========================================================================
AI_PHRASE_REPLACEMENTS = {
    "furthermore": ["also", "and", "beyond this", "on top of that"],
    "additionally": ["also", "and", "as well", "on top of this"],
    "it is worth noting": ["notably", "interesting", "worth mentioning"],
    "it is important to note": ["importantly", "notably", "worth noting"],
    "in conclusion": ["overall", "in the end", "to close"],
    "to summarize": ["briefly", "in short", "to recap"],
    "consequently": ["so", "as a result", "because of this"],
    "it is essential": ["one must", "it matters that", "crucially"],
    "in order to": ["to", "so as to"],
    "due to the fact that": ["because", "since", "given that"],
    "in light of": ["given", "considering", "in view of"],
    "has been shown": ["appears", "seems", "the evidence suggests"],
    "research has shown": ["studies suggest", "evidence points to", "work in this area indicates"],
    "plays a crucial role": ["matters greatly", "is central to", "drives"],
    "delve into": ["examine", "look at", "explore"],
    "it is clear that": ["clearly", "evidently", "it seems"],
    "in today's world": ["today", "currently", "at present"],
    "in recent years": ["recently", "over the past few years", "lately"],
    "needless to say": ["of course", "naturally", "unsurprisingly"],
    "it goes without saying": ["of course", "naturally"],
    "a wide range of": ["many", "various", "a number of"],
    "a variety of": ["many", "several", "various"],
    "when it comes to": ["regarding", "on", "concerning"],
    "with regard to": ["on", "regarding", "about"],
    "in terms of": ["for", "regarding", "on"],
    "the fact that": ["that", "how"],
    "this demonstrates": ["this shows", "this points to", "the data here reflect"],
    "this indicates": ["this points to", "this shows", "the numbers reflect"],
    "this suggests": ["this points to", "it seems", "the evidence here implies"],
    "as previously mentioned": ["as noted", "as discussed", "earlier we saw"],
    "it can be seen": ["we can see", "the data show", "it appears"],
    "has become increasingly": ["has grown more", "is now more", "has turned more"],
    "it is crucial": ["it matters", "this is key", "critically"],
    "it is necessary": ["one must", "we need to", "this requires"],
    "in the context of": ["within", "in", "under"],
    "with respect to": ["on", "regarding", "about"],
    "this highlights": ["this shows", "this points to", "this reveals"],
    "this underscores": ["this reinforces", "this confirms", "this shows"],
    "shed light on": ["clarify", "reveal", "help explain"],
    "it is imperative": ["it is critical", "one must", "we must"],
    "in the realm of": ["in", "within", "across"],
    "plays an important role": ["matters", "is important", "contributes significantly"],
    "there is no doubt": ["clearly", "evidently", "the evidence confirms"],
    "as mentioned earlier": ["as noted", "as discussed above", "earlier"],
    "taking everything into consideration": ["overall", "on balance", "weighing all this"],
    "all things considered": ["overall", "on balance", "in the end"],
    "reshaping modern industries": ["changing how industries work", "shifting modern business"],
    "by enabling machines to": ["by letting machines", "through machines that"],
    "data-driven decisions": ["decisions based on facts", "informed choices", "evidence-led logic"],
    "improve operational efficiency": ["make things run smoother", "speed up workflows"],
    "large volumes of data": ["massive datasets", "huge amounts of info", "mountains of data"],
    "difficult for humans to analyze manually": ["hard for people to track alone", "too much for manual review"],
    "support complex decision-making processes": ["help with tough calls", "aid in tricky choices"],
}

# Map spaCy POS to WordNet POS
SPACY_TO_WORDNET = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
}

DOMAIN_KEYWORDS = {
    "computer_science": ["algorithm", "software", "computing", "network", "data", "processing", "server", "distributed", "routing", "latency", "interface", "protocol"],
    "engineering": ["system", "infrastructure", "design", "efficiency", "optimization", "prototype", "load", "structural", "mechanical", "electrical", "material", "testing"],
    "medicine": ["clinical", "patient", "treatment", "symptoms", "diagnosis", "medical", "outcome", "protocol", "therapy", "trial", "hospital", "physician"],
    "law": ["statute", "precedent", "legal", "jurisprudence", "court", "ruling", "statutory", "litigation", "jurisdiction", "burden", "justice", "article"],
    "social_sciences": ["qualitative", "social", "behavior", "cultural", "society", "demographic", "narrative", "contextual", "observation", "study", "human", "community"],
    "life_sciences": ["biological", "organism", "cellular", "genetic", "sample", "experimental", "species", "evolution", "metabolic", "protein", "dna", "lab"],
    "economics": ["market", "fiscal", "monetary", "capital", "inflation", "growth", "supply", "demand", "equilibrium", "economic", "policy", "trade"],
    "business": ["strategy", "revenue", "roi", "market", "corporate", "management", "operational", "growth", "industry", "profit", "stakeholder", "firm"],
}

FIELD_TEMPLATES = {
    "medicine": ["The clinical outcome was clear.", "Patient history played a role.", "Protocols were maintained.", "Side effects remained minimal."],
    "law": ["The precedent is established.", "Jurisprudence supports this.", "The legal burden was met.", "Statutes provide clarity here."],
    "engineering": ["The system design holds.", "Efficiency was optimized.", "Testing confirmed the load.", "The prototype functioned well."],
    "humanities": ["A narrative arc emerges.", "Contextual layers matter.", "The symbolism is profound.", "Cultural impact remains."],
    "business": ["The market reacted swiftly.", "Margins remained stable.", "ROI met expectations.", "Strategy drove the growth."],
    "computer_science": ["The algorithm converged.", "Latency dropped.", "Throughput matched targets.", "The protocol held."],
    "social_sciences": ["The social dynamics shifted.", "Behavioral patterns emerged.", "Culture played a role.", "The community responded."],
    "life_sciences": ["The organism adapted.", "Cellular response was noted.", "Genetic markers appeared.", "The samples confirmed this."],
    "economics": ["Market equilibrium shifted.", "Fiscal policy mattered.", "Supply matched demand.", "Growth remained steady."],
    "general": ["This matters.", "The pattern holds.", "Results confirmed this.", "Evidence supports this view."],
}

# Discourse markers for injecting unpredictability
DISCOURSE_MARKERS = [
    "as one might expect,", "interestingly enough,", "to some extent,",
    "in a sense,", "broadly speaking,", "at first glance,",
    "on closer inspection,", "to put it differently,", "in practical terms,",
    "from a broader perspective,", "curiously,", "in retrospect,",
]

# ===========================================================================
# Utility functions
# ===========================================================================

def load_spacy():
    global nlp
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load("en_core_web_md")
        except OSError:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.error("[SpaCy] No spaCy model found. Install with: python -m spacy download en_core_web_sm")
                nlp = False
        except ImportError:
            logger.error("[SpaCy] spaCy is not installed. Install with: pip install spacy")
            nlp = False
    return nlp


def classify_domain(text: str) -> str:
    """Zero-shot keyword classification for domain routing."""
    text_lower = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += text_lower.count(kw)

    sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = sorted_domains[0]

    if top_score > 0:
        total = sum(scores.values())
        confidence = (top_score / total) * 100
        logger.info("[Semantic Router] Document classified as: %s with %.1f%% confidence", top_domain, confidence)
        return top_domain

    logger.info("[Semantic Router] No strong domain match. Falling back to 'general'.")
    return "general"


def get_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)


def is_valid_sentence(tokens: List[Any]) -> bool:
    """Sanity Check: Ensure sentence has a VERB and an nsubj."""
    has_verb = any(t.pos_ == "VERB" or t.pos_ == "AUX" for t in tokens)
    has_subj = any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass") for t in tokens)
    return has_verb and has_subj


# ===========================================================================
# Citation protection
# ===========================================================================
CITATION_PATTERNS = [
    r'\[\d+(?:,\s*\d+)*\]',
    r'\(\w+(?:(?:\s+et\s+al\.?)?,\s*\d{4})?\)',
]

def protect_citations(text: str) -> Tuple[str, Dict[str, str]]:
    citations: Dict[str, str] = {}
    protected_text = text
    for pattern in CITATION_PATTERNS:
        matches = re.findall(pattern, protected_text)
        for match in set(matches):
            # Use word-like token that spaCy won't split (no underscores/punctuation)
            token = f"XCIT{len(citations)}REF"
            citations[token] = match
            protected_text = protected_text.replace(match, token)
    return protected_text, citations


def restore_citations(text: str, citations: Dict[str, str]) -> str:
    restored = text
    for token, original in citations.items():
        restored = restored.replace(token, original)
    return restored


# ===========================================================================
# Pass 1: AI Phrase Replacement
# ===========================================================================
def pass_phrase_replacement(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    count = 0
    new_text = text
    sorted_phrases = sorted(AI_PHRASE_REPLACEMENTS.keys(), key=len, reverse=True)

    for phrase in sorted_phrases:
        # Intensity controls probability of replacing
        if random.random() > intensity:
            continue
        replacements = AI_PHRASE_REPLACEMENTS[phrase]
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)

        def replace_fn(match):
            nonlocal count
            original = match.group(0)
            replacement = random.choice(replacements)
            count += 1
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        new_text = pattern.sub(replace_fn, new_text)

    return new_text, count


# ===========================================================================
# Pass 2: Sentence Restructuring (spaCy-based split/join)
# ===========================================================================
def pass_restructuring(text: str, intensity: float = 1.0) -> Tuple[str, int]:
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

        # Split long sentences at coordinating conjunctions
        if word_count > 20 and random.random() < intensity:
            split_token = None
            for token in sent:
                if token.pos_ == "CCONJ" and token.head.dep_ == "ROOT":
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

        # Join very short consecutive sentences
        if i < len(sentences) - 1 and word_count < 8:
            next_sent = sentences[i + 1]
            if len(next_sent) < 8:
                connector = random.choice([", and ", "; ", ", but "])
                combined = sent_text.rstrip('.,; ') + connector + next_sent.text.strip()[0].lower() + next_sent.text.strip()[1:]
                new_sentences.append(combined)
                changes += 1
                i += 2
                continue

        new_sentences.append(sent_text)
        i += 1

    return " ".join(new_sentences), changes


# ===========================================================================
# Pass 2.5: Paragraph Rhythm Fix
# ===========================================================================
def pass_paragraph_rhythm(text: str) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs = []
    total_changes = 0
    for p in paragraphs:
        doc = sp(p)
        sentences = list(doc.sents)
        if len(sentences) < 3:
            new_paragraphs.append(p)
            continue
        new_sents = []
        the_count = 0
        this_count = 0
        for i, sent in enumerate(sentences):
            opener = sent[0].text.lower()
            if opener == "the":
                the_count += 1
                this_count = 0
            elif opener == "this":
                this_count += 1
                the_count = 0
            else:
                the_count = 0
                this_count = 0
            if (the_count >= 2 or this_count >= 2) and i > 0:
                if len(sent) > 2:
                    new_sent = sent[1].text.capitalize() + " " + "".join([t.text_with_ws for t in sent[2:]]).strip()
                    new_sents.append(new_sent)
                    total_changes += 1
                    continue
            new_sents.append(sent.text.strip())
        final_sents = []
        for sent_text in new_sents:
            s_doc = sp(sent_text)
            if len(s_doc) > 35:
                split_token = next(
                    (t for t in s_doc if t.pos_ in ("CCONJ", "PUNCT") and t.head.dep_ == "ROOT" and t.text in (",", ";", "and", "but")),
                    None,
                )
                if split_token:
                    p1, p2 = [t for t in s_doc if t.i < split_token.i], [t for t in s_doc if t.i > split_token.i]
                    if is_valid_sentence(p1) and is_valid_sentence(p2):
                        part1 = "".join([t.text_with_ws for t in p1]).strip().rstrip(",; ") + "."
                        part2 = "".join([t.text_with_ws for t in p2]).strip().capitalize()
                        final_sents.append(part1)
                        final_sents.append(part2)
                        total_changes += 1
                        continue
            final_sents.append(sent_text)
        new_paragraphs.append(" ".join(final_sents))
    return "\n\n".join(new_paragraphs), total_changes


# ===========================================================================
# Pass 3: Burstiness Injection (domain-aware)
# ===========================================================================
def pass_burstiness(text: str, profile: Optional[Dict[str, Any]], field: str = "general", intensity: float = 1.0) -> Tuple[str, int]:
    sentences = sent_tokenize(text)
    if not sentences:
        return text, 0

    lengths = [len(word_tokenize(s)) for s in sentences]
    std_dev = np.std(lengths) if len(lengths) > 1 else 0

    if std_dev >= 8:
        return text, 0

    new_sentences = []
    injections = 0

    sample_sents = profile.get("sample_sentences", []) if profile else []
    # Only use templates from the matched field (fix: no more cross-domain injection)
    templates = FIELD_TEMPLATES.get(field, FIELD_TEMPLATES["general"])
    qualifiers = [
        " — at least in the cases examined here",
        ", though this varies by context",
        ", a finding consistent with earlier work",
        ", which remains a point of debate",
    ]

    for idx, sent in enumerate(sentences):
        new_sentences.append(sent)

        if (idx + 1) % 5 == 0 and random.random() < intensity:
            punchy = random.choice(sample_sents) if sample_sents and random.random() < 0.5 else random.choice(templates)
            if not punchy.endswith('.'):
                punchy += "."
            # Clean any template residue
            punchy = re.sub(r'<[^>]+>', 'this', punchy)
            new_sentences.append(punchy)
            injections += 1

        elif (idx + 1) % 7 == 0 and 10 < len(word_tokenize(sent)) < 25 and random.random() < intensity:
            qualifier = random.choice(qualifiers)
            new_sentences[-1] = new_sentences[-1].rstrip('.,; ') + qualifier + "."
            injections += 1

    return " ".join(new_sentences), injections


# ===========================================================================
# Pass 3.5: Rhythm Sculpting
# ===========================================================================
def pass_rhythm_sculpting(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0
    target_mean = 19
    target_std = 9
    if profile and "sentence_stats" in profile:
        target_mean = profile["sentence_stats"].get("mean", target_mean)
        target_std = profile["sentence_stats"].get("std", target_std)
    doc = sp(text)
    sentences = list(doc.sents)
    if len(sentences) < 5:
        return text, 0
    lengths = [len(s) for s in sentences]
    current_std = float(np.std(lengths))
    changes = 0
    new_sentences = []
    qualifiers = [
        "at least in the cases examined here",
        "though this varies considerably across contexts",
        "a pattern consistent with earlier observations",
    ]
    for s in sentences:
        s_len = len(s)
        if current_std < (target_std - 3) and s_len > target_mean + target_std and random.random() < intensity:
            split_token = next((t for t in s if t.pos_ == "CCONJ" and t.head.dep_ == "ROOT"), None)
            if split_token:
                p1, p2 = [t for t in s if t.i < split_token.i], [t for t in s if t.i >= split_token.i]
                if is_valid_sentence(p1) and is_valid_sentence(p2):
                    new_sentences.append("".join([t.text_with_ws for t in p1]).strip().rstrip(",; ") + ".")
                    new_sentences.append("".join([t.text_with_ws for t in p2]).strip().capitalize())
                    changes += 1
                    continue
        elif current_std < (target_std - 3) and s_len < 12 and random.random() < 0.3 * intensity:
            q = random.choice(qualifiers)
            new_sentences.append(s.text.strip().rstrip('.') + ", " + q + ".")
            changes += 1
            continue
        new_sentences.append(s.text.strip())
    return " ".join(new_sentences), changes


# ===========================================================================
# Pass 4: Lexical Humanization (WordNet + inflection)
# ===========================================================================
def pass_lexical(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0

    doc = sp(text)
    new_tokens = []
    changes = 0

    avoid_words = set(profile.get("top_vocab", []) if profile else [])
    replace_chance = 0.20 * intensity  # intensity modulates substitution probability

    for token in doc:
        if token.pos_ not in SPACY_TO_WORDNET or not token.is_alpha or token.text.lower() in avoid_words:
            new_tokens.append(token.text_with_ws)
            continue

        wn_pos = SPACY_TO_WORDNET[token.pos_]
        if random.random() < replace_chance:
            synsets = wordnet.synsets(token.text.lower(), pos=wn_pos)
            if synsets:
                synonyms = []
                for syn in synsets:
                    for lemma in syn.lemmas():
                        syn_name = lemma.name().replace('_', ' ')
                        if ' ' not in syn_name:
                            synonyms.append(syn_name)

                synonyms = list(dict.fromkeys(synonyms))
                synonyms = [s for s in synonyms if s.lower() != token.text.lower()]

                if synonyms:
                    context_vec = token.sent.vector
                    valid_synonyms = []

                    for syn_word in synonyms:
                        syn_vec_doc = sp.vocab[syn_word]
                        if syn_vec_doc.has_vector:
                            norm_product = np.linalg.norm(context_vec) * np.linalg.norm(syn_vec_doc.vector)
                            if norm_product > 0:
                                similarity = np.dot(context_vec, syn_vec_doc.vector) / norm_product
                                if similarity >= 0.65:
                                    valid_synonyms.append((syn_word, similarity))

                    if valid_synonyms:
                        valid_synonyms.sort(key=lambda x: x[1], reverse=True)
                        replacement_lemma = valid_synonyms[0][0]

                        inflected_replacement = replacement_lemma
                        inflection = getInflection(replacement_lemma, tag=token.tag_)
                        if inflection:
                            inflected_replacement = inflection[0]

                        if token.text[0].isupper():
                            inflected_replacement = inflected_replacement.capitalize()

                        new_tokens.append(inflected_replacement + token.whitespace_)
                        changes += 1
                        continue

        new_tokens.append(token.text_with_ws)

    return "".join(new_tokens), changes


# ===========================================================================
# Pass 5: Corpus Style Overlay (SAFE — never replaces whole sentences)
# ===========================================================================
def extract_primary_subject(text: str) -> str:
    """Extracts the primary subject from the source text using spaCy."""
    sp = load_spacy()
    if not sp:
        return "the subject matter"
    doc = sp(text)
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.dep_ == "ROOT":
            subtree = [t.text for t in token.subtree if t.i <= token.i]
            return " ".join(subtree).lower()
    return "this system"


def pass_style_overlay(text: str, profile: Optional[Dict[str, Any]], field: str = "general", intensity: float = 1.0) -> Tuple[str, int]:
    """
    Injects hedging phrases and opening patterns from corpus.
    SAFE: never replaces entire user sentences (only prepends hedging/patterns).
    """
    if not profile:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs = []
    changes = 0

    opening_patterns = profile.get("opening_patterns", [])
    hedging = profile.get("hedging_phrases", [])

    for p in paragraphs:
        sentences = sent_tokenize(p)
        new_sentences = []

        for s_idx, sent in enumerate(sentences):
            # Opening pattern: prepend a few words from corpus patterns (not replace!)
            if s_idx == 0 and opening_patterns and random.random() < 0.3 * intensity:
                pattern = random.choice(opening_patterns)
                pattern_words = word_tokenize(pattern)[:4]
                sent = " ".join(pattern_words) + ", " + sent[0].lower() + sent[1:]
                changes += 1

            # Hedging injection: prepend hedging phrase periodically
            if hedging and s_idx % 5 == 0 and s_idx > 0 and random.random() < 0.4 * intensity:
                phrase = random.choice(hedging)
                sent = phrase.capitalize() + ", " + sent[0].lower() + sent[1:]
                changes += 1

            new_sentences.append(sent)

        paragraph_text = " ".join(new_sentences)
        paragraph_text = re.sub(r'[<>]', '', paragraph_text)
        new_paragraphs.append(paragraph_text)

    return "\n\n".join(new_paragraphs), changes


# ===========================================================================
# NEW Pass: Active/Passive Voice Conversion
# ===========================================================================
def pass_voice_conversion(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Convert some active voice sentences to passive and vice versa.
    This changes syntactic structure without changing meaning, breaking
    the token-prediction patterns that Turnitin looks for.
    """
    sp = load_spacy()
    if not sp:
        return text, 0

    doc = sp(text)
    new_sentences = []
    changes = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if random.random() > 0.15 * intensity or len(sent) < 5:
            new_sentences.append(sent_text)
            continue

        # Try active → passive: "X verb Y" → "Y is verbed by X"
        root = None
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                root = token
                break

        if root is None:
            new_sentences.append(sent_text)
            continue

        nsubj = None
        dobj = None
        for child in root.children:
            if child.dep_ == "nsubj":
                nsubj = child
            elif child.dep_ in ("dobj", "attr"):
                dobj = child

        if nsubj and dobj:
            # Build passive construction
            subj_text = " ".join([t.text for t in nsubj.subtree])
            obj_text = " ".join([t.text for t in dobj.subtree])
            verb_lemma = root.lemma_

            # Get past participle
            pp = getInflection(verb_lemma, tag="VBN")
            if pp:
                passive = f"{obj_text.capitalize()} was {pp[0]} by {subj_text.lower()}."
                new_sentences.append(passive)
                changes += 1
                continue

        new_sentences.append(sent_text)

    return " ".join(new_sentences), changes


# ===========================================================================
# NEW Pass: Clause Reordering
# ===========================================================================
def pass_clause_reorder(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Reorder clauses in sentences with subordinate structures.
    'Because X, Y happened' → 'Y happened because X'
    'Although X, Y' → 'Y, although X'
    """
    subordinators = {
        "because": "because", "since": "since", "although": "although",
        "while": "while", "whereas": "whereas", "even though": "even though",
    }

    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0

    for sent in sentences:
        if random.random() > 0.2 * intensity:
            new_sentences.append(sent)
            continue

        reordered = False
        sent_lower = sent.lower().strip()
        for sub in subordinators:
            if sent_lower.startswith(sub + " ") or sent_lower.startswith(sub + ","):
                # Find the comma break
                comma_idx = sent.find(",")
                if comma_idx > 0 and comma_idx < len(sent) - 5:
                    clause1 = sent[:comma_idx].strip()
                    clause2 = sent[comma_idx + 1:].strip()
                    if clause2:
                        # Move the subordinate clause to the end
                        new_sent = clause2[0].upper() + clause2[1:].rstrip('.') + " " + clause1[0].lower() + clause1[1:] + "."
                        new_sentences.append(new_sent)
                        changes += 1
                        reordered = True
                        break

        if not reordered:
            new_sentences.append(sent)

    return " ".join(new_sentences), changes


# ===========================================================================
# NEW Pass: Discourse Marker Injection
# ===========================================================================
def pass_discourse_markers(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Insert discourse markers at sentence starts to break token prediction.
    These markers introduce unpredictability without changing meaning.
    """
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0

    for i, sent in enumerate(sentences):
        # Only inject every ~4th sentence, controlled by intensity
        if i > 0 and i % 4 == 0 and random.random() < 0.3 * intensity:
            marker = random.choice(DISCOURSE_MARKERS)
            # Ensure the marker + sentence flows grammatically
            sent = marker + " " + sent[0].lower() + sent[1:]
            changes += 1

        new_sentences.append(sent)

    return " ".join(new_sentences), changes


# ===========================================================================
# AST-style Final Cleanup
# ===========================================================================
def pass_final_cleanup(text: str) -> str:
    """AST-style artifact cleaner to strip orphaned verbs and boundary glitches."""
    sp = load_spacy()
    if not sp:
        return text

    # Pre-clean known text-based glitches
    text = re.sub(r'[\s,;>]+([.!?])', r'\1', text)
    text = re.sub(r'[<>]', '', text)

    doc = sp(text)
    new_sentences = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Detect orphaned verbs at start
        first_token = None
        for token in sent:
            if not token.is_punct and not token.is_space:
                first_token = token
                break

        if first_token and first_token.pos_ in ("VERB", "AUX"):
            has_subj = any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl") for t in sent)
            if not has_subj:
                logger.debug("[Artifact Cleaner] Stripping orphaned verb sentence: %s", sent_text)
                continue

        new_sentences.append(sent_text)

    return " ".join(new_sentences)


# ===========================================================================
# Main Pipeline
# ===========================================================================
from app.core.voice import extract_voice, apply_voice


def humanize_text(text: str, field: str = "general", intensity: float = 0.7) -> Dict[str, Any]:
    """
    Main humanization pipeline with:
    - Confidence-gradient processing (light/full pipeline per segment)
    - Intensity parameter controlling pass aggressiveness
    - All 7+ passes
    """
    voice_profile = extract_voice(text)
    initial_score = detect_ai_score(text)
    words_total = word_tokenize(text)
    word_count = len(words_total)

    # ── Large text chunking (sentence-boundary aware) ──────────────────────
    if word_count > 5000:
        sentences = sent_tokenize(text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0
        for sent in sentences:
            sent_len = len(word_tokenize(sent))
            if current_len + sent_len > 1000 and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = sent_len
            else:
                current_chunk.append(sent)
                current_len += sent_len
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        chunk_results = [humanize_text(chunk, field, intensity) for chunk in chunks]
        final_text = " ".join([c["humanized_text"] for c in chunk_results])
        final_changes: Dict[str, Any] = {}
        for c in chunk_results:
            for k, v in c["changes_made"].items():
                if isinstance(v, (int, float)):
                    final_changes[k] = final_changes.get(k, 0) + v
        return {
            "humanized_text": final_text,
            "passes_applied": chunk_results[0]["passes_applied"],
            "changes_made": final_changes,
            "original_score": initial_score,
            "humanized_score": detect_ai_score(final_text),
            "voice_profile": voice_profile,
        }

    random.seed(get_seed(text))

    # ── Semantic Routing ───────────────────────────────────────────────────
    routed_field = classify_domain(text) if field == "general" else field
    profile = load_profile(routed_field)
    current_text, citations_map = protect_citations(text)

    changes: Dict[str, Any] = {
        "phrases_replaced": 0,
        "sentences_restructured": 0,
        "paragraph_rhythm_fixes": 0,
        "rhythm_adjustments": 0,
        "burstiness_injections": 0,
        "lexical_substitutions": 0,
        "style_overlays": 0,
        "voice_conversions": 0,
        "clause_reorders": 0,
        "discourse_markers": 0,
        "audit_iterations": 0,
        "voice_adjustments": 0,
        "final_cleanups": 0,
    }

    # ── Confidence-gradient processing ─────────────────────────────────────
    is_short_text = word_count < 50

    def apply_light_passes(txt: str) -> str:
        """Minimal passes for segments already scoring well."""
        c_txt = txt
        c_txt, c1 = pass_phrase_replacement(c_txt, intensity)
        changes["phrases_replaced"] += c1
        c_txt, c4 = pass_lexical(c_txt, profile, intensity)
        changes["lexical_substitutions"] += c4
        return c_txt

    def apply_full_passes(txt: str) -> str:
        """Full pipeline for segments that need heavy processing."""
        c_txt = txt
        c_txt, c1 = pass_phrase_replacement(c_txt, intensity)
        changes["phrases_replaced"] += c1
        c_txt, c2 = pass_restructuring(c_txt, intensity)
        changes["sentences_restructured"] += c2
        c_txt, c25 = pass_paragraph_rhythm(c_txt)
        changes["paragraph_rhythm_fixes"] += c25
        c_txt, c35 = pass_rhythm_sculpting(c_txt, profile, intensity)
        changes["rhythm_adjustments"] += c35
        c_txt, c3 = pass_burstiness(c_txt, profile, routed_field, intensity)
        changes["burstiness_injections"] += c3
        c_txt, c4 = pass_lexical(c_txt, profile, intensity)
        changes["lexical_substitutions"] += c4
        c_txt, c5 = pass_style_overlay(c_txt, profile, routed_field, intensity)
        changes["style_overlays"] += c5
        # New passes
        c_txt, c6 = pass_voice_conversion(c_txt, intensity)
        changes["voice_conversions"] += c6
        c_txt, c7 = pass_clause_reorder(c_txt, intensity)
        changes["clause_reorders"] += c7
        c_txt, c8 = pass_discourse_markers(c_txt, intensity)
        changes["discourse_markers"] += c8
        return c_txt

    # ── Apply passes based on segment risk ─────────────────────────────────
    if is_short_text:
        current_text = apply_light_passes(current_text)
        passes_applied = ["citation_guard", "phrase_replacement", "lexical"]
    else:
        # Segment-aware processing: score each ~250-word segment
        segments = _split_into_segments(current_text, target_words=250)
        processed_segments = []

        for seg in segments:
            seg_score = score_segment(seg)
            if seg_score < 25:
                # Safe segment — leave mostly untouched (light pass only)
                processed_segments.append(apply_light_passes(seg))
                logger.debug("[Confidence Gradient] Segment score %.1f < 25 — light pass", seg_score)
            elif seg_score < 45:
                # Moderate risk — phrase replacement + burstiness + lexical
                tmp = seg
                tmp, c1 = pass_phrase_replacement(tmp, intensity)
                changes["phrases_replaced"] += c1
                tmp, c3 = pass_burstiness(tmp, profile, routed_field, intensity)
                changes["burstiness_injections"] += c3
                tmp, c4 = pass_lexical(tmp, profile, intensity)
                changes["lexical_substitutions"] += c4
                tmp, c8 = pass_discourse_markers(tmp, intensity)
                changes["discourse_markers"] += c8
                processed_segments.append(tmp)
                logger.debug("[Confidence Gradient] Segment score %.1f — moderate pass", seg_score)
            else:
                # High risk — full pipeline
                processed_segments.append(apply_full_passes(seg))
                logger.debug("[Confidence Gradient] Segment score %.1f >= 45 — full pass", seg_score)

        current_text = " ".join(processed_segments)
        passes_applied = [
            "citation_guard", "confidence_gradient", "phrase_replacement",
            "restructuring", "paragraph_rhythm", "rhythm_sculpting",
            "burstiness", "lexical", "style_overlay",
            "voice_conversion", "clause_reorder", "discourse_markers",
        ]

    # ── Self-audit loop ────────────────────────────────────────────────────
    if not is_short_text:
        max_iterations = 3
        last_score = detect_ai_score(restore_citations(current_text, citations_map))
        for iteration in range(max_iterations):
            if last_score <= 35:
                break
            changes["audit_iterations"] += 1
            sentences_with_scores = score_sentences(current_text)
            new_sentences = []
            for s in sentences_with_scores:
                sent_text = s["text"]
                if s["score"] > 45:
                    sent_text = apply_light_passes(sent_text)
                new_sentences.append(sent_text)
            current_text = " ".join(new_sentences)
            current_restored = restore_citations(current_text, citations_map)
            new_score = detect_ai_score(current_restored)
            if new_score >= last_score:
                break
            last_score = new_score
        if changes["audit_iterations"] > 0:
            passes_applied.append("self_audit")

    # ── Voice Preservation (drift correction) ──────────────────────────────
    if voice_profile:
        current_text = apply_voice(current_text, voice_profile)
        changes["voice_adjustments"] += 1
        passes_applied.append("voice_preserved")

    # ── Final AST Cleanup ──────────────────────────────────────────────────
    original_final = current_text
    current_text = pass_final_cleanup(current_text)
    if current_text != original_final:
        changes["final_cleanups"] += 1
        passes_applied.append("artifact_cleanup")

    # ── Post-processing fixes ──────────────────────────────────────────────
    current_text = restore_citations(current_text, citations_map)
    current_text = current_text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")

    def fix_caps(m):
        return m.group(1) + m.group(2).upper()

    current_text = re.sub(r'([.!?]\s+)([a-z])', fix_caps, current_text)

    return {
        "humanized_text": current_text,
        "passes_applied": passes_applied,
        "changes_made": changes,
        "original_score": initial_score,
        "humanized_score": detect_ai_score(current_text),
        "voice_profile": voice_profile,
    }
