import random
import hashlib
import re
import os
import numpy as np
import nltk
import logging
import contextvars
import json
import spacy
from deep_translator import GoogleTranslator
from typing import List, Dict, Any, Optional, Tuple
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from lemminflect import getInflection, getLemma
from app.core.detector import (
    detect_ai_score, score_sentences, calculate_burstiness,
    calculate_perplexity_proxy, _split_into_segments, score_segment,
    AI_SIGNATURE_PHRASES, get_tree_depth
)
from app.core.voice import extract_voice, apply_voice

# Load Harvested Academic DNA
DNA_PATH = os.path.join(os.path.dirname(__file__), "academic_dna.json")
try:
    with open(DNA_PATH, "r") as f:
        ACADEMIC_DNA = json.load(f)
except Exception:
    ACADEMIC_DNA = {}

# Configure Logging
logger = logging.getLogger("humaniser.nlp")
logger.setLevel(logging.DEBUG)

rng_var = contextvars.ContextVar('rng', default=random.Random())
def get_rng():
    return rng_var.get()

# Injection Guard: Track unique phrases added per request to avoid repetition
used_phrases_var = contextvars.ContextVar('used_phrases', default=set())
def get_used_phrases():
    return used_phrases_var.get()

# Pre-load spaCy model variable to avoid latency
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None

# ===========================================================================
# AI Phrase Replacement Map (EXPANDED with variety and corpus patterns)
# ===========================================================================
AI_PHRASE_REPLACEMENTS = {
    # --- Transitions & Connectors ---
    "furthermore": ["also", "plus", "beyond that", "what's more", "besides", "moreover", "as well", "additionally"],
    "additionally": ["also", "and", "on top of this", "further", "along with this", "plus", "moreover"],
    "moreover": ["also", "and", "plus", "what's more", "besides", "in addition"],
    "it is worth noting": ["notably", "worth mentioning", "interestingly", "of note", "it's useful to see", "markedly"],
    "it is important to note": ["importantly", "worth noting", "one should note", "notably", "critically", "it's key to remember"],
    "in conclusion": ["overall", "in the end", "to close", "lastly", "finally", "to wrap up", "all in all"],
    "to summarize": ["briefly", "in short", "to recap", "in essence", "summing up"],
    "consequently": ["so", "as a result", "because of this", "thus", "hence", "therefore"],
    "therefore": ["so", "thus", "hence", "for this reason", "accordingly"],
    "it is essential": ["one must", "it matters that", "crucially", "it's vital", "it is key", "it is necessary"],
    "in order to": ["to", "so as to", "for the purpose of"],
    "due to the fact that": ["because", "since", "given that", "as"],
    "due to the fact": ["because", "since", "as"],
    "in light of": ["given", "considering", "following", "because of"],
    "has been shown": ["appears", "seems", "the evidence suggests", "it seems", "looks to be", "is widely seen"],
    "research has shown": ["studies suggest", "evidence points to", "work here indicates", "data shows", "scholars find"],
    "plays a crucial role": ["matters greatly", "is central to", "drives", "is vital", "is key", "counts for a lot"],
    "delve into": ["examine", "look at", "explore", "consider", "dig into", "investigate"],
    "it is clear that": ["clearly", "evidently", "plainly", "it's obvious", "without doubt"],
    "in today's world": ["today", "currently", "at present", "nowadays", "these days"],
    "in recent years": ["recently", "over the past few years", "lately", "of late"],
    "needless to say": ["of course", "naturally", "obviously"],
    "it goes without saying": ["of course", "naturally", "clearly"],
    "a wide range of": ["many", "various", "numerous", "a host of", "plenty of", "a broad array of"],
    "a variety of": ["many", "several", "diverse", "various", "multiple kinds of"],
    "when it comes to": ["regarding", "on", "concerning", "about", "as for"],
    "with regard to": ["on", "regarding", "about", "concerning"],
    "in terms of": ["for", "regarding", "on", "as for"],
    "the fact that": ["that", "how", "the reality that"],
    "this demonstrates": ["this shows", "this points to", "this proves", "this makes clear"],
    "this indicates": ["this points to", "this shows", "this suggests"],
    "this suggests": ["this points to", "it seems", "this might show"],
    "as previously mentioned": ["as noted", "as discussed", "earlier we saw", "as said before"],
    "it can be seen": ["we can see", "the data show", "it is evident", "one sees"],
    "has become increasingly": ["has grown more", "is now more", "is becoming"],
    "it is crucial": ["it matters", "this is key", "it's vital", "it's critical"],
    "it is necessary": ["one must", "we need to", "it's required"],
    "in the context of": ["within", "in", "under", "given the setting of"],
    "with respect to": ["on", "regarding", "about", "concerning"],
    "this highlights": ["this shows", "this reveals", "this points out"],
    "this underscores": ["this reinforces", "this confirms", "this stresses"],
    "shed light on": ["clarify", "reveal", "help explain", "show"],
    "it is imperative": ["it is critical", "one must", "it's vital"],
    "in the realm of": ["in", "within", "across", "in the field of"],
    "plays an important role": ["matters", "counts for a lot", "drives things", "is significant"],
    "there is no doubt": ["clearly", "the evidence confirms", "plainly", "certainly"],
    "as mentioned earlier": ["as noted", "earlier", "as we saw"],
    "taking everything into consideration": ["overall", "on balance", "all in all"],
    "all things considered": ["overall", "on the whole", "generally"],
    "it is undeniable": ["clearly", "certainly", "no one can deny"],
    "without a doubt": ["certainly", "definitely", "surely", "unquestionably"],
    "this emphasizes": ["this stresses", "this marks", "this points to"],
    "firstly": ["first", "to start", "initially"],
    "secondly": ["next", "second", "then"],
    "thirdly": ["finally", "third", "lastly"],
    "as well as": ["along with", "and also", "plus", "and"],
    "helps reduce": ["cuts", "lowers", "trims down", "limits", "eases"],
    "it should be noted": ["notably", "importantly", "one might note", "markedly"],
    "at the end of the day": ["ultimately", "finally", "in the end", "basically"],
    "it can be argued": ["one might argue", "it's possible to say", "arguably"],
    "one must consider": ["one should look at", "it's worth considering", "we must see"],
    "minimizes harm": ["reduces damage", "cuts down on harm", "lowers risk"],
    "crucial for development": ["key for growth", "vital for progress", "central to development"],
    "notably": ["interestingly", "markedly", "importantly", "of note"],

    # --- Corpus-specific & Phrases often found in AI output ---
    "is a subject of growing concern": ["worries many people", "has raised flags", "keeps coming up", "is a rising issue"],
    "has the potential to": ["could", "might", "may well", "can potentially"],
    "are some of the factors": ["count among the reasons", "help explain why", "matter here"],
    "is essential for": ["really matters for", "is key to", "is vital for"],
    "it is important to": ["one should", "you want to", "it's good to"],
    "is a critical concern": ["really matters", "is a pressing issue", "is a big worry"],
    "raises significant concerns": ["worries people", "causes unease", "raises flags"],
    "poses significant": ["creates real", "brings serious", "presents major"],
    "significant challenges": ["tough problems", "real hurdles", "hard parts", "major issues"],
    "significant implications": ["big consequences", "real effects", "major fallout"],
    "a key focus of": ["central to", "at the heart of", "a main part of"],
    "is a fundamental": ["is a core", "is a basic", "is a primary"],
    "is increasingly being": ["is more and more", "has started to be", "is now being"],
    "the potential for": ["the risk of", "the chance of", "the possibility of"],
    "the prevalence of": ["how common", "the spread of", "the frequency of"],
    "require a combination of": ["need a mix of", "call for several", "take a blend of"],
    "combination of": ["a mix of", "a blend of", "an array of", "a set of"],
    "in the history of": ["ever in", "historically in", "throughout the history of", "traditionally in"],
    "need to be": ["should be", "must be", "ought to be", "have to be"],
    "potential benefits of": ["possible gains from", "upsides of", "advantages of", "likely benefits of"],
    "robust security measures": ["strong protections", "solid defenses", "tough security"],
    "the rise of": ["the growth of", "the surge in", "the emergence of"],
    "the development of": ["building", "creating", "progress on", "making"],
    "a growing concern": ["an increasing worry", "something more people notice", "a rising problem"],
    "the use of": ["using", "relying on", "employment of"],
    "the impact of": ["how much", "the effect of", "the weight of"],
    "has resulted in": ["led to", "caused", "brought about", "triggered"],
    "addressing these challenges": ["tackling these issues", "dealing with this", "solving these problems"],
    "however, there are": ["but", "still", "yet", "even so, there are"],
    "nevertheless, there are": ["but", "still", "even so", "all the same"],
    "however, the": ["but the", "still, the", "yet the"],
    "nevertheless, the": ["but the", "still, the"],
    "is expected to": ["will likely", "should", "looks set to", "is predicted to"],
    "the advancement of": ["progress in", "gains in", "advances in"],
    "the challenges of": ["the hard parts of", "difficulties with", "issues with"],
    "in various aspects of": ["across different parts of", "in many areas of"],
    "contributing to these": ["feeding into these", "adding to these", "helping cause these"],
    "promoting the use of": ["pushing for", "encouraging", "backing"],
    "ensuring the well-being": ["looking after", "safeguarding", "protecting"],
    "the protection and promotion": ["defending and advancing", "safeguarding and pushing"],
    "one of the most": ["among the biggest", "quite possibly the", "a major"],
    "it has become increasingly": ["it now seems more", "people have started to", "it's more and more"],
    "is the process of": ["means", "involves", "refers to", "is about"],
    "in a way that": ["so that", "allowing", "whereby"],
    "play an important role": ["matter a lot", "count for something", "drive change", "are significant"],
    "has become essential": ["now matters more than ever", "is now vital", "is now a must"],
    "not only ... but also": ["not just ... but", "both ... and", "as well as"],
    "by contrast": ["on the other hand", "in contrast", "conversely"],
    "in a similar fashion": ["similarly", "likewise", "in the same way"],
    "it is evident that": ["clearly,", "it's obvious that", "plainly,"],
    "there is a need to": ["we should", "one must", "it's time to"],
    "highly effective": ["very good", "quite successful", "really works"],
    "a wide array of": ["many", "lots of", "various", "a broad range of"],
}

COMPILED_AI_PHRASES = []
for phrase in sorted(AI_PHRASE_REPLACEMENTS.keys(), key=len, reverse=True):
    pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
    COMPILED_AI_PHRASES.append((pattern, AI_PHRASE_REPLACEMENTS[phrase]))

# Map spaCy POS to WordNet POS
SPACY_TO_WORDNET = {
    "NOUN": wordnet.NOUN,
    "VERB": wordnet.VERB,
    "ADJ": wordnet.ADJ,
    "ADV": wordnet.ADV,
}

# Discourse markers for injecting unpredictability
DISCOURSE_MARKERS = [
    "as one might expect,", "interestingly enough,", "to some extent,",
    "in a sense,", "broadly speaking,", "at first glance,",
    "on closer inspection,", "to put it differently,", "in practical terms,",
    "from a broader perspective,", "curiously,", "in retrospect,",
    "turns out,", "thing is,", "point being,",
]

# Determiner alternatives for breaking "The X" patterns
DETERMINER_ALTERNATIVES = {
    "the": ["this", "that", "one particular", "a given", "each"],
    "these": ["such", "those particular", "the mentioned"],
    "this": ["that", "the", "one such"],
    "many": ["a handful of", "quite a few", "several", "numerous"],
    "various": ["different", "assorted", "multiple"],
}

# Transition word alternatives
TRANSITION_ALTERNATIVES = {
    "however": ["but then", "still", "that said", "yet"],
    "nevertheless": ["still", "even so", "all the same"],
    "furthermore": ["also", "plus", "and"],
    "moreover": ["on top of that", "also", "and"],
    "therefore": ["so", "thus", "for that reason"],
    "consequently": ["so", "as a result"],
    "additionally": ["also", "and", "plus"],
    "nonetheless": ["still", "even so"],
}

MODERN_ACADEMIC_JARGON = [
    "granularity", "robustness", "paradigm shift", "leveraging", 
    "contextualization", "methodological rigor", "nuanced", "multidimensional",
    "transformative", "synergistic", "empirical evidence", "theoretical framework",
    "optimization", "scalability", "interdisciplinary", "holistic",
    "ontology", "epistemological", "discursive", "intersectionality",
    "agential", "post-structural", "phenomenological", "heuristic", "reification",
    "interoperability", "modality", "synchronicity", "performativity", "normativity",
    "teleological", "hermeneutical", "axiomatic", "juxtaposition", "reductive",
    "hegemonic", "paradoxical", "recapitulate", "dichotomy", "divergence"
]

JARGON_REPLACEMENT_MAP = {
    "detail": "granularity",
    "precision": "granularity",
    "strength": "robustness",
    "stability": "robustness",
    "change": "paradigm shift",
    "using": "leveraging",
    "use": "leverage",
    "rigor": "methodological rigor",
    "complex": "multidimensional",
    "complexity": "multidimensionality",
    "big": "transformative",
    "complete": "holistic",
    "method": "methodology",
    "methods": "methodological approaches",
    "evidence": "empirical evidence",
    "framework": "theoretical framework",
    "result": "empirical outcome",
    "study": "investigation",
    "connection": "interoperability",
    "interface": "interoperability",
    "part": "componentry",
    "show": "demonstrate",
    "shows": "demonstrates",
    "think": "conceptualize",
    "thinks": "conceptualizes",
    "problem": "dilemma",
    "problems": "dilemmas",
    "good": "optimal",
    "bad": "suboptimal",
    "different": "divergent",
    "similar": "analogous",
    "common": "prevalent",
    "rare": "infrequent",
    "start": "initiate",
    "end": "terminate",
    "help": "facilitate",
}

# ===========================================================================
# Utility functions
# ===========================================================================

def load_spacy():
    global nlp
    if nlp is None:
        logger.error("[SpaCy] Model not pre-loaded or missing.")
    return nlp

DEEPENING_PHRASES = [
    ", which implies that the situation, as observed, is multidimensional",
    ", an observation that, while seemingly minor, complicates the framework",
    ", precisely because the underlying modality, though complex, remains relevant",
    ", suggesting that the inherent complexity, as it stands, warrants further analysis",
    ", primarily because the contextual factors, although varied, play a decisive role",
    ", which further indicates that the conceptual structure, as discussed, is quite intricate",
    ", largely because the multifaceted nature of the issue, though subtle, is significant",
    ", specifically because the foundational elements, while diverse, are interconnected"
]

def safe_split_sentence(sent_doc):
    """Splits a sentence doc into two parts at a safe point to avoid artifacts like 'Artificial. Intelligence'."""
    if not sent_doc or len(sent_doc) < 6:
        return sent_doc.text if sent_doc else "", ""
    
    # Preferred split points: at a comma or a conjunction
    best_idx = -1
    for token in sent_doc:
        idx = token.i - sent_doc[0].i
        if 2 < idx < len(sent_doc) - 3:
            if token.text == ",":
                best_idx = idx + 1
                break
            if token.pos_ == "CCONJ":
                best_idx = idx
                break
                
    if best_idx == -1:
        # Fallback: around the middle, but avoiding sensitive boundaries
        mid = len(sent_doc) // 2
        best_idx = mid
        for i in range(mid, min(mid + 3, len(sent_doc) - 2)):
            t1, t2 = sent_doc[i], sent_doc[i+1]
            # Don't split between ADJ-NOUN, NOUN-NOUN, or DET-NOUN
            if t1.pos_ == "ADJ" and t2.pos_ in ("NOUN", "PROPN"): continue
            if t1.pos_ in ("NOUN", "PROPN") and t2.pos_ in ("NOUN", "PROPN"): continue
            if t1.dep_ == "det": continue
            best_idx = i + 1
            break

    part1 = "".join([t.text_with_ws for t in sent_doc[:best_idx]]).strip().rstrip(",; ") + "."
    part2 = "".join([t.text_with_ws for t in sent_doc[best_idx:]]).strip()
    if part2:
        part2 = re.sub(r'^(and|but|or|yet|so)\s+', '', part2, flags=re.IGNORECASE)
        part2 = part2[0].upper() + part2[1:] if part2 else ""
    return part1, part2

def get_seed(text: str) -> int:
    return int(hashlib.md5(text.encode()).hexdigest(), 16) % (10**8)

def is_valid_sentence(tokens: List[Any]) -> bool:
    """Sanity Check: Ensure sentence has a VERB/AUX and a subject attached properly."""
    if not tokens: return False
    if len(tokens) < 3: return False
    
    has_verb = False
    has_subj = False
    for t in tokens:
        if t.pos_ in ("VERB", "AUX"):
            if t.dep_ not in ("amod", "compound"):
                has_verb = True
        if t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl"):
            has_subj = True
        if t.tag_ == "VBG" and t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
            has_subj = True
            
    return has_verb and has_subj

def extract_primary_subject(text: str) -> str:
    sp = load_spacy()
    if not sp:
        return "the subject matter"
    doc = sp(text)
    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass") and token.head.dep_ == "ROOT":
            subtree = [t.text for t in token.subtree if t.i <= token.i]
            return " ".join(subtree).lower()
    return "this system"

def fix_caps(m):
    return m.group(1) + m.group(2).upper()

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
# Pass functions
# ===========================================================================

def pass_back_translation(text: str) -> Tuple[str, int]:
    """Back-translation pre-processor: EN -> JA -> AR -> EN with jargon protection."""
    jargon_pattern = re.compile(r'\b[A-Z0-9\-]{3,}\b|\b[a-z]{15,}\b')
    placeholders = {}
    
    def replace_jargon(match):
        token = f"XJARGON{len(placeholders)}X"
        placeholders[token] = match.group(0)
        return token
    
    text_with_placeholders = jargon_pattern.sub(replace_jargon, text)
    
    try:
        # EN -> JA
        ja_text = GoogleTranslator(source='en', target='ja').translate(text_with_placeholders)
        # JA -> AR
        ar_text = GoogleTranslator(source='ja', target='ar').translate(ja_text)
        # AR -> EN
        en_text = GoogleTranslator(source='ar', target='en').translate(ar_text)
        
        # Restore placeholders
        for token, original in placeholders.items():
            en_text = en_text.replace(token, original)
        return en_text, 1
    except Exception as e:
        logger.error(f"Back-translation failed: {e}")
        return text, 0

def pass_morphological_shifting(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Identify high-frequency verbs and convert them to noun phrases."""
    sp = load_spacy()
    if not sp: return text, 0
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

def pass_human_wordiness(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Inject 'padding' qualifiers into concise sentences."""
    sentences = sent_tokenize(text)
    new_sentences, changes = [], 0
    padding = [
        "as it happens,", "for all intents and purposes,", "in a very real sense,",
        "to be perfectly honest,", "as far as the data suggests,", "essentially,",
        "by and large,", "more or less,", "in some ways,"
    ]
    
    for sent in sentences:
        chance = 1.0 if intensity >= 1.0 else (0.5 * intensity)
        if len(sent.split()) < 12 and get_rng().random() < chance:
            pad = get_rng().choice(padding)
            sent = pad.capitalize() + " " + sent[0].lower() + sent[1:]
            changes += 1
        new_sentences.append(sent)
    return " ".join(new_sentences), changes

def pass_yoda_inversion(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Move prepositional phrases or objects to the front."""
    sp = load_spacy()
    if not sp: return text, 0
    doc = sp(text)
    new_sentences, changes = [], 0
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        chance = 1.0 if intensity >= 1.0 else (0.2 * intensity)
        if len(sent) > 8 and get_rng().random() < chance:
            # Look for prepositional phrases at the end
            pp = None
            for token in reversed(sent):
                if token.dep_ == "prep" and token.head.dep_ == "ROOT":
                    pp = token
                    break
            if pp and pp.i > len(sent) // 2:
                pp_text = "".join([t.text_with_ws for t in pp.subtree]).strip().rstrip(".,")
                main_part = "".join([t.text_with_ws for t in sent if t.i < pp.i]).strip().rstrip(",")
                if main_part:
                    sent_text = f"{pp_text.capitalize()}, {main_part[0].lower() + main_part[1:]}."
                    changes += 1
        new_sentences.append(sent_text)
    return " ".join(new_sentences), changes

def pass_appositive_injection(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Insert descriptive comma-clauses into the middle of sentences."""
    sp = load_spacy()
    if not sp: return text, 0
    doc = sp(text)
    new_sentences, changes = [], 0
    appositives = [
        ", a point that bears repeating,", ", which is often overlooked,",
        ", as many scholars have noted,", ", a detail of significant importance,",
        ", which complicates matters slightly,", ", in a manner of speaking,"
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

def pass_zwj_jitter(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Inject \u200C (ZWNJ) inside words that are AI triggers."""
    triggers = ["furthermore", "moreover", "additionally", "consequently", "therefore", "essential", "crucial"]
    new_text, changes = text, 0
    for t in triggers:
        if t in new_text.lower() and get_rng().random() < intensity:
            # Inject ZWNJ in the middle
            mid = len(t) // 2
            jittered = t[:mid] + "\u200C" + t[mid:]
            new_text = re.sub(rf"\b{t}\b", jittered, new_text, flags=re.IGNORECASE)
            changes += 1
    return new_text, changes

def pass_invisible_padding(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Use Hair Spaces (U+200A) between words in common bigrams."""
    common_bigrams = [("of", "the"), ("in", "the"), ("to", "the"), ("on", "the"), ("and", "the")]
    new_text, changes = text, 0
    for b1, b2 in common_bigrams:
        pattern = re.compile(rf"\b{b1}\s+{b2}\b", re.IGNORECASE)
        if get_rng().random() < intensity:
            new_text = pattern.sub(f"{b1}\u200A{b2}", new_text)
            changes += 1
    return new_text, changes

def pass_punctuation_personality(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Replace some commas with em-dashes or semicolons."""
    new_sentences, changes = [], 0
    for sent in sent_tokenize(text):
        chance = 1.0 if intensity >= 1.0 else (0.3 * intensity)
        if "," in sent and get_rng().random() < chance:
            choices = [" — ", "; "]
            sent = sent.replace(",", get_rng().choice(choices), 1)
            changes += 1
        new_sentences.append(sent)
    return " ".join(new_sentences), changes

def pass_cross_referencing(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Inject phrases like 'consistent with the earlier points...'."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs, changes = [], 0
    markers = ["as previously noted,", "consistent with earlier points,", "as we saw before,", "bearing in mind the previous discussion,"]
    
    for i, p in enumerate(paragraphs):
        sentences = sent_tokenize(p)
        chance = 1.0 if intensity >= 1.0 else (0.4 * intensity)
        if i > 0 and sentences and get_rng().random() < chance:
            m = get_rng().choice(markers)
            sentences[0] = m.capitalize() + " " + sentences[0][0].lower() + sentences[0][1:]
            changes += 1
        new_paragraphs.append(" ".join(sentences))
    return "\n\n".join(new_paragraphs), changes

def pass_hedging_uncertainty(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Add uncertainty markers to sentences."""
    sentences = sent_tokenize(text)
    new_sentences, changes = [], 0
    hedges = ["it seems that", "perhaps", "potentially,", "arguably,", "one might suggest that"]
    
    for sent in sentences:
        chance = 1.0 if intensity >= 1.0 else (0.5 * intensity)
        if get_rng().random() < chance:
            h = get_rng().choice(hedges)
            sent = h.capitalize() + " " + sent[0].lower() + sent[1:]
            changes += 1
        new_sentences.append(sent)
    return " ".join(new_sentences), changes

def pass_signature_phrase_breaker(text: str) -> Tuple[str, int]:
    """
    Force-replaces every single occurrence of phrases found in AI_SIGNATURE_PHRASES.
    """
    count = 0
    new_text = text
    sorted_phrases = sorted(AI_SIGNATURE_PHRASES, key=len, reverse=True)
    
    for phrase in sorted_phrases:
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        
        def replace_fn(match):
            nonlocal count
            original = match.group(0)
            replacements = AI_PHRASE_REPLACEMENTS.get(phrase.lower(), ["also", "and", "in fact", "actually", "notably"])
            replacement = get_rng().choice(replacements)
            count += 1
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        new_text = pattern.sub(replace_fn, new_text)
        
    return new_text, count

def pass_phrase_replacement(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    count = 0
    new_text = text
    used = get_used_phrases()
    force_all = intensity >= 0.8

    for pattern, replacements in COMPILED_AI_PHRASES:
        if not force_all and get_rng().random() > intensity:
            continue

        def replace_fn(match):
            nonlocal count
            original = match.group(0)
            available = [r for r in replacements if r.lower() not in used]
            if not available:
                available = replacements
            replacement = get_rng().choice(available)
            used.add(replacement.lower())
            count += 1
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        new_text = pattern.sub(replace_fn, new_text)

    return new_text, count

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

        if word_count > 18 and get_rng().random() < intensity:
            split_token = None
            for token in sent:
                if token.pos_ == "CCONJ" and token.head.dep_ == "ROOT":
                    is_list = False
                    for prev in reversed(sent[:token.i - sent.start]):
                        if prev.text.lower() in ("such", "including", "like"): is_list = True; break
                        if prev.pos_ == "PUNCT" or prev.i < token.i - 5: break
                    if is_list: continue
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

def pass_burstiness(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs = []
    total_changes = 0
    used = get_used_phrases()

    for paragraph in paragraphs:
        doc = sp(paragraph)
        sentences = list(doc.sents)
        if not sentences:
            processed_paragraphs.append(paragraph)
            continue

        p_words = word_tokenize(paragraph)
        p_word_count = len(p_words)
        lengths = [len(word_tokenize(s.text)) for s in sentences]
        has_very_short = any(l < 5 for l in lengths)
        has_very_long = any(l > 35 for l in lengths)
        new_sentences = [s.text.strip() for s in sentences]
        changes = 0

        if p_word_count > 60:
            if not has_very_short:
                for i, s_text in enumerate(new_sentences):
                    s_doc = sp(s_text)
                    if len(s_doc) > 10:
                        p1, p2 = safe_split_sentence(s_doc)
                        if p2:
                            new_sentences[i] = p1
                            new_sentences.insert(i + 1, p2)
                            changes += 1
                            has_very_short = True
                            break
            
            if not has_very_long and len(new_sentences) >= 2:
                i = 0
                while i < len(new_sentences) - 1:
                    s1, s2 = new_sentences[i], new_sentences[i+1]
                    s1_words = len(word_tokenize(s1))
                    s2_words = len(word_tokenize(s2))
                    if s1_words + s2_words > 15:
                        connector = get_rng().choice([", and furthermore, ", ", which indicates that ", "; moreover, ", ", and this suggests that "])
                        merged = s1.rstrip(".!?") + connector + s2[0].lower() + s2[1:]
                        new_sentences[i] = merged
                        new_sentences.pop(i+1)
                        changes += 1
                        if len(word_tokenize(merged)) > 35:
                            has_very_long = True
                            break
                    else:
                        i += 1

        current_sentences = []
        i = 0
        while i < len(new_sentences):
            s_text = new_sentences[i]
            s_len = len(s_text.split())
            if 3 < s_len < 10 and get_rng().random() < 0.5 * intensity:
                qualifiers = [
                    " — at least in the cases examined here",
                    ", though this varies by context",
                    ", a finding consistent with earlier work",
                    ", which remains a point of debate",
                    ", something researchers keep exploring",
                    " — an observation that merits further study",
                ]
                available = [q for q in qualifiers if q.lower() not in used]
                if not available: available = qualifiers
                q = get_rng().choice(available)
                used.add(q.lower())
                s_text = s_text.rstrip('.!?') + q + "."
                changes += 1
            current_sentences.append(s_text)
            i += 1
        
        processed_paragraphs.append(" ".join(current_sentences))
        total_changes += changes

    return "\n\n".join(processed_paragraphs), total_changes

def pass_rhythm_sculpting(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0

    dna_stats = ACADEMIC_DNA.get("sentence_stats", {})
    target_std = dna_stats.get("std_dev_length", 9)
    doc = sp(text)
    sentences = list(doc.sents)
    if len(sentences) < 5:
        return text, 0
    lengths = [len(s) for s in sentences]
    current_std = float(np.std(lengths))
    changes = 0
    new_sentences = []

    for s in sentences:
        s_len = len(s)
        if current_std < (target_std - 3) and s_len > 25 and get_rng().random() < intensity:
            split_token = next((t for t in s if t.pos_ == "CCONJ" and t.head.dep_ == "ROOT"), None)
            if split_token:
                is_list = False
                for prev in reversed(s[:split_token.i - s.start]):
                    if prev.text.lower() in ("such", "including", "like"): is_list = True; break
                    if prev.pos_ == "PUNCT" or prev.i < split_token.i - 5: break
                if is_list:
                    new_sentences.append(s.text.strip())
                    continue

                p1, p2 = [t for t in s if t.i < split_token.i], [t for t in s if t.i >= split_token.i]
                if is_valid_sentence(p1) and is_valid_sentence(p2):
                    new_sentences.append("".join([t.text_with_ws for t in p1]).strip().rstrip(",; ") + ".")
                    p2_text = "".join([t.text_with_ws for t in p2]).strip()
                    p2_text = re.sub(r'^(and|but|or|yet|so)\s+', '', p2_text, flags=re.IGNORECASE)
                    if p2_text:
                        new_sentences.append(p2_text[0].upper() + p2_text[1:])
                    changes += 1
                    continue
        new_sentences.append(s.text.strip())
    return " ".join(new_sentences), changes

def pass_lexical(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0

    doc = sp(text)
    new_tokens = []
    changes = 0
    avoid_words = set(profile.get("top_vocab", []) if profile else [])
    replace_chance = 0.55 * intensity

    for token in doc:
        if token.pos_ not in SPACY_TO_WORDNET or not token.is_alpha or token.text.lower() in avoid_words:
            new_tokens.append(token.text_with_ws)
            continue

        if token.pos_ in ("NOUN", "VERB", "ADJ") and get_rng().random() < 0.1 * intensity:
            t_lower = token.text.lower()
            if t_lower in JARGON_REPLACEMENT_MAP:
                replacement = JARGON_REPLACEMENT_MAP[t_lower]
                if token.text[0].isupper(): replacement = replacement.capitalize()
                new_tokens.append(replacement + token.whitespace_)
                changes += 1
                continue
            elif get_rng().random() < 0.05:
                jargon = get_rng().choice(MODERN_ACADEMIC_JARGON)
                new_tokens.append(jargon + " " + token.text_with_ws)
                changes += 1
                continue

        wn_pos = SPACY_TO_WORDNET[token.pos_]
        if get_rng().random() < replace_chance:
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
                        top_n = min(5, len(valid_synonyms))
                        replacement_lemma = get_rng().choice(valid_synonyms[:top_n])[0]
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

def pass_style_overlay(text: str, profile: Optional[Dict[str, Any]], intensity: float = 1.0) -> Tuple[str, int]:
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
            if s_idx == 0 and opening_patterns and get_rng().random() < 0.3 * intensity:
                pattern = get_rng().choice(opening_patterns)
                pattern_words = word_tokenize(pattern)[:4]
                sent = " ".join(pattern_words) + ", " + sent[0].lower() + sent[1:]
                changes += 1
            if hedging and s_idx % 5 == 0 and s_idx > 0 and get_rng().random() < 0.4 * intensity:
                phrase = get_rng().choice(hedging)
                sent = phrase.capitalize() + ", " + sent[0].lower() + sent[1:]
                changes += 1
            new_sentences.append(sent)
        new_paragraphs.append(" ".join(new_sentences))

    return "\n\n".join(new_paragraphs), changes

def pass_voice_conversion(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    sp = load_spacy()
    if not sp:
        return text, 0

    doc = sp(text)
    new_sentences = []
    changes = 0

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if get_rng().random() > 0.20 * intensity or len(sent) < 5:
            new_sentences.append(sent_text)
            continue

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
            subj_text = " ".join([t.text for t in nsubj.subtree])
            obj_text = " ".join([t.text for t in dobj.subtree])
            verb_lemma = root.lemma_
            pp = getInflection(verb_lemma, tag="VBN")
            if pp:
                passive = f"{obj_text.capitalize()} was {pp[0]} by {subj_text.lower()}."
                new_sentences.append(passive)
                changes += 1
                continue

        new_sentences.append(sent_text)

    return " ".join(new_sentences), changes

def pass_clause_reorder(text: str, intensity: float = 1.0) -> Tuple[str, int]:
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
                if comma_idx > 0 and comma_idx < len(sent) - 5:
                    clause1 = sent[:comma_idx].strip()
                    clause2 = sent[comma_idx + 1:].strip()
                    if clause2:
                        new_sent = clause2[0].upper() + clause2[1:].rstrip('.') + " " + clause1[0].lower() + clause1[1:] + "."
                        new_sentences.append(new_sent)
                        changes += 1
                        reordered = True
                        break
        if not reordered:
            new_sentences.append(sent)

    return " ".join(new_sentences), changes

def pass_discourse_markers(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0

    for i, sent in enumerate(sentences):
        if i > 0 and i % 3 == 0 and get_rng().random() < 0.35 * intensity:
            marker = get_rng().choice(DISCOURSE_MARKERS)
            sent = marker + " " + sent[0].lower() + sent[1:]
            changes += 1
        new_sentences.append(sent)

    return " ".join(new_sentences), changes

def pass_nuance_injection(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Injects human-like nuance and subtle hedging to disrupt AI's over-confident tone.
    """
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0
    used = get_used_phrases()
    nuances = [
        "one might argue,", "it seems that", "arguably,", 
        "to some extent,", "in a sense,", "potentially,",
        "it appears that", "broadly speaking,", "in many respects,",
        "one could suggest that", "seemingly,", "ostensibly,",
        "it would appear that", "for the most part,", "it is possible that",
        "in light of this,", "one may conclude that", "conceivably,"
    ]
    
    for i, sent in enumerate(sentences):
        if i % 4 == 0 and get_rng().random() < 0.4 * intensity and len(sent.split()) > 10:
            available = [n for n in nuances if n.lower() not in used]
            if not available: available = nuances
            nuance = get_rng().choice(available)
            used.add(nuance.lower())
            prefix = nuance.capitalize()
            sent = prefix + " " + sent[0].lower() + sent[1:]
            changes += 1
        new_sentences.append(sent)
        
    return " ".join(new_sentences), changes

def pass_determiner_scramble(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Scrambles "The X" openers and varies transition words.
    """
    sentences = sent_tokenize(text)
    new_sentences = []
    changes = 0

    for i, sent in enumerate(sentences):
        words = sent.split()
        if not words:
            new_sentences.append(sent)
            continue
        first_word = words[0].lower().rstrip(",;:")
        modified = False
        if first_word == "the" and len(words) > 3 and get_rng().random() < 0.35 * intensity:
            alts = DETERMINER_ALTERNATIVES.get("the", [])
            replacement = get_rng().choice(alts)
            words[0] = replacement.capitalize() if words[0][0].isupper() else replacement
            modified = True
            changes += 1
        if first_word in TRANSITION_ALTERNATIVES and get_rng().random() < 0.5 * intensity:
            alts = TRANSITION_ALTERNATIVES[first_word]
            replacement = get_rng().choice(alts)
            had_comma = words[0].endswith(",")
            words[0] = replacement.capitalize() + ("," if had_comma and not replacement.endswith(",") else "")
            modified = True
            changes += 1
        if modified:
            new_sentences.append(" ".join(words))
        else:
            new_sentences.append(sent)

    return " ".join(new_sentences), changes

def pass_syntactic_variance(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Targets the detector's 'Syntactic Variance' metric.
    Hard-enforce depth extremes in every paragraph: one < 4, one > 12.
    """
    sp = load_spacy()
    if not sp:
        return text, 0

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs = []
    total_changes = 0
    parentheticals = [
        " — perhaps unsurprisingly —", " — at least for now —",
        " — and this is key —", " — to be fair —",
        " — one could argue —", " — in a practical sense —",
        " (as noted by some) ", " — naturally — ",
        " — so to speak — ", " — in effect — ",
        " — for better or worse — ", " — as it were — ",
        " — arguably — ", " — in many respects — ",
        " — for the most part — ", " — if you will — ",
        " — to some extent — ", " — quite literally — ",
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

        if not has_flat:
            # Skip the first sentence to avoid weird fragments at the start of paragraphs
            for i, s_text in enumerate(new_sentences):
                if i == 0: continue
                s_words = s_text.split()
                if 4 < len(s_words) < 15:
                    new_sentences[i] = " ".join(s_words[:min(5, len(s_words))]).rstrip(",; ") + "."
                    changes += 1
                    has_flat = True
                    break
        if not has_deep:
            for i, s_text in enumerate(new_sentences):
                if len(s_text.split()) > 10:
                    s_doc = sp(s_text)
                    root = next((t for t in s_doc if t.dep_ == "ROOT"), None)
                    if root:
                        extra = get_rng().choice([
                            ", which implies that the situation, as observed, is multidimensional",
                            ", an observation that, while seemingly minor, complicates the framework",
                            ", precisely because the underlying modality, though complex, remains relevant"
                        ])
                        new_sentences[i] = s_text.rstrip(".!?") + extra + "."
                        changes += 1
                        has_deep = True
                        break

        final_paragraph_sentences = []
        for s_text in new_sentences:
            s_doc = sp(s_text)
            sent_text = s_text
            if 6 < len(s_doc) < 20 and get_rng().random() < (0.3 * intensity):
                root = next((t for t in s_doc if t.dep_ == "ROOT" and t.lemma_ == "be"), None)
                if root:
                    nsubj = next((t for t in root.children if t.dep_ == "nsubj"), None)
                    attr = next((t for t in root.children if t.dep_ in ("attr", "acomp")), None)
                    if nsubj and attr:
                        subj_text = "".join([t.text_with_ws for t in nsubj.subtree]).strip()
                        attr_text = "".join([t.text_with_ws for t in attr.subtree]).strip()
                        remaining_tokens = [t for t in s_doc if t.i > max(nsubj.i, attr.i, root.i)]
                        remaining = "".join([t.text_with_ws for t in remaining_tokens]).strip()
                        new_sent = f"Being {attr_text.lower()}, {subj_text} {root.text} {remaining}"
                        new_sent = new_sent.strip().rstrip(".,; ") + "."
                        sent_text = new_sent[0].upper() + new_sent[1:]
                        changes += 1
            if len(s_doc) > 12 and get_rng().random() < (0.35 * intensity):
                root = next((t for t in s_doc if t.dep_ == "ROOT"), None)
                if root and root.pos_ == "VERB":
                    subj = next((t for t in root.children if "subj" in t.dep_), None)
                    if subj:
                        p_text = get_rng().choice(parentheticals)
                        subj_end_idx = subj.i - s_doc[0].i + 1
                        if 0 < subj_end_idx < len(s_doc):
                            part1 = "".join([t.text_with_ws for t in s_doc[:subj_end_idx]]).strip()
                            part2 = "".join([t.text_with_ws for t in s_doc[subj_end_idx:]]).strip()
                            sent_text = f"{part1}{p_text} {part2}"
                            changes += 1
            final_paragraph_sentences.append(sent_text)
        processed_paragraphs.append(" ".join(final_paragraph_sentences))
        total_changes += changes

    return "\n\n".join(processed_paragraphs), total_changes

def pass_structural_reset(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Aggressive structural deconstruction with list-avoidance.
    Ensures that splits result in two valid sentences and skips paragraph-starts.
    """
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
        
        # Skip the first sentence to avoid weird fragments at the start of paragraphs
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
                    
                    # Prevent splitting after a verb or determiner that would leave a fragment
                    if token.i > sent.start and sent[token.i - sent.start - 1].pos_ in ("VERB", "AUX", "ADP", "DET"):
                        continue

                    is_list = False
                    if token.i > sent.start and sent[token.i - sent.start - 1].text.lower() in ("such", "including", "like", "as"): is_list = True
                    if sent_text.count(",") > 2: is_list = True
                    if is_list: continue
                    
                    idx_off = token.i - sent.start
                    part1_t, part2_t = [t for t in sent[:idx_off]], [t for t in sent[idx_off:]]
                    if is_valid_sentence(part1_t) and is_valid_sentence(part2_t):
                        p1 = "".join([t.text_with_ws for t in part1_t]).strip().rstrip(",; ") + "."
                        p2 = "".join([t.text_with_ws for t in part2_t]).strip()
                        p2 = re.sub(r'^[,;\s]+', '', p2)
                        if p2:
                            p2_lower = p2.lower()
                            if p2_lower.startswith("which "): p2 = "This " + p2[6:]
                            elif p2_lower.startswith("who "): p2 = "They " + p2[4:]
                            elif p2_lower.startswith("where "): p2 = "There, " + p2[6:]
                            p2 = p2[0].upper() + p2[1:]; p2 = p2.rstrip(".") + "."
                            new_sentences.append(p1); new_sentences.append(p2)
                            changes += 1; split_done = True; break
            if not split_done:
                for token in sent:
                    if (token.pos_ == "CCONJ" and token.text.lower() in ("and", "but", "yet", "so") and
                        7 < (token.i - sent.start) < s_len - 5):
                        
                        # Grammar Guard: Don't split right after a verb
                        if token.i > sent.start and sent[token.i - sent.start - 1].pos_ in ("VERB", "AUX"):
                            continue

                        is_list = False
                        for prev in reversed(sent[:token.i - sent.start]):
                            if prev.text.lower() in ("such", "including", "like", "as"): is_list = True; break
                            if prev.pos_ == "PUNCT" or prev.i < token.i - 5: break
                        if sent_text.count(",") > 2: is_list = True
                        if is_list: continue
                        idx_off = token.i - sent.start
                        part1_t, part2_t = [t for t in sent[:idx_off]], [t for t in sent[idx_off+1:]]
                        if is_valid_sentence(part1_t) and is_valid_sentence(part2_t):
                            p1 = "".join([t.text_with_ws for t in part1_t]).strip().rstrip(",; ") + "."
                            p2 = "".join([t.text_with_ws for t in part2_t]).strip()
                            p2 = re.sub(r'^[,;\s]+', '', p2)
                            if p2 and len(p1.split()) > 4 and len(p2.split()) > 4:
                                p2 = p2[0].upper() + p2[1:]
                                p2 = re.sub(r'^(and|but|or|yet|so)\s+', '', p2, flags=re.IGNORECASE)
                                p2 = p2[0].upper() + p2[1:]; p2 = p2.rstrip(".") + "."
                                new_sentences.append(p1); new_sentences.append(p2)
                                changes += 1; split_done = True; break
            if split_done:
                i += 1
                continue
        if s_len < 10 and i < len(sentences) - 1:
            next_sent = sentences[i + 1]
            next_sent_text = next_sent.text.strip()
            if len(next_sent) < 12 and sent_text and next_sent_text and get_rng().random() < 0.6 * intensity:
                connector = get_rng().choice([", and ", " — ", "; in fact, ", ", which means "])
                merged = sent_text.rstrip('.!?') + connector + next_sent_text[0].lower() + next_sent_text[1:]
                new_sentences.append(merged); changes += 1; i += 2; continue
        new_sentences.append(sent_text); i += 1

    return " ".join(new_sentences), changes

def pass_rhetorical_questions(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """
    Occasionally turn a declarative sentence into a rhetorical question followed by an answer.
    """
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_sentences = []
    changes = 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if get_rng().random() < (0.10 * intensity) and len(sent) > 12:
            root = next((t for t in sent if t.dep_ == "ROOT"), None)
            if root and root.lemma_ in ("show", "indicate", "suggest", "demonstrate", "reveal", "illustrate", "provide"):
                subj = next((t for t in root.children if "subj" in t.dep_), None)
                if subj:
                    subj_text = "".join([t.text_with_ws for t in subj.subtree]).strip()
                    aux = "does" if subj.tag_ in ("NNP", "NN", "PRP") and subj.text.lower() not in ("i", "you", "we", "they") else "do"
                    question = f"What {aux} {subj_text.lower()} {root.lemma_}?"
                    pronoun = "They" if subj.tag_ in ("NNS", "NNPS") else "It"
                    predicate_tokens = [t for t in sent if t.i >= root.i]
                    predicate_text = "".join([t.text_with_ws for t in predicate_tokens]).strip()
                    new_sent = f"{question} {pronoun} {predicate_text}"
                    new_sentences.append(new_sent); changes += 1; continue
        new_sentences.append(sent_text)
    return " ".join(new_sentences), changes

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

def pass_jargon_injection(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Injects modern academic jargon."""
    sp = load_spacy()
    if not sp:
        return text, 0
    doc = sp(text)
    new_tokens, changes = [], 0
    for token in doc:
        t_lower = token.text.lower()
        if t_lower in JARGON_REPLACEMENT_MAP and get_rng().random() < 0.4 * intensity:
            replacement = JARGON_REPLACEMENT_MAP[t_lower]
            if token.text[0].isupper(): replacement = replacement.capitalize()
            new_tokens.append(replacement + token.whitespace_); changes += 1
        elif token.pos_ == "ADJ" and get_rng().random() < 0.05 * intensity:
            injection = "nuanced" if get_rng().random() < 0.5 else "thoroughly contextualized"
            replacement = injection + " " + token.text
            if token.text[0].isupper(): replacement = replacement.capitalize()
            new_tokens.append(replacement + token.whitespace_); changes += 1
        else:
            new_tokens.append(token.text_with_ws)
    return "".join(new_tokens), changes

def pass_structural_chaos(text: str) -> Tuple[str, int]:
    """Randomly reorders sentences that don't have strong temporal markers."""
    sp = load_spacy()
    if not sp: return text, 0
    temporal_markers = {
        "first", "second", "third", "then", "finally", "next", "lastly",
        "initially", "after", "before", "meanwhile", "simultaneously",
        "subsequently", "eventually", "later", "afterward", "to start",
        "to conclude", "moreover", "furthermore", "however", "nevertheless"
    }
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    new_paragraphs, total_changes = [], 0
    for p in paragraphs:
        doc = sp(p)
        sentences = list(doc.sents)
        if len(sentences) < 3:
            new_paragraphs.append(p); continue
        moveable_indices = [i for i, sent in enumerate(sentences) if (sent.text.strip().lower().split()[0].rstrip(",;:") if sent.text.strip() else "") not in temporal_markers]
        if len(moveable_indices) > 2:
            shuffled_indices = moveable_indices.copy()
            get_rng().shuffle(shuffled_indices)
            new_sent_list = [s.text.strip() for s in sentences]
            for i, original_idx in enumerate(moveable_indices):
                new_sent_list[original_idx] = sentences[shuffled_indices[i]].text.strip()
            new_paragraphs.append(" ".join(new_sent_list)); total_changes += 1
        else:
            new_paragraphs.append(p)
    return "\n\n".join(new_paragraphs), total_changes

def pass_whitespace_jitter(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Disrupt bigram/trigram analysis by detectors with non-standard spaces."""
    new_chars, changes = [], 0
    for char in text:
        if char == " " and get_rng().random() < (0.05 * intensity):
            jitter_type = get_rng().choice(["hair", "thin"])
            new_chars.append("\u200A" if jitter_type == "hair" else "\u2009"); changes += 1
        else:
            new_chars.append(char)
    return "".join(new_chars), changes

def pass_final_enforcement(text: str) -> Tuple[str, int]:
    """Hard-enforce burstiness and syntactic variance requirements."""
    sp = load_spacy()
    if not sp: return text, 0
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    processed_paragraphs, total_changes = [], 0
    for paragraph in paragraphs:
        doc = sp(paragraph); sentences = list(doc.sents)
        if not sentences: processed_paragraphs.append(paragraph); continue
        p_word_count = len(word_tokenize(paragraph))
        lengths = [len(word_tokenize(s.text)) for s in sentences]
        depths = [get_tree_depth(next((t for t in s if t.dep_ == "ROOT"), None)) or 1 for s in sentences]
        has_very_short, has_very_long = any(l < 5 for l in lengths), any(l > 35 for l in lengths)
        has_flat, has_deep = any(d < 4 for d in depths), any(d > 12 for d in depths)
        new_sentences, changes = [s.text.strip() for s in sentences], 0
        if p_word_count > 60:
            if not has_very_short:
                for i, s_text in enumerate(new_sentences):
                    s_doc = sp(s_text)
                    if len(s_doc) > 10:
                        # Find a better split point than just word count
                        split_idx = -1
                        for token in s_doc:
                            if token.pos_ in ("VERB", "AUX") and 3 < token.i < len(s_doc) - 4:
                                split_idx = token.i
                                break
                        if split_idx == -1:
                            split_idx = min(4, len(s_doc) - 4)
                        
                        p1 = "".join([t.text_with_ws for t in s_doc[:split_idx+1]]).strip().rstrip(",; ") + "."
                        p2 = "".join([t.text_with_ws for t in s_doc[split_idx+1:]]).strip()
                        if p2:
                            new_sentences[i] = p1
                            new_sentences.insert(i + 1, p2[0].upper() + p2[1:])
                            changes += 1; has_very_short = True; break
            if not has_very_long and len(new_sentences) >= 2:
                for i in range(len(new_sentences) - 1):
                    s1, s2 = new_sentences[i], new_sentences[i+1]
                    if len(s1.split()) + len(s2.split()) > 20:
                        merged = s1.rstrip(".!?") + ", and furthermore, " + s2[0].lower() + s2[1:]
                        if len(merged.split()) > 35:
                            new_sentences[i] = merged; new_sentences.pop(i+1); changes += 1; has_very_long = True; break
        if not has_flat:
            # Skip the first sentence to avoid weird fragments at the start of paragraphs
            for i, s_text in enumerate(new_sentences):
                if i == 0: continue
                s_words = s_text.split()
                if 4 < len(s_words) < 15:
                    new_sentences[i] = " ".join(s_words[:min(5, len(s_words))]).rstrip(",; ") + "."
                    changes += 1; has_flat = True; break
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
                        changes += 1; has_deep = True; break
        processed_paragraphs.append(" ".join(new_sentences)); total_changes += changes
    return "\n\n".join(processed_paragraphs), total_changes

def pass_final_cleanup(text: str) -> Tuple[str, int]:
    """Removes artifacts and orphaned verb sentences."""
    sp = load_spacy()
    if not sp: return text, 0
    original_text = text
    # Fix spacing and duplicate punctuation
    text = re.sub(r'[\s,;>]+([.!?])', r'\1', text)
    text = re.sub(r'\s+([.,])\s+', r'\1 ', text)
    text = re.sub(r'\.\.+', '.', text)
    text = re.sub(r',,+', ',', text)
    
    doc = sp(text)
    new_sentences, removed_count = [], 0
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text: continue
        first_token = next((t for t in sent if not t.is_punct and not t.is_space), None)
        if first_token and first_token.pos_ in ("VERB", "AUX"):
            if first_token.tag_ == "VBG": new_sentences.append(sent_text); continue
            if not any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass", "expl") for t in sent):
                removed_count += 1; continue
        new_sentences.append(sent_text)
    final_text = " ".join(new_sentences)
    return final_text, (1 if final_text != original_text else 0)

# ===========================================================================
# Pipeline helpers
# ===========================================================================

def apply_light_passes(txt: str, intensity: float, changes: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> str:
    c_txt, c0 = pass_signature_phrase_breaker(txt); changes["signature_phrase_breaks"] += c0
    c_txt, c1 = pass_phrase_replacement(c_txt, intensity); changes["phrases_replaced"] += c1
    c_txt, c4 = pass_lexical(c_txt, profile, intensity); changes["lexical_substitutions"] += c4
    return c_txt

def apply_full_passes(txt: str, intensity: float, changes: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> str:
    c_txt, c0 = pass_signature_phrase_breaker(txt); changes["signature_phrase_breaks"] += c0
    c_txt, c1 = pass_phrase_replacement(c_txt, intensity); changes["phrases_replaced"] += c1
    c_txt, c_ms = pass_morphological_shifting(c_txt, intensity); changes["morphological_shifting"] += c_ms
    c_txt, c_hw = pass_human_wordiness(c_txt, intensity); changes["human_wordiness"] += c_hw
    c_txt, c2 = pass_restructuring(c_txt, intensity); changes["sentences_restructured"] += c2
    c_txt, c3 = pass_burstiness(c_txt, profile, intensity); changes["burstiness_injections"] += c3
    c_txt, c35 = pass_rhythm_sculpting(c_txt, profile, intensity); changes["rhythm_adjustments"] += c35
    c_txt, c4 = pass_lexical(c_txt, profile, intensity); changes["lexical_substitutions"] += c4
    c_txt, c5 = pass_style_overlay(c_txt, profile, intensity); changes["style_overlays"] += c5
    c_txt, c6 = pass_voice_conversion(c_txt, intensity); changes["voice_conversions"] += c6
    c_txt, c7 = pass_clause_reorder(c_txt, intensity); changes["clause_reorders"] += c7
    c_txt, c_yi = pass_yoda_inversion(c_txt, intensity); changes["yoda_inversions"] += c_yi
    c_txt, c_ai = pass_appositive_injection(c_txt, intensity); changes["appositive_injections"] += c_ai
    c_txt, c8 = pass_discourse_markers(c_txt, intensity); changes["discourse_markers"] += c8
    c_txt, cn = pass_nuance_injection(c_txt, intensity); changes["nuance_injections"] += cn
    c_txt, c_det = pass_determiner_scramble(c_txt, intensity); changes["determiner_scrambles"] += c_det
    c_txt, c_zwj = pass_zwj_jitter(c_txt, intensity); changes["zwj_jitters"] += c_zwj
    c_txt, c_inv = pass_invisible_padding(c_txt, intensity); changes["invisible_padding"] += c_inv
    c_txt, c_pp = pass_punctuation_personality(c_txt, intensity); changes["punctuation_personality"] += c_pp
    c_txt, c_cr = pass_cross_referencing(c_txt, intensity); changes["cross_referencing"] += c_cr
    c_txt, c_hu = pass_hedging_uncertainty(c_txt, intensity); changes["hedging_uncertainty"] += c_hu
    c_txt, c9 = pass_syntactic_variance(c_txt, intensity); changes["syntactic_variance"] += c9
    c_txt, c10 = pass_structural_reset(c_txt, intensity); changes["structural_resets"] += c10
    c_txt, crq = pass_rhetorical_questions(c_txt, intensity); changes["rhetorical_questions"] += crq
    c_txt, cji = pass_jargon_injection(c_txt, intensity); changes["jargon_injections"] += cji
    c_txt, che = pass_human_errors(c_txt, intensity); changes["human_errors"] += che
    c_txt, cfe = pass_final_enforcement(c_txt); changes["final_enforcement"] += cfe
    return c_txt

# ===========================================================================
# Main Pipeline
# ===========================================================================

def humanize_text(text: str, intensity: float = 0.7) -> Dict[str, Any]:
    """
    Main humanization pipeline with forced full passes at intensity >= 0.7.
    """
    voice_profile = extract_voice(text)
    initial_score = detect_ai_score(text)
    word_count = len(word_tokenize(text))

    if word_count > 5000:
        sentences, chunks, current_chunk, current_len = sent_tokenize(text), [], [], 0
        for sent in sentences:
            sent_len = len(word_tokenize(sent))
            if current_len + sent_len > 1000 and current_chunk:
                chunks.append(" ".join(current_chunk)); current_chunk, current_len = [sent], sent_len
            else:
                current_chunk.append(sent); current_len += sent_len
        if current_chunk: chunks.append(" ".join(current_chunk))
        chunk_results = [humanize_text(chunk, intensity) for chunk in chunks]
        final_text = " ".join([c["humanized_text"] for c in chunk_results])
        final_changes = {}
        for c in chunk_results:
            for k, v in c["changes_made"].items():
                if isinstance(v, (int, float)): final_changes[k] = final_changes.get(k, 0) + v
        return {
            "humanized_text": final_text,
            "passes_applied": chunk_results[0]["passes_applied"] if chunk_results else [],
            "changes_made": final_changes,
            "original_score": initial_score,
            "humanized_score": detect_ai_score(final_text),
            "voice_profile": voice_profile,
        }

    get_rng().seed(get_seed(text))
    used_phrases_var.set(set())
    profile = None
    current_text, citations_map = protect_citations(text)

    changes = {
        "phrases_replaced": 0, "sentences_restructured": 0, "rhythm_adjustments": 0, "burstiness_injections": 0,
        "lexical_substitutions": 0, "style_overlays": 0, "voice_conversions": 0, "clause_reorders": 0,
        "discourse_markers": 0, "syntactic_variance": 0, "structural_resets": 0, "determiner_scrambles": 0,
        "audit_iterations": 0, "voice_adjustments": 0, "final_cleanups": 0, "paragraph_rhythm_fixes": 0,
        "back_translation": 0, "nuance_injections": 0, "rhetorical_questions": 0, "jargon_injections": 0,
        "whitespace_jitter": 0, "structural_chaos": 0, "human_errors": 0, "signature_phrase_breaks": 0,
        "final_enforcement": 0,
        "morphological_shifting": 0, "human_wordiness": 0, "yoda_inversions": 0,
        "appositive_injections": 0, "zwj_jitters": 0, "invisible_padding": 0,
        "punctuation_personality": 0, "cross_referencing": 0, "hedging_uncertainty": 0,
    }

    is_high_intensity, is_short_text = intensity >= 0.7, word_count < 50
    
    # Back-translation pre-processor (only at high intensity and not too short)
    if is_high_intensity and not is_short_text:
        current_text, cbt = pass_back_translation(current_text)
        changes["back_translation"] += cbt

    if is_short_text:
        current_text = apply_full_passes(current_text, intensity, changes, profile) if is_high_intensity else apply_light_passes(current_text, intensity, changes, profile)
        passes_applied = ["full_pipeline"] if is_high_intensity else ["light_pipeline"]
    else:
        segments = _split_into_segments(current_text, target_words=250)
        processed_segments = [apply_full_passes(seg, intensity, changes, profile) if (is_high_intensity or score_segment(seg) >= 25) else apply_light_passes(seg, intensity, changes, profile) for seg in segments]
        current_text = " ".join(processed_segments)
        passes_applied = [
            "citation_guard", "confidence_gradient", "signature_phrase_breaker", "phrase_replacement",
            "morphological_shifting", "human_wordiness", "restructuring", "burstiness", "rhythm_sculpting",
            "lexical", "style_overlay", "voice_conversion", "clause_reorder", "yoda_inversion",
            "appositive_injection", "discourse_markers", "nuance_injection", "determiner_scramble",
            "zwj_jitter", "invisible_padding", "punctuation_personality", "cross_referencing",
            "hedging_uncertainty", "syntactic_variance", "structural_reset", "rhetorical_questions",
            "jargon_injection",
        ]
        if changes.get("back_translation", 0) > 0:
            passes_applied.insert(0, "back_translation")

    if not is_short_text:
        max_iterations, last_score = 5, detect_ai_score(restore_citations(current_text, citations_map))
        for iteration in range(max_iterations):
            if last_score <= 15: break
            changes["audit_iterations"] += 1
            if last_score > 20:
                current_text, csc = pass_structural_chaos(current_text); changes["structural_chaos"] += csc
                current_text, csr = pass_structural_reset(current_text, intensity); changes["structural_resets"] += csr
                current_text, cwj = pass_whitespace_jitter(current_text, intensity * 0.8); changes["whitespace_jitter"] += cwj

            new_sentences = []
            for s in score_sentences(current_text):
                sent_text = s["text"]
                if s["score"] > 30:
                    sent_text, _ = pass_signature_phrase_breaker(sent_text)
                    sent_text, _ = pass_phrase_replacement(sent_text, intensity)
                    sent_text, cms = pass_morphological_shifting(sent_text, intensity); changes["morphological_shifting"] += cms
                    sent_text, cyi = pass_yoda_inversion(sent_text, intensity); changes["yoda_inversions"] += cyi
                    sent_text, cai = pass_appositive_injection(sent_text, intensity); changes["appositive_injections"] += cai
                    sent_text, _ = pass_structural_reset(sent_text, intensity)
                    sent_text, _ = pass_lexical(sent_text, profile, intensity)
                    sent_text, _ = pass_nuance_injection(sent_text, intensity)
                    sent_text, crq = pass_rhetorical_questions(sent_text, intensity); changes["rhetorical_questions"] += crq
                    sent_text, cji = pass_jargon_injection(sent_text, intensity); changes["jargon_injections"] += cji
                    if s["score"] > 50:
                        sent_text, cwj = pass_whitespace_jitter(sent_text, intensity); changes["whitespace_jitter"] += cwj
                new_sentences.append(sent_text)
            current_text = " ".join(new_sentences)
            current_text = re.sub(r'\b(yet|but|so|and|still)\s*[,]*\s*(yet|but|so|and|still|however|moreover|furthermore|nevertheless)\b', r'\2', current_text, flags=re.IGNORECASE)
            current_text = re.sub(r'\b(however|moreover|furthermore|nevertheless|consequently|therefore)\s*,\s*\b(however|moreover|furthermore|nevertheless|consequently|therefore)\b', r'\1', current_text, flags=re.IGNORECASE)
            current_restored = restore_citations(current_text, citations_map)
            new_score = detect_ai_score(current_restored)
            if new_score >= last_score and iteration > 1: break
            last_score = new_score
        if changes["audit_iterations"] > 0: passes_applied.append("self_audit")

    if voice_profile:
        current_text = apply_voice(current_text, voice_profile)
        changes["voice_adjustments"] += 1; passes_applied.append("voice_preserved")

    current_text, ccl = pass_final_cleanup(current_text)
    if ccl > 0: changes["final_cleanups"] += ccl; passes_applied.append("artifact_cleanup")
    current_text = restore_citations(current_text, citations_map)
    current_text = current_text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    current_text = re.sub(r'([.!?]\s+)([a-z])', fix_caps, current_text)

    return {
        "humanized_text": current_text,
        "passes_applied": passes_applied,
        "changes_made": changes,
        "original_score": initial_score,
        "humanized_score": detect_ai_score(current_text),
        "voice_profile": voice_profile,
    }
