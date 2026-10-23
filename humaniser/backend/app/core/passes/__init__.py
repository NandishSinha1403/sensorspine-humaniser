# passes/ — Modular NLP pass functions for the Humaniser pipeline.
# Each sub-module groups related transformations for maintainability.

from app.core.passes.utils import (
    get_rng, get_used_phrases, load_spacy, safe_split_sentence,
    is_valid_sentence, extract_primary_subject, protect_citations,
    restore_citations, get_seed, fix_caps,
    SPACY_TO_WORDNET, DETERMINER_ALTERNATIVES, TRANSITION_ALTERNATIVES,
    DEEPENING_PHRASES, MODERN_ACADEMIC_JARGON, JARGON_REPLACEMENT_MAP,
)
from app.core.passes.phrase_replacement import (
    pass_signature_phrase_breaker, pass_phrase_replacement,
    AI_PHRASE_REPLACEMENTS, COMPILED_AI_PHRASES,
)
from app.core.passes.structural import (
    pass_restructuring, pass_structural_reset, pass_structural_chaos,
    pass_morphological_shifting, pass_appositive_injection,
)
from app.core.passes.rhythm import (
    pass_burstiness, pass_rhythm_sculpting, pass_paragraph_wave,
)
from app.core.passes.syntactic import (
    pass_syntactic_clefting, pass_syntactic_variance, pass_clause_reorder,
)
from app.core.passes.lexical import (
    pass_lexical, pass_academic_ngrams, pass_jargon_injection,
)
from app.core.passes.style import (
    pass_style_overlay, pass_nuance_injection, pass_determiner_scramble,
    pass_epistemic_hedging, pass_hedging_uncertainty, pass_cross_referencing,
)
from app.core.passes.adversarial import (
    pass_adversarial_punctuation, pass_zwj_jitter, pass_invisible_padding,
    pass_whitespace_jitter, pass_punctuation_personality,
)
from app.core.passes.cleanup import (
    pass_final_enforcement, pass_final_cleanup, pass_human_errors,
)
