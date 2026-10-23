"""
Phrase replacement passes — AI signature phrase breaking and expanded phrase substitution.
"""

import re
from typing import Tuple
from app.core.passes.utils import get_rng, get_used_phrases

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
    "has been shown": ["appears", "seems", "the evidence suggests", "is widely seen"],
    "research has shown": ["studies suggest", "evidence points to", "work here indicates", "data shows", "scholars find"],
    "plays a crucial role": ["matters greatly", "is central to", "drives", "is vital", "is key"],
    "delve into": ["examine", "look at", "explore", "consider", "investigate"],
    "it is clear that": ["clearly", "evidently", "plainly", "without doubt"],
    "in today's world": ["today", "currently", "at present", "nowadays"],
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
    "at the end of the day": ["ultimately", "finally", "in the end"],
    "it can be argued": ["one might argue", "it's possible to say", "arguably"],
    "one must consider": ["one should look at", "it's worth considering", "we must see"],
    "minimizes harm": ["reduces damage", "cuts down on harm", "lowers risk"],
    "crucial for development": ["key for growth", "vital for progress", "central to development"],
    "notably": ["interestingly", "markedly", "importantly", "of note"],
    # --- Corpus-specific & Phrases often found in AI output ---
    "is a subject of growing concern": ["a rising issue"],
    "has the potential to": ["could", "might", "may well", "can potentially"],
    "are some of the factors": ["count among the reasons", "matter here"],
    "is essential for": ["really matters for", "is key to", "is vital for"],
    "it is important to": ["one should", "it matters to"],
    "is a critical concern": ["really matters", "is a pressing issue"],
    "raises significant concerns": ["causes unease", "raises flags"],
    "poses significant": ["creates real", "brings serious", "presents major"],
    "significant challenges": ["major issues"],
    "significant implications": ["big consequences", "real effects", "major fallout"],
    "a key focus of": ["central to", "at the heart of"],
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
    "the development of": ["the advancement of", "the evolution of", "the progression of", "progress in"],
    "a growing concern": ["an increasing worry", "a rising problem"],
    "the use of": ["using", "relying on", "employment of"],
    "the impact of": ["how much", "the effect of", "the weight of"],
    "has resulted in": ["led to", "caused", "brought about", "triggered"],
    "addressing these challenges": ["tackling these issues", "dealing with this", "solving these problems"],
    "however, there are": ["but", "still", "yet", "even so, there are"],
    "nevertheless, there are": ["but", "still", "even so", "all the same"],
    "however, the": ["but the", "still, the", "yet the"],
    "nevertheless, the": ["but the", "still, the"],
    "is expected to": ["will likely", "should", "is predicted to"],
    "the advancement of": ["progress in", "gains in", "advances in"],
    "the challenges of": ["difficulties with", "issues with"],
    "in various aspects of": ["across different parts of", "in many areas of"],
    "contributing to these": ["feeding into these", "adding to these", "helping cause these"],
    "promoting the use of": ["pushing for", "encouraging", "backing"],
    "ensuring the well-being": ["safeguarding", "protecting"],
    "the protection and promotion": ["defending and advancing", "safeguarding and pushing"],
    "one of the most": ["among the biggest", "quite possibly the", "a major"],
    "it has become increasingly": ["it now seems more", "is more and more"],
    "is the process of": ["means", "involves", "refers to", "is about"],
    "in a way that": ["so that", "allowing", "whereby"],
    "play an important role": ["matter a lot", "count for something", "drive change", "are significant"],
    "has become essential": ["now matters more than ever", "is now vital", "is now a must"],
    "not only ... but also": ["both ... and", "as well as"],
    "by contrast": ["on the other hand", "in contrast", "conversely"],
    "in a similar fashion": ["similarly", "likewise", "in the same way"],
    "it is evident that": ["clearly,", "plainly,"],
    "there is a need to": ["we should", "one must"],
    "highly effective": ["very good", "quite successful"],
    "a wide array of": ["many", "lots of", "various", "a broad range of"],
}

# Pre-compiled regex patterns sorted longest-first
COMPILED_AI_PHRASES = []
for phrase in sorted(AI_PHRASE_REPLACEMENTS.keys(), key=len, reverse=True):
    pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
    COMPILED_AI_PHRASES.append((pattern, AI_PHRASE_REPLACEMENTS[phrase]))


def pass_signature_phrase_breaker(text: str) -> Tuple[str, int]:
    """Force-replaces every occurrence of AI signature phrases."""
    from app.core.detector import AI_SIGNATURE_PHRASES

    count = 0
    new_text = text
    sorted_phrases = sorted(AI_SIGNATURE_PHRASES, key=len, reverse=True)

    for phrase in sorted_phrases:
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)

        def replace_fn(match, _phrase=phrase):
            nonlocal count
            original = match.group(0)
            replacements = AI_PHRASE_REPLACEMENTS.get(
                _phrase.lower(),
                ["also", "and", "in fact", "actually", "notably"]
            )
            replacement = get_rng().choice(replacements)
            count += 1
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        new_text = pattern.sub(replace_fn, new_text)

    return new_text, count


def pass_phrase_replacement(text: str, intensity: float = 1.0) -> Tuple[str, int]:
    """Replace common AI phrases with varied alternatives."""
    count = 0
    new_text = text
    used = get_used_phrases()
    force_all = intensity >= 0.8

    for pattern, replacements in COMPILED_AI_PHRASES:
        if not force_all and get_rng().random() > intensity:
            continue

        def replace_fn(match, _replacements=replacements):
            nonlocal count
            original = match.group(0)
            available = [r for r in _replacements if r.lower() not in used]
            if not available:
                available = _replacements
            replacement = get_rng().choice(available)
            used.add(replacement.lower())
            count += 1
            if original[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement

        new_text = pattern.sub(replace_fn, new_text)

    return new_text, count
