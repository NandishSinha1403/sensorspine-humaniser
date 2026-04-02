import pytest
import spacy
import random
from app.core.humanizer import pass_lexical
from nltk.corpus import wordnet

@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")

def test_verb_conjugation(nlp):
    """Verify that verbs are inflected correctly to match the original tense."""
    text = "The algorithms were continuously optimizing the routing protocols."
    random.seed(42)
    
    result, changes = pass_lexical(text, profile={})
    
    doc_res = nlp(result)
    doc_orig = nlp(text)
    
    # Find the token that replaced 'optimizing'
    # 'optimizing' in original
    orig_opt = next(t for t in doc_orig if t.text == "optimizing")
    
    # In result, we look for the token at the same position relative to surrounding words
    # "continuously [replacement] the"
    # Wait, 'continuously' was also swapped in the log to 'incessantly'
    # "incessantly [replacement] the"
    
    # Let's just find the token that has the same index as orig_opt.i
    res_opt = doc_res[orig_opt.i]
    
    if res_opt.text.lower() != "optimizing":
        assert res_opt.tag_ == "VBG", f"Replacement '{res_opt.text}' should have tag VBG, but has {res_opt.tag_}"
        assert res_opt.text.endswith("ing"), f"Replacement '{res_opt.text}' should end in -ing"

def test_noun_plurality(nlp):
    """Verify that nouns maintain their plurality after substitution."""
    text = "The administrative departments processed the queries."
    random.seed(10)
    
    result, changes = pass_lexical(text, profile={})
    doc_res = nlp(result)
    doc_orig = nlp(text)
    
    orig_dept = next(t for t in doc_orig if t.text == "departments")
    res_dept = doc_res[orig_dept.i]
    
    if res_dept.text.lower() != "departments":
        assert res_dept.tag_ == "NNS", f"Replacement '{res_dept.text}' should be plural (NNS)"

def test_pos_context_awareness(nlp):
    """Verify that synonyms are chosen based on the actual POS in context."""
    text_adj = "He drives a fast car."
    random.seed(42)
    result_adj, _ = pass_lexical(text_adj, profile={})
    
    text_verb = "He decided to fast today."
    random.seed(42)
    result_verb, _ = pass_lexical(text_verb, profile={})
    
    doc_adj = nlp(result_adj)
    doc_verb = nlp(result_verb)
    
    orig_adj = next(t for t in nlp(text_adj) if t.text == "fast")
    orig_verb = next(t for t in nlp(text_verb) if t.text == "fast")
    
    repl_adj = doc_adj[orig_adj.i].text.lower()
    repl_verb = doc_verb[orig_verb.i].text.lower()
    
    if repl_adj != "fast":
        syns = [l.name().replace('_', ' ') for s in wordnet.synsets("fast", pos=wordnet.ADJ) for l in s.lemmas()]
        assert repl_adj in syns
        
    if repl_verb != "fast":
        syns = [l.name().replace('_', ' ') for s in wordnet.synsets("fast", pos=wordnet.VERB) for l in s.lemmas()]
        assert repl_verb in syns

def test_capitalization_preservation(nlp):
    """Verify that capitalization is preserved in swapped words."""
    text = "Departments are essential."
    random.seed(1)
    
    result, _ = pass_lexical(text, profile={})
    doc_res = nlp(result)
    first_word = doc_res[0].text
    if first_word.lower() != "departments":
        assert first_word[0].isupper(), f"Replacement '{first_word}' should be capitalized."
