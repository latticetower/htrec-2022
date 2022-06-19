# todo: need to fix paths to ensure everything works as expected
from lm.markov.models import LM

def lmr(text, word=" ς ", replacements=["ς ", " "], lm=lm):
    scores = []
    for the_candidate in replacements:
        scores.append(lm.cross_entropy(text.replace(word, the_candidate)))
    text_out = text.replace(word, replacements[scores.index(min(scores))])
    return text_out

def make_lm(train_text, verbose=False):
    lm = LM(gram="CHAR").train(train_text); #cslm.generate_text()
    if verbose:
        print(lm.generate_text())
    return lm
    
