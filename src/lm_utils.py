# todo: need to fix paths to ensure everything works as expected
# from lm.markov.models import LM
from lm.markov.models import LM

def lm_score(text, replacements, lm=None, return_index=False, return_corrected=False):
    texts = [text]
    scores = [lm.cross_entropy(text)]
    for the_candidate in replacements:
        texts.append(the_candidate)
        scores.append(lm.cross_entropy(the_candidate))
    best_index = scores.index(min(scores))
    if return_corrected:
        return texts[best_index], best_index > 0
    if return_index:
        return texts[best_index], best_index
    return texts[best_index]
    

def lmr(text, word=" ς ", replacements=["ς ", " ", " σ ", " σ"], lm=None):
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
    
