from tqdm.auto import tqdm
import numpy as np

from datastruct import align_spaces
from datastruct import Word
from common import word_regex, word_regex2

def collect_word_space_pairs(sequence):
    def _collect_word_space_pairs(sequence):
        last_word = ''
        for w in sequence:
            if word_regex.match(w):
                yield last_word, w
                last_word = None
            else:
                last_word = w
        if last_word is not None:
            yield last_word, ''
    return list(_collect_word_space_pairs(sequence))


def split_to_fragments(words):
    buf = []
    for w, s, is_known in words:
        if is_known:
            if len(buf) > 0:
                yield buf, False
                buf = []
            yield [(w, s)], True
        else:
            buf.append((w, s))
    if len(buf) > 0:
        yield buf, False

from datastruct import align_spaces

def find_in_fixer(mt_word, fixer):
    if not isinstance(mt_word, Word):
        mt_word = Word(mt_word)
    N = max(fixer.vocabs)
    for i in range(N):
        for ref_word, tree_dist in fixer.vocabs[i + 1].find_closest(mt_word.sequence):
            if tree_dist != 0:
                continue
            alignment_result = mt_word.distance_to(ref_word)
            if alignment_result.distance == 0:
                return ref_word
    return None
        
def fix_spaces(seq, fixer, max_words=4, verbose=False):
    if len(seq) == 0:
        return seq
    # print(len(seq))
    max_words = min(max_words, len(seq))
    # print(max_words)
    # max words to glue before splitting
    i = 0
    new_seq = []
    while i < len(seq):
        close_word_is_found = False
        for k in range(max_words):
            # i:i + k + 1
            subseq = seq[i:i + k + 1]
            space = subseq[-1][1]
            mt_word = Word([w for w, s in subseq])
            if len(mt_word) == 0:
                break
            ref_word = find_in_fixer(mt_word, fixer)
            if ref_word is None:
                continue
            alignment_result = mt_word.distance_to(ref_word)
            alignment = alignment_result.do_backtracing(
                "".join(mt_word.sequence), ref_word.tokens)
            mt, rt, m = list(zip(*alignment))
            
            corrected_words = tuple([
                ("".join(word), "".join([x.char for x in ref]))
                for word, ref in align_spaces(mt, rt)
            ])
            if np.all([len(w) > 1 for w, r in corrected_words]):
                if k > 0 and verbose:
                    print(mt_word, ref_word)
                seq_fragment = [(tuple([w for w, r in corrected_words]), space)]
                new_seq += seq_fragment
                i += k
                close_word_is_found = True
                break
        if not close_word_is_found:
            new_seq.append(seq[i])
        i += 1
    return new_seq

def greedy_correction_one(raw_seq, fixer):
    # print(seq)
    if isinstance(raw_seq, str):
        seq = word_regex2.split(raw_seq)
    else:
        seq = raw_seq
    pairs = collect_word_space_pairs(seq)
    words = [
        (word, space, len(fixer.vocabs[1].find(word)) > 0)
        for word, space in pairs
    ]
    new_fragments = []
    for fragment, is_known in split_to_fragments(words):
        if is_known:
            # print("fragment", fragment)
            new_fragments += fragment
        else:
            fixed_fragment = fix_spaces(fragment, fixer)
            new_fragments+= fixed_fragment
    # fragments = [join_if_tuple(w) + s for w, s in new_fragments]
    # text = "".join(fragments)
    return new_fragments

def greedy_correction(texts, fixer):
    for raw_seq in tqdm(texts):
        yield greedy_correction_one(raw_seq, fixer)


# next(greedy_correction(mt_sequences_train))
# mt_sequences_train[1]