from dataclasses import dataclass
import unicodedata
from itertools import islice
from collections import deque
import numpy as np
import re
word_regex = re.compile("\W+")
word_regex2 = re.compile("(\W+)")

UP = 1 # "\u2191"  # 1
LEFT = 2 # "\u2190"  # 2
DIAG = 3 # "\u2196"  # 3

backtrace_symbols = {
    UP: "\u2191",
    LEFT: "\u2190",
    DIAG: "\u2196"
}




def _remove_cap(char):
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", char)[:1])
def remove_cap(s):
    return "".join([_remove_cap(char) for char in s])

VOWELS = set(["α", "ε", "η" "ι", "ο", "υ", "ω"])
def is_vowel(ch):
    return _remove_cap(ch) in VOWELS


@dataclass
class AlignmentResult:
    distance: int
    weight_matrix: np.array
    backtrace_matrix: np.array
    x: str
    y: str
    def show_backtrace(self):        
        for line in self.backtrace_matrix:
            print(" ".join([backtrace_symbols.get(w, " ") for w in line]))

    def do_backtracing(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        i = len(x)
        j = len(y)
        xs = []
        ys = []
        matches = []
        while i > 0 or j > 0:
            last_pos = self.backtrace_matrix[i, j]
            
            xs.append(x[i - 1] if i > 0 and last_pos != LEFT else None)
            ys.append(y[j - 1] if j > 0 and last_pos != UP else None)
            if last_pos != LEFT:
                i -= 1
            if last_pos != UP:
                j -= 1
            matches.append(last_pos == DIAG)
        edits = list(zip(xs, ys, matches))
        edits.reverse()
        return edits
    def __str__(self):
        return f"AlignmentResult distance={self.distance}, x={self.x}, y={self.y}"

    def __repr__(self):
        return str(self)


def align(x, y, match_cost=1, gap_cost=-1, mismatch_cost=-np.inf, decision_func=max):
    weight_matrix = np.zeros((len(x) + 1, len(y) + 1), dtype=int)
    backtrace_matrix = np.ones_like(weight_matrix) # * gap_cost
    weight_matrix[0, 1:] = gap_cost * np.arange(1, len(y) + 1)
    weight_matrix[1:, 0] = gap_cost * np.arange(1, len(x) + 1)
    backtrace_matrix[0, 1:] = LEFT
    backtrace_matrix[1:, 0] = UP

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            delta = match_cost if x[i - 1] == y[j - 1] else mismatch_cost
            diag_value = weight_matrix[i - 1, j - 1] + delta
            up_value = weight_matrix[i - 1, j] + gap_cost
            left_value = weight_matrix[i, j - 1] + gap_cost
            d = decision_func(up_value, left_value)
            weight_matrix[i, j] = d
            if d == up_value:
                backtrace_matrix[i, j] = UP
            elif d == left_value:
                backtrace_matrix[i, j] = LEFT
            # if x[i - 1] == y[j - 1]:
            #     match_cost 
            # if diag_value >= d:
            #    diag_value = weight_matrix[i - 1, j - 1] + delta
            if diag_value == decision_func(diag_value, d):
                weight_matrix[i, j] = diag_value
                backtrace_matrix[i, j] = DIAG
            
    return AlignmentResult(weight_matrix[-1, -1], weight_matrix, backtrace_matrix, x, y)



def sliding_window(iterable, n):
    # Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
    # sliding_window('ABCDEFG', 4) -> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = deque(islice(it, n), maxlen=n)
    if len(window) == n:
        yield tuple(window)
    for x in it:
        window.append(x)
        yield tuple(window)


def separate_sequences(structure):
    prefixes = []
    for item in structure:
        new_prefixes = []
        if isinstance(item, list):
            if len(prefixes) == 0:
                for it in item:
                    if not isinstance(it, list):
                        it = [it]
                    new_prefixes.append(it)
            else:
                for it in item:
                    if not isinstance(it, list):
                        it = [it]
                    for p in prefixes:
                        new_prefixes.append(p + it)
        else:
            if len(prefixes) == 0:
                new_prefixes.append([item])
            else:
                for p in prefixes:
                    new_prefixes.append(p + [item])
        prefixes = new_prefixes
    return prefixes


def correct_sigmas_in_word(word):
    if word_regex.match(word):
        return word
    #"ς"
    cw = []
    if len(word) < 1:
        return word
    for c in word[:-1]:
        name = unicodedata.name(c)
        new_char = c
        if name.find("FINAL SIGMA") >= 0:
            name = name.replace("FINAL ", "")
            try:
                new_char = unicodedata.lookup(name)
            except:
                new_char = c
        cw.append(new_char)
    last_char = word[-1]
    name = unicodedata.name(last_char)
    if name.find("SIGMA") >= 0 and name.find("FINAL SIGMA") < 0:
        name = name.replace("SIGMA", "FINAL SIGMA")
        try:
            new_char = unicodedata.lookup(name)
        except:
            new_char = last_char
    else:
        new_char = last_char
    cw.append(new_char)
    return "".join(cw)

def postprocess_sigmas(sentence):
    words = [correct_sigmas_in_word(x) for x in word_regex2.split(sentence)]
    return "".join(words)

and_regex = re.compile("\s+AND\s+")
def split_accents(ch):
    name = unicodedata.name(ch)
    i = name.find("WITH")
    if i < 0:
        return ch, []
    ch1 = remove_cap(ch)
    accents = and_regex.split(name[i + 5:])
    return ch1, accents
    

def append_accents(ch, accents):
    # ch1 = remove_cap(ch)
    name = unicodedata.name(ch)
    i = name.find("WITH")
    accents2 = []
    prefix = name
    if i >= 0:
        prefix = name[:i]
    accents2 = and_regex.split(name[i+5:])
    for a in accents:
        if not a in set(accents2):
            accents2.append(a)
    accents2 = " AND ".join(accents2)
    if len(accents2) > 0:
        name = prefix + " WITH " + accents2
    try:
        new_char = unicodedata.lookup(name)
    except:
        new_char = ch
    return new_char

def fix_accent_diphthong(text):
    if len(text) < 1:
        return text
    vowels = [is_vowel(ch) for ch in text]
    chars = []
    skip = -1
    for i, ch in enumerate(text):
        if i == skip:
            continue
        if vowels[i]:
            if i == len(text) - 1:
                chars.append(ch)
                continue
            if vowels[i + 1] and text[i + 1] in ["ι", "υ"]:
                # in diphthong -> fix accents if there are any
                ch1, accents = split_accents(ch)
                chars.append(ch1)
                ch2 = append_accents(text[i + 1], accents)
                chars.append(ch2)
                skip = i + 1
            else:
                chars.append(ch)
        else:
            chars.append(ch)
    return "".join(chars)

def fix_accents(text):
    from datastruct import Word, remove_cap
    words = word_regex2.split(text)
    corrected = []
    cases = {
        "τόν": "όν",
        "τοῦ": "οῦ",
        "τήν": "ᾱ́ν",
        "τῆς": "ᾶς",
        "τοῦ": "οῦ",
        #"τοῖν": "",
        "τούς": "ούς",
        "τῶν": "ῶν",
        "τοῖς": "οῖς",
        "ταῖς": "αῖς", 
    }
    cases = {remove_cap(k): (k, remove_cap(v), v) for k, v in cases.items()}

    for i, w in enumerate(words):
        # print(words)
        if len(w) == 0:
            corrected.append(w)
            continue
        if word_regex.match(w):
            corrected.append(w)
        else:
            word = Word(w)
            if word.no_caps == remove_cap("καὶ"):
                corrected.append("καὶ")
                continue
            if word.no_caps == remove_cap("ἐις"):
                corrected.append("ἐις")
                continue
            # if is_vowel(w[0]):
            #     pass
            case_info = cases.get(word.no_caps, None)
            if case_info is None or i >= len(words) - 2:
                corrected.append(w)
                continue
            the, ending_no_caps, ending = case_info
            if remove_cap(words[i + 2]).endswith(ending_no_caps):
                # print(the, ending_no_caps)
                corrected.append(the)
            else:
                corrected.append(w)
    return "".join(corrected)
            

            