from collections import defaultdict
from collections import Counter
from sklearn.neighbors import BallTree
import numpy as np
from dataclasses import dataclass

from common import remove_cap
from common import align, AlignmentResult


@dataclass
class CharToken:
    char: str
    bow: bool = False
    eow: bool = False
    def __eq__(self, other):
        if isinstance(other, CharToken):
            return self.char == other.char
        return self.char == other


def word2tokens(word):
    if len(word) < 1:
        return []
    if len(word) == 1:
        return [CharToken(word[0], bow=True, eow=True)]

    tokens = [CharToken(word[0], bow=True, eow=False)]
    if len(word) > 2:
        tokens.extend([CharToken(ch, bow=False, eow=False) for ch in word[1:-1]])
    tokens.append(CharToken(word[-1], eow=True))
    return tokens


def tokens2words(tokens):
    words = []
    word = []
    for t in tokens:
        if t is None:
            continue
        word.append(t.char)
        if t.eow:
            if len(word) > 0:
                words.append("".join(word))
                word = []

    if len(word) > 0:
        words.append("".join(word))
    return words


def split_by_word(tokens1, tokens2, verbose=False):
    pairs = []
    all_pairs = []
    for t1, t2 in zip(tokens1, tokens2):
        if verbose:
            print(t1, t2)
        if t1 is None or t2 is None:
            pairs.append((t1, t2))
            continue
        if t1.bow and t2.bow:
            if len(pairs) > 0:
                all_pairs.append(pairs)
            pairs = [(t1, t2)]
            continue
        pairs.append((t1, t2))
        if t1.eow and t2.eow:
            # pairs.append((t1, t2))
            if len(pairs) > 0:
                all_pairs.append(pairs)
            pairs = []
            # continue
    if len(pairs) > 0:
        all_pairs.append(pairs)
    return all_pairs


def align_spaces(mt_tokens, ref_tokens):
    word = []
    parts = []
    for mt, rt in zip(mt_tokens, ref_tokens):
        if rt is None:
            word.append((mt, rt))
            continue
        if rt.bow:
            if len(word) > 0:
                parts.append(word)
                word = []
        word.append((mt, rt))
        if rt.eow:
            if len(word) > 0:
                parts.append(word)
                word = []
    if len(word) > 0:
        parts.append(word)
    # [for word in parts]
    # parts = [[ mt for mt, rt in word if mt is not None] for word in parts]
    # parts = [word for word in parts if len(word) > 0]
    # parts = [word2tokens([token.char for token in word]) for word in parts]
    parts = [list(zip(*word)) for word in parts ]
    parts = [
        (
            [x for x in mt_word if x is not None],
            [x for x in ref_word if x is not None]
        )
        for mt_word, ref_word in parts
    ]
    return parts


def bow2vector(bow, letter2index, n_letters):
    data = np.zeros((n_letters, ), dtype=int)
    for letter, count in bow.items():
        if not letter in letter2index:
            continue
        i = letter2index[letter]
        data[i] = count
    return data


class BasicVocab:
    def __init__(self):
        self.words = []
        self.word_counter = defaultdict(int)
    def add_sentence(self, words):
        for word in words:
            if word not in self.word_counter:
                self.words.append(word)
            self.word_counter[word] += 1
    def __repr__(self):
        return f"BasicVocab with {len(self.word_counter)} words"


class Word:
    def __init__(self, sequence):
        self.sequence = sequence  # sequence of words (word1, word2, word3)
        self.seq_no_caps = [remove_cap(w).lower() for w in sequence]
        self.no_caps = "".join(self.seq_no_caps)

    def distance_to(self, other):
        # isinstance(seq)
        alignment_result = align(
            self.no_caps, other.no_caps,
            match_cost=0, gap_cost=1, mismatch_cost=1, 
            decision_func=min
        )
        return alignment_result
    def __len__(self):
        return len(self.no_caps)

    def __str__(self):
        return f"{self.sequence}, {self.no_caps}"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if len(self.sequence) != len(other.sequence):
            return False
        for x, y in zip(self.sequence, other.sequence):
            if x != y:
                return False
        return True

    @property
    def bow(self):
        return Counter(self.no_caps)
    
    @property
    def tokens_no_caps(self):
        return [token for word in self.seq_no_caps for token in word2tokens(word)]

    @property
    def tokens(self):
        return [token for word in self.sequence for token in word2tokens(word)]
        


class AdvancedVocab:
    def __init__(self):
        self.clear()

    def clear(self):
        self.nocaps2index = defaultdict(int)
        self.index2seq = defaultdict(set)
        self.words = []
        self.word2index = dict()
        self.bows = []
        self.bow_letters = set()
        self.letter2index = dict()
        self.search_tree = None

    def get_words(self, n):
        for w in self.words:
            if len(w) == n:
                yield w

    def append(self, word_seq):
        word_seq = tuple(word_seq)
        if word_seq in self.word2index:
            word_index = self.word2index[word_seq]
            word = self.words[word_index]
        else:
            word = Word(word_seq)
            self.word2index[word_seq] = len(self.words)
            self.words.append(word)
            word_index = self.word2index[word.sequence]  # len(self.words)
        if not word.no_caps in self.nocaps2index:
            self.nocaps2index[word.no_caps] = len(self.nocaps2index)
            bow = word.bow
            self.bows.append(bow)
            self.bow_letters.update(list(bow))
        index = self.nocaps2index[word.no_caps]
        # if word.sequence not in self.word2index:
        # word_index = self.word2index[word.sequence]  # len(self.words)
        
        self.index2seq[index].add(word_index)
        
    def precompute(self):
        """
        builds vector-based nearest neighbours structure 
        to search similar sequences more effectively
        """
        self.n_letters = len(self.bow_letters)
        self.letter2index = {
            letter: index 
            for index, letter in enumerate(sorted(self.bow_letters))
        }
        # make sparse matrix out of the bows in vocab
        # size of sparse matrix is len(totalwords) * len(bow_letters)
        # TODO: make this sparse (maybe later)
        self.n_words = len(self.bows)
        # print(len(self.n_words))
        self.count_vectors = np.zeros((self.n_words, self.n_letters), dtype=int)
        for i, bow in enumerate(self.bows):
            vector = self.bow2vector(bow)
            self.count_vectors[i, :] = vector[:]

        self.search_tree = BallTree(self.count_vectors, metric="manhattan")
    
    def bow2vector(self, bow):
        return bow2vector(bow, self.letter2index, self.n_letters)

    def find(self, word_seq, verbose=False):
        word = Word(word_seq)
        if verbose:
            print("input seq:", word)
        indices = []
        if not word.no_caps in self.nocaps2index:
            if verbose:
                print(f"{word.no_caps} not in nocaps2index")
            return indices
        
        index = self.nocaps2index[word.no_caps]
        if verbose:
            print("Checking index:", index)
        for word_index in self.index2seq[index]:
            if verbose:
                print("word_index=", word_index)
            if self.words[word_index] == word:
                indices.append(word_index)
        return indices

    def find_nearby(self, query_str, max_rad=10, use_exact_rad=False, threshold=0.4, verbose=False):
        #if isinstance(query_str, str) or isinstance(query_str, tuple):
        word = Word(query_str)
        #else:
        #    word = query_str
        
        query_vector = self.bow2vector(word.bow)  # , self.letter2index, self.n_letters)
        d_max = query_vector.sum()
        if use_exact_rad:
            rad = max_rad
        else:
            rad = min(max(int(d_max*threshold), 1), max_rad)

        if verbose:
            print("nearby radius used:", rad)
        indices, distances = self.search_tree.query_radius(
            query_vector.reshape(1, -1),
            r=rad,
            return_distance=True,
            sort_results=False # we don't need to sort this, since next postprocess it next
        )
        
        if verbose:
            print(indices, distances)
        # all_results = []
        for ind, dist in zip(indices, distances):
            result = [(self.words[word_index], d) for i, d in zip(ind, dist) for word_index in self.index2seq[i]]
            # print([(str(w), d)for w, d in result])
            # all_results.append(result)
            return result
        # return all_results

    def find_closest(self, query_str, verbose=False):
        word = Word(query_str)
        query_vector = self.bow2vector(word.bow)
        distances, indices = self.search_tree.query(query_vector.reshape(1, -1))
        # print(indices, distances)
        rad = np.min(distances)
        # print("find closest", query_str, rad, word)
        return self.find_nearby(
            query_str,
            max_rad=rad,
            use_exact_rad=True,
            verbose=verbose)
        