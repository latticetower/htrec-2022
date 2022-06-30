from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

from datastruct import AdvancedVocab
from datastruct import Word
from datastruct import align_spaces
from common import sliding_window

def get_closest_data(mt_word, vocabs, vocabs_used=None):
    closest_data = []
    min_distance = np.inf
    if vocabs_used is None:
        vocabs_used = list(vocabs)
    for k in vocabs_used:
        # print("vocab", k, mt_word)
        for word, dist in vocabs[k].find_nearby(mt_word.sequence, max_rad=10, verbose=False):
            # print(word, dist)
            result = mt_word.distance_to(word)
            # result = align(mt_word.no_caps, word.no_caps)
            # print(word, mt_word, result.distance)
            if result.distance < min_distance:
                min_distance = result.distance
                closest_data = []
            if result.distance == min_distance:
                closest_data.append((result, k, word))
        if k == 1:
            # print(mt_words[2])
            for word, dist in vocabs[k].find_closest(mt_word.sequence):
                result = mt_word.distance_to(word)
                # if result.distance < min_distance:
                #     min_distance = result.distance
                # closest_data = []
                # if result.distance == min_distance:
                closest_data.append((result, k, word))

    return min_distance, closest_data


def build_path_matrix(mt_words, vocabs, max_split_size=4, cutoff=None, equal_length=False, verbose=False):
    N = len(mt_words)
    dmatrix = []  # defaultdict(list)
    dmatrix.append((0, []))  # at the beginning the edit distance is zero
    # b = 0
    # e = 1

    # ht_words, mt_words, mt_word.sequence
    for i in range(len(mt_words)):
        if verbose:
            print(i, "last word")
        e = i+1
        candidates = []
        for nwords in range(1, max_split_size + 1):
            b = e - nwords
            if b < 0:
                continue
            # print(b, e, dmatrix[b])
            mt_word = Word(mt_words[b:e])
            dist, closest_data = get_closest_data(mt_word, vocabs)
            if equal_length:
                closest_data = [(ar, k, w) for ar, k, w in closest_data if len(w) == len(mt_word)]
            if verbose:
                print(dist, closest_data)
            if dist < np.inf and len(closest_data) > 0:    
                candidates.append((dmatrix[b][0] + dist, (dist, b, e), closest_data))
            # print(dist)
        # print([c[:2] for c in candidates])
        if len(candidates) > 0:
            min_dist = min([d for d, x, cd in candidates])
            data = [(x, cd) for d, x, cd in candidates if d == min_dist]
        else:
            # TODO: use the word as it is also with distance equal to the minimal one.
            # use last word as it is
            # min_dist = 0
            min_dist = dmatrix[i][0] + 0
            # candidates.append()
            mt_word = Word(mt_words[i:i+1])
            result = mt_word.distance_to(mt_word)
            data = [((0, i, i + 1), [(result, -1, mt_word)])]

        # print(min_dist)
        dmatrix.append((min_dist, data))
    return dmatrix

def build_split_matrix(mt_words, vocabs, max_split_size=1, verbose=False, cutoff=2, equal_length=True):
    N = len(mt_words)
    dmatrix = []
    for i in range(N):
        dmatrix.append(mt_words[i])
        # at the beginning the edit distance is zero
    # b = 0
    # e = 1
    vocabs_used = [i for i in vocabs if i > 1]

    # ht_words, mt_words, mt_word.sequence
    for i in range(N):
        if verbose:
            print(i, "last word")
        # candidates = []
        for i, word in enumerate(mt_words):
            mt_word = Word(word)
            
            dist, closest_data = get_closest_data(mt_word, vocabs, vocabs_used=vocabs_used)
            if equal_length:
                closest_data = [(ar, k, w) for ar, k, w in closest_data if len(w) == len(mt_word)]
            if len(word) >= cutoff * 3:
                closest_data = [(ar, k, w) for ar, k, w in closest_data if ar.distance <= cutoff]
            else:
                closest_data = [(ar, k, w) for ar, k, w in closest_data if ar.distance <= 0]
            if verbose:
                print(dist, closest_data)
            if dist <= np.inf:
                # callable(obj)
                if len(closest_data) > 0:
                    dmatrix[i] = closest_data
        # for nwords in range(1, max_split_size + 1):
        #    b = e - nwords
        #    if b < 0:
        #        continue
        #    # print(b, e, dmatrix[b])
        #    mt_word = Word(mt_words[b:e])
        #    dist, closest_data = get_closest_data(mt_word, vocabs)
        #    if verbose:
        #        print(dist, closest_data)
        #    if dist < np.inf:    
        #        candidates.append((dmatrix[b][0] + dist, (dist, b, e), closest_data))
        #    # print(dist)
        # print([c[:2] for c in candidates])
        # if len(candidates) > 0:
        #     min_dist = min([d for d, x, cd in candidates])
        #     data = [(x, cd) for d, x, cd in candidates if d == min_dist]
        # else:
        #     # TODO: use the word as it is also with distance equal to the minimal one.
        #     # use last word as it is
        #     # min_dist = 0
        #     min_dist = dmatrix[i][0] + 0
        #     # candidates.append()
        #     result = mt_word.distance_to(mt_word)
        #     data = [((0, i, i + 1), [(result, -1, mt_word)])]
        # if len(candidates) > 0:
        #     data = [(x, cd) for d, x, cd in candidates]

        #     # print(min_dist)
        #     dmatrix.append((0, data))
        # else:
        #     dmatrix.append((0, []))
    return dmatrix



# using dmatrix we build graph and find all paths from last to first words
# dmatrix[-1]
def extract_paths(dmatrix):
    n = len(dmatrix) - 1
    # path_start, [path segments]
    finished_paths = []
    unfinished_paths = defaultdict(list)
    unfinished_paths[n] = []
    while len(unfinished_paths) > 0:
        new_paths = defaultdict(list)
        for start_vertex, suffixes in unfinished_paths.items():
            min_dist, path_fragments = dmatrix[start_vertex]
            for (frag_dist, b, e), segs in path_fragments:
                if b >= start_vertex:
                    print("got some sort of loop")
                    continue
                if len(suffixes) == 0:
                    new_suffixes = [[(frag_dist, b, e, segs),]]
                else:
                    new_suffixes = [[(frag_dist, b, e, segs),] + suffix for suffix in suffixes]
                if b == 0:
                    finished_paths.extend(new_suffixes)
                else:
                    new_paths[b] = new_suffixes                

        unfinished_paths = new_paths
        # break
    return finished_paths

# finished_paths = extract_paths(dmatrix)

# using dmatrix we build graph and find all paths from last to first words
# dmatrix[-1]
def extract_splitted_paths(dmatrix):
    n = len(dmatrix)
    all_prefixes = []
    for i in range(n):
        temp_prefixes = []
        # if dmatrix[i+1]
        word = dmatrix[i]
        if isinstance(word, str):
            if len(all_prefixes) == 0:
                temp_prefixes = [[word]]
            else:
                temp_prefixes = [a + [word] for a in all_prefixes]
        else:
            for alignment_result, distance, vocab_word in dmatrix[i]:
                if len(all_prefixes) == 0:
                    temp_prefixes.append([(alignment_result, vocab_word)])
                else:
                    temp_prefixes.extend([a + [(alignment_result, vocab_word)] for a in all_prefixes])
        all_prefixes = temp_prefixes
    return all_prefixes
# finished_paths = extract_paths(dmatrix)


# alignment = result.do_backtracing("".join(mt_word.sequence), word.tokens)
# mt, rt, m = list(zip(*alignment))
# # print("aligned:", ["".join(word) for word in ])
# print("aligned:")
# for word, ref in align_spaces(mt, rt):
#     print("".join(word), "<-", "".join([x.char for x in ref]))

# finished_paths[2][0][-1]
# vocabs[1].nocaps2index["της"]
def resplit_with_refs(path, mt_words, return_spaces=False):
    all_splits = []
    all_space_positions = []
    # bsp = mt_spaces[0]
    # all_splits = [(bsp, bsp)] # don't replace space with anything
    for (frag_dist, b, e, segs) in path:
        # space_positions.append(e)
        space_positions = []
        # print(f"{b} -> {e}, dist={frag_dist}, {len(segs)} variants", mt_words[b:e])
        mt_fragment = Word(mt_words[b:e])
        corrected_fragments = set()
        for alignment_result, k, ref_word in segs:
            alignment = alignment_result.do_backtracing("".join(mt_fragment.sequence), ref_word.tokens)
            mt, rt, m = list(zip(*alignment))
            # print(mt, rt)
            corrected_words = tuple([
                ("".join(word), "".join([x.char for x in ref]))
                for word, ref in align_spaces(mt, rt)])
            # print("".join(word), "<-", "".join([x.char for x in ref]))
            corrected_fragments.add(corrected_words)
        # print(corrected_fragments)
        new_splits = []
        for frag in corrected_fragments:
            n_words = len(frag)
            if len(all_splits) == 0:
                # print(frag)
                new_splits.append(frag)
            else:
                for prefix in all_splits:
                    new_splits.append(prefix + frag)
                    # print(len(new_splits[0]), new_splits[0])
            space_frag = [0]*(n_words - 1)
            if len(all_space_positions) == 0:
                space_positions.append(space_frag)
            else:
                for prefix in all_space_positions:
                    space_positions.append(prefix + space_frag)

        all_splits = new_splits
        all_space_positions = [sp+[e] for sp in space_positions]
    if return_spaces:
        return all_splits, all_space_positions
    return all_splits
    # # print("splits:")
    # grouped_splits = defaultdict(set)
    # for spl in set(all_splits):
    #     word_split, refs = list(zip(*spl))
    #     grouped_splits[word_split].add(refs)

    # grouped_splits = {k: [tuple(set(w)) for w in list(zip(*list(v)))] for k, v in grouped_splits.items()}
    # return grouped_splits


def resplit_paths(paths, mt_words, mt_spaces):
    all_splits = dict()
    for path in paths:
        path_splits, space_positions = resplit_with_refs(path, mt_words, return_spaces=True)
        # print(path_splits)
        for path_split, sp in zip(path_splits, space_positions):
            all_splits[path_split] = sp
    grouped_splits = defaultdict(set)
    spaces_dict = dict()
    for spl, space_positions in all_splits.items():
        # print(spl, space_positions)
        word_split, refs = list(zip(*spl))
        # print(word_split, space_positions)
        spaces_after = [mt_spaces.get(s, " ") for s in space_positions]
        #word_split =  + [w + space for w, space in zip(word_split, spaces_after)]
        #word_split = tuple(word_split)
        grouped_splits[word_split].add(refs)
        spaces_dict[word_split] = spaces_after

    grouped_splits = {k: [tuple(set(w)) for w in list(zip(*list(v)))] for k, v in grouped_splits.items()}
    return grouped_splits, spaces_dict


def resplit_splitted_paths(paths, mt_words, mt_spaces):
    all_splits = defaultdict(set)
    spaces_dict = dict()
    # print(mt_spaces)
    for path in paths:
        spaces_after = []
        # print("path", path)
        fixed_words = []
        replacements = []
        for i, replacement_word in enumerate(path):
            # print(i, replacement_word)
            n_words = 0
            new_word = []
            ref_word = []
            if isinstance(replacement_word, str):
                # not changed
                fixed_words.append(replacement_word)
                replacements.append(replacement_word)
            else:
                (alignment_result, vocab_word) = replacement_word
                # print("alignment_result", alignment_result)
                
                alignment = alignment_result.do_backtracing(mt_words[i], vocab_word.tokens)
                mt, rt, m = list(zip(*alignment))
                # print(mt, rt)
                corrected_words = tuple([
                    ("".join(word), "".join([x.char for x in ref]))
                    for word, ref in align_spaces(mt, rt)])
                # corrected_words = [(w, ref) for w, ref in corrected_words if len(w) > 0 ]
                # spaces_after.extend([" "]*(len(corrected_words) - 1))
                
                for w, ref in corrected_words:
                    if len(w) > 0:
                        new_word.append(w)
                        ref_word.append(ref)
                    if len(w) > 0:
                        # spaces_after.append(" ")
                        n_words += 1
                if len(new_word) > 0:
                    new_word = tuple(new_word)
                    ref_word = tuple(ref_word)
                    fixed_words.append(new_word)
                    replacements.append(ref_word)
                # spaces_after = spaces_after[:-1]
                #if n_words > 0:
                #    spaces_after.extend([" "]*(n_words - 1))
            sp_after = mt_spaces.get(i + 1, '')
            if n_words > 0 or isinstance(replacement_word, str):
                spaces_after.append(sp_after)
        fixed_words = tuple(fixed_words)
        spaces_dict[fixed_words] = spaces_after
        all_splits[fixed_words].add(tuple(replacements))

    # grouped_splits = defaultdict(set)
    # spaces_dict = dict()
    # for spl, space_positions in all_splits.items():
    #     word_split, refs = list(zip(*spl))
    #     # print(word_split, space_positions)
    #     # spaces_after = [mt_spaces.get(s, " ") for s in space_positions]
    #     #word_split =  + [w + space for w, space in zip(word_split, spaces_after)]
    #     #word_split = tuple(word_split)
    #     grouped_splits[word_split].add(refs)
    #     spaces_dict[word_split] = spaces_after

    grouped_splits = {
        k: [tuple(set(w)) for w in list(zip(*list(vs)))] 
        for k, vs in all_splits.items()
    }
    return grouped_splits, spaces_dict



# dmatrix = build_path_matrix(mt_words, vocabs)
# finished_paths = extract_paths(dmatrix)
# for k in resplit_paths(finished_paths):
#     variant = " ".join(k)
#     print(variant)


class SpaceFixer:
    def __init__(self, max_words=4, verbose=False):
        self.max_words = max_words
        self.vocabs = {
            int(n): AdvancedVocab()
            for n in np.arange(max_words) + 1
        }
        self.verbose = verbose

    def fill(self, sequences):
        for sequence in tqdm(sequences):
            words = [word for word in sequence if len(word) > 0]
            for n in self.vocabs:
                for word_seq in sliding_window(words, n):
                    # print(word_seq)
                    self.vocabs[n].append(word_seq)
        if self.verbose:
            print("Precompute search trees in vocabs")
        for n in self.vocabs:
            self.vocabs[n].precompute()

    def resplit(self, sequence, spaces=dict(), max_split_size=4, cutoff=None):
        self.dmatrix = build_path_matrix(sequence, self.vocabs, max_split_size=max_split_size, equal_length=True)
        # return
        finished_paths = extract_paths(self.dmatrix)
        resplit_dict, spaces_dict = resplit_paths(finished_paths, sequence, spaces)
        for k, v in resplit_dict.items():
            # variant = " ".join(k)
            assert len(k) == len(spaces_dict[k])
            yield k, v, spaces_dict[k]

    def split_words(self, sequence, spaces=dict(), max_split_size=4, cutoff=2, verbose=False):
        """ Returns: tuples of [words], [replacement lists], spaces after each word
        """
        self.split_matrix = build_split_matrix(sequence, self.vocabs, max_split_size=max_split_size, cutoff=cutoff)
        finished_paths = extract_splitted_paths(self.split_matrix)
        if verbose:
            print("finished_paths", finished_paths)
            print("sequence", sequence)
            print("spaces", spaces)
        resplit_dict, spaces_dict = resplit_splitted_paths(finished_paths, sequence, spaces)
        for k, v in resplit_dict.items():
            # variant = " ".join(k)
            # print("resplit", k, v, spaces, sequence, spaces_dict[k])
            assert len(k) == len(spaces_dict[k])
            yield k, v, spaces_dict[k]


    