from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

from datastruct import AdvancedVocab
from datastruct import Word
from datastruct import align_spaces
from common import sliding_window

def get_closest_data(mt_word, vocabs):
    closest_data = []
    min_distance = np.inf
    for k in vocabs:
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
    return min_distance, closest_data


def build_path_matrix(mt_words, vocabs, max_split_size=4, verbose=False):
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
            if dist < np.inf:    
                candidates.append((dmatrix[b][0] + dist, (dist, b, e), closest_data))
            # print(dist)
        # print([c[:2] for c in candidates])
        if len(candidates) > 0:
            min_dist = min([d for d, x, cd in candidates])
            data = [(x, cd) for d, x, cd in candidates if d == min_dist]
        else:
            # use last word as it is
            # min_dist = 0
            min_dist = dmatrix[i][0] + 0
            # candidates.append()
            result = mt_word.distance_to(mt_word)
            data = [((0, i, i + 1), [(result, -1, mt_word)])]

        # print(min_dist)
        dmatrix.append((min_dist, data))
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
        word_split, refs = list(zip(*spl))
        # print(word_split, space_positions)
        spaces_after = [mt_spaces.get(s, " ") for s in space_positions]
        #word_split =  + [w + space for w, space in zip(word_split, spaces_after)]
        #word_split = tuple(word_split)
        grouped_splits[word_split].add(refs)
        spaces_dict[word_split] = spaces_after

    grouped_splits = {k: [tuple(set(w)) for w in list(zip(*list(v)))] for k, v in grouped_splits.items()}
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

    def resplit(self, sequence, spaces=dict(), max_split_size=4):
        dmatrix = build_path_matrix(sequence, self.vocabs, max_split_size=max_split_size)
        finished_paths = extract_paths(dmatrix)
        resplit_dict, spaces_dict = resplit_paths(finished_paths, sequence, spaces)
        for k, v in resplit_dict.items():
            # variant = " ".join(k)
            assert len(k) == len(spaces_dict[k])
            yield k, v, spaces_dict[k]

    