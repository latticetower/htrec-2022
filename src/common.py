from dataclasses import dataclass
import unicodedata
from itertools import islice
from collections import deque
import numpy as np
import re
word_regex = re.compile("\W+")

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