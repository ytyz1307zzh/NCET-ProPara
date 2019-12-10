# map state labels to indices
state2idx = {'<PAD>': 0, 'O_C': 1, 'O_D': 2, 'E': 3, 'M': 4, 'C': 5, 'D': 6}
idx2state = {0: '<PAD>', 1: 'O_C', 2: 'O_D', 3: 'E', 4: 'M', 5: 'C', 6: 'D'}

UNK_LOC = -1
NIL_LOC = -2
PAD_LOC = -3

MAX_LOC_CANDS = 45