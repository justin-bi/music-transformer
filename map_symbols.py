import argparse
import glob
import os
import re
import json
import pickle as pkl
from collections import defaultdict, Counter
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_root", type=str, default="./dataset/event",
                        help="Root folder for the dataset")

    return parser.parse_args()

def _pitch_up(chord, pitch_amt):
    new_chord = list(chord)
    alter_idxs = []

    for _ in range(pitch_amt):
        prev_chord = ''.join(new_chord)

        alter_idxs = []
        # C
        idxs = [i.start() for i in re.finditer('C(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'D'
            alter_idxs.append((idx + 1, 'b'))

        # Db
        idxs = [i.start() for i in re.finditer('Db', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'D'
            alter_idxs.append((idx + 1, 'del'))

        # D
        idxs = [i.start() for i in re.finditer('D(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'E'
            alter_idxs.append((idx + 1, 'b'))

        # Eb
        idxs = [i.start() for i in re.finditer('Eb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'E'
            alter_idxs.append((idx + 1, 'del'))

        # E
        idxs = [i.start() for i in re.finditer('E(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'F'

        # F
        idxs = [i.start() for i in re.finditer('F(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'G'
            alter_idxs.append((idx + 1, 'b'))

        # Gb
        idxs = [i.start() for i in re.finditer('Gb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'G'
            alter_idxs.append((idx + 1, 'del'))

        # G
        idxs = [i.start() for i in re.finditer('G(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'A'
            alter_idxs.append((idx + 1, 'b'))

        # Ab
        idxs = [i.start() for i in re.finditer('Ab', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'A'
            alter_idxs.append((idx + 1, 'del'))

        # A
        idxs = [i.start() for i in re.finditer('A(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'B'
            alter_idxs.append((idx + 1, 'b'))

        # Bb
        idxs = [i.start() for i in re.finditer('Bb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'B'
            alter_idxs.append((idx + 1, 'del'))

        # B
        idxs = [i.start() for i in re.finditer('B(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'C'

        # -----------------------------------------------------------------------------
        # LOWERCASE
        # -----------------------------------------------------------------------------

        # c
        idxs = [i.start() for i in re.finditer('c(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'd'
            alter_idxs.append((idx + 1, 'b'))

        # db
        idxs = [i.start() for i in re.finditer('db', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'd'
            alter_idxs.append((idx + 1, 'del'))

        # d
        idxs = [i.start() for i in re.finditer(
            '(?<![ad])d(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'e'
            alter_idxs.append((idx + 1, 'b'))

        # eb
        idxs = [i.start() for i in re.finditer('eb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'e'
            alter_idxs.append((idx + 1, 'del'))

        # e
        idxs = [i.start() for i in re.finditer('e(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'f'

        # f
        idxs = [i.start() for i in re.finditer('f(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'g'
            alter_idxs.append((idx + 1, 'b'))

        # gb
        idxs = [i.start() for i in re.finditer('gb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'g'
            alter_idxs.append((idx + 1, 'del'))

        # g
        idxs = [i.start() for i in re.finditer('g(?![#b])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'a'
            alter_idxs.append((idx + 1, 'b'))

        # ab
        idxs = [i.start() for i in re.finditer('ab', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'a'
            alter_idxs.append((idx + 1, 'del'))

        # a
        idxs = [i.start() for i in re.finditer('a(?![#bdj])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'b'
            alter_idxs.append((idx + 1, 'b'))

        # bb
        idxs = [i.start() for i in re.finditer('bb', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'b'
            alter_idxs.append((idx + 1, 'del'))

        # b
        idxs = [i.start() for i in re.finditer(
            '(?![A-G|a|c-g])b(?![#b|0-9])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'c'

        for idx, val in sorted(alter_idxs, reverse=True):
            if val == 'del':
                del new_chord[idx]
            else:
                new_chord.insert(idx, val)

    return ''.join(new_chord)


def main(data_root):
    d = defaultdict(Counter)
    for f_name in tqdm(glob.glob(os.path.join(data_root, "**/*symbol_key.json"), recursive=True)):
        with open(f_name) as f:
            json_dict = json.load(f)
            cur_set = set()
            for chord in json_dict['tracks']['chord']:
                if chord:
                    cur_symbol = chord['symbol']
                    cur_comp = chord['composition']
                    # Add the chord translations, including all transpositions
                    for _ in range(12):
                        cur_symbol = _pitch_up(cur_symbol, 1)
                        minim = min(cur_comp)
                        for i in range(len(cur_comp)):
                            cur_comp[i] += 1
                            if minim >= 12:
                                cur_comp[i] %= 12
                        if tuple(cur_comp) not in cur_set:
                            d[cur_symbol] += Counter([tuple(cur_comp)])
                            cur_set.add(tuple(cur_comp))

    for k in sorted(d.keys()):
        print((k + ": ").ljust(20), max(d[k], key=d[k].get))

    with open('chord_to_notes_dict.pkl', 'wb') as f:
        pkl.dump(d, f)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_root)
