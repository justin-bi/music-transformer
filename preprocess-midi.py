import argparse
import os
import glob
import numpy as np
import pickle
import errno
import json
import re
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator

def prep_files(dataset_root, output_dir, pitch_augment, time_augment):

    # Create the directories to house the splits
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    total_count = 0
    train_count = 0
    val_count = 0
    test_count = 0

    events = set()

    np.random.seed(0)
    file_encodings_list = []

    num_pitches = 12 if pitch_augment else 1
    time_stretch = 0
    augment_amt = round(time_augment, 1)
    if 0 < augment_amt < 1:
        time_stretch = augment_amt
    time_range = np.arange(1 - time_stretch, 1 + time_stretch, 0.1)
    # If time_stretch = 0, the above gives empty list for some reason
    if len(time_range) == 0:
        time_range = [1.0]
    augment_count = num_pitches * len(time_range)

    for f_name in tqdm(glob.glob(os.path.join(dataset_root, "**/*symbol_key.json"), recursive=True)):
        # First get the split type, and from that determine which directory to add it to
        split_type = np.random.choice(
            ["train", "val", "test"], p=[0.7, 0.15, 0.15])
        if split_type == "train":
            o_file = f_name.replace('/event/', '/mel-chords/train/')
            train_count += augment_count
        elif split_type == "val":
            o_file = f_name.replace('/event/', '/mel-chords/val/')
            val_count += augment_count
        elif split_type == "test":
            o_file = f_name.replace('/event/', '/mel-chords/test/')
            test_count += augment_count
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False
        total_count += augment_count

        # All augmentations will go into the same split, just to speed things up a bit
        for pitch in range(num_pitches):
            for stretch in time_range:
                st = round(stretch, 1)
                iter_o_file = o_file.replace('_symbol_key.json', '/p=' + str(pitch) + '&t=' + str(st))

                # If we want to do pitch augmentation, then we'll cycle through all of the
                _encode_midi_file(f_name, num_pitches, time_range, events, file_encodings_list, iter_o_file)

    print("Num Total:", total_count)
    print("Num Train:", train_count)
    print("Num Val:", val_count)
    print("Num Test:", test_count)

    # Setup the vocab that'll be used by the models
    temp = [[w] for w in events]
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    music_vocab = build_vocab_from_iterator(temp,
                                            min_freq=1,
                                            specials=special_symbols,
                                            special_first=True)
    music_vocab.set_default_index(0)

    for o_file, encoded in tqdm(file_encodings_list):
        # Create the dir if it doesn't already exist
        if not os.path.exists(os.path.dirname(o_file)):
            try:
                os.makedirs(os.path.dirname(o_file))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with open(o_file, "wb") as o_stream:
            pickle.dump(music_vocab(encoded), o_stream)

    with open('./utilities/music_vocab.pkl', 'wb') as f:
        pickle.dump(music_vocab, f)

    return True

def _encode_midi_file(f_name, num_pitches, time_range, event_set, file_list, iter_o_file):
    with open(f_name) as f:
        json_dict = json.load(f)

        events = []
        for melody_note in json_dict['tracks']['melody']:
            if not melody_note:
                continue
            midi = int(melody_note['pitch'] + 60)
            events.append((midi, melody_note['event_on'], 'NON'))
            events.append((midi, melody_note['event_off'], 'NOFF'))

        for chord in json_dict['tracks']['chord']:
            if not chord:
                continue
            events.append((chord['symbol'], chord['event_on'], 'CHON'))
            events.append((chord['symbol'], chord['event_off'], 'CHOFF'))

        events.sort(key=lambda i: (i[1], i[2]))

        meta = json_dict['metadata']
        bpm = float(meta.get('BPM'))
        if bpm == 0:
            bpm = 120

        for pitch in range(num_pitches):
            for stretch in time_range:
                st = round(stretch, 1)
                cur_time = 0
                encoded = []
                for event in events:
                    if cur_time != event[1]:
                        time = _beats_to_time(event[1] - cur_time, bpm * st)
                        while time > 1000:
                            encoded.append('TS<1000>')
                            time -= 1000
                        encoded.append('TS<' + str(time) + '>')
                        cur_time = event[1]
                    val = event[0]
                    if isinstance(event[0], int):
                        val += pitch
                    else:
                        val = _pitch_up(val, pitch)
                    encoded.append(event[2] + '<' + str(val) + '>')
                event_set.update(encoded)
                file_list.append(iter_o_file, encoded)
    return True

# Converts the number of beats to the rounded wall time (to the nearest
# ms) given the provided bpm
def _beats_to_time(beats, bpm) -> int:
    return int(round(beats * 60 / bpm * 100)) * 10

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
        idxs = [i.start() for i in re.finditer('(?<![ad])d(?![#b])', prev_chord)]
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
        idxs = [i.start() for i in re.finditer('(?![A-G|a|c-g])b(?![#b|0-9])', prev_chord)]
        for idx in reversed(idxs):
            new_chord[idx] = 'c'

        for idx, val in sorted(alter_idxs, reverse=True):
            if val == 'del':
                del new_chord[idx]
            else:
                new_chord.insert(idx, val)

    return ''.join(new_chord)

def parse_args():
    """
    Parse the arguments to the function
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_root", type=str, default="./dataset/event",
                        help="Root folder for the dataset")
    parser.add_argument("-output_dir", type=str, default="./dataset/mel-chords",
                        help="Output folder to put the preprocessed midi into")
    parser.add_argument("-pitch_augment", type=bool, default=False,
                        help="If True, augments the song to all keys")
    parser.add_argument("-time_augment", type=float, default=0,
                        help="A value in (0, 1), dictates how much to timeshift in both directions")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    pitch_augment = args.pitch_augment
    time_augment = args.time_augment

    print("Preprocessing midi files and saving to", output_dir)
    prep_files(data_root, output_dir, pitch_augment, time_augment)
    print("Done!\n")
