import argparse
import os
import glob
import numpy as np
import pickle
import errno
import json
from tqdm import tqdm
from torchtext.vocab import build_vocab_from_iterator

def prep_files(dataset_root, output_dir):

    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(val_dir, exist_ok=True)
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    print("Preprocessing...")

    total_count = 0
    train_count = 0
    val_count = 0
    test_count = 0

    events = set()

    np.random.seed(0)

    for f_name in tqdm(glob.glob(os.path.join(dataset_root, "**/*symbol_key.json"), recursive=True)):
        split_type = np.random.choice(
            ["train", "val", "test"], p=[0.7, 0.15, 0.15])
        if split_type == "train":
            o_file = f_name.replace('/event/', '/mel-chords/train/')
            train_count += 1
        elif split_type == "val":
            o_file = f_name.replace('/event/', '/mel-chords/val/')
            val_count += 1
        elif split_type == "test":
            o_file = f_name.replace('/event/', '/mel-chords/test/')
            test_count += 1
        else:
            print("ERROR: Unrecognized split type:", split_type)
            return False
        total_count += 1

        encoded = _encode_midi_file(f_name)
        events.update(encoded)
        # break

        # Creat the dir if it doesn't already exist
        if not os.path.exists(os.path.dirname(o_file)):
            try:
                os.makedirs(os.path.dirname(o_file))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(o_file, "wb") as o_stream:
            pickle.dump(o_file, o_stream)

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
    with open('./utilities/music_vocab.pkl', 'wb') as f:
        pickle.dump(music_vocab, f)
    return True

def _encode_midi_file(f_name):
    with open(f_name) as f:
        json_dict = json.load(f)
        meta = json_dict['metadata']
        bpm = float(meta.get('BPM'))
        if bpm == 0:
            bpm = 120

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

        cur_time = 0
        encoded = []
        for event in events:
            if cur_time != event[1]:
                time = _beats_to_time(event[1] - cur_time, bpm)
                while time > 1000:
                    encoded.append('TS<1000>')
                    time -= 1000
                encoded.append('TS<' + str(time) + '>')
                cur_time = event[1]
            encoded.append(event[2] + '<' + str(event[0]) + '>')
    return encoded

# Converts the number of beats to the rounded wall time (to the nearest
# ms) given the provided bpm
def _beats_to_time(beats, bpm) -> int:
    return int(round(beats * 60 / bpm * 100)) * 10

def parse_args():
    """
    Parse the arguments to the function
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_root", type=str, default="./dataset/event",
                        help="Root folder for the dataset")
    parser.add_argument("-output_dir", type=str, default="./dataset/mel-chords",
                        help="Output folder to put the preprocessed midi into")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir

    print("Preprocessing midi files and saving to", output_dir)
    prep_files(data_root, output_dir)
    print("Done!\n")
