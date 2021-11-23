# ----------------------------------------------------------------------
# Given a sequence of tokens, return a midi version of it.
# ----------------------------------------------------------------------
import random
import note_seq
import bokeh
import pickle as pkl
from note_seq.protobuf import music_pb2
from bokeh.models import Label

def convert_tokens_to_midi(tokens):
    notes = []
    chords = []
    cur_offset = 0
    for item in enumerate(tokens):
        content = item[item.index('<') + 1: item.index('>')]
        if item.startswith('CHON'):
            chords.append([content, cur_offset, -1])
        elif item.startswith('NON'):
            notes.append([content, cur_offset, -1])
        elif item.startswith('TS'):
            cur_offset += int(content)
        elif item.startswith('CHOFF'):
            changed = False
            for chord in reversed(chords):
                if chord[0] == content and chord[-1] == -1:
                    chord[-1] = cur_offset
                    changed = True
                    break
            if not changed:
                print('CHOFF error')
                print(item)
        elif item.startswith('NOFF'):
            changed = False
            for note in reversed(notes):
                if note[0] == content and note[-1] == -1:
                    note[-1] = cur_offset
                    changed = True
                    break
            if not changed:
                print('NOFF error')
                print(item)
        else:
            print('KEY error')
    return notes, chords

def display_as_midi(notes, chords, weighted=False):
    demo = music_pb2.NoteSequence()

    # Put in the melody notes
    for note in notes:
        if note[-1] != -1:
            demo.notes.add(pitch=int(note[0]), start_time=note[1] / 1000,
                           end_time=note[2] / 1000, velocity=80)

    min_note = int(min(notes, key=lambda i: int(i[0]))[0])

    # Put in the chord notes
    with open("./chord_to_notes_dict.pkl", 'rb') as f:
        chord_to_notes = pkl.load(f)

    max_chord_note = 0
    min_chord_note = 128
    chord_choice = {}
    for chord in chords:
        if chord[0] not in chord_choice:
            choices = chord_to_notes[chord[0]]
            if weighted:
                chord_choice[chord[0]] = random.choices(
                    list(choices.keys()), weights=list(choices.values()))[0]
            else:
                chord_choice[chord[0]] = max(
                    choices.keys(), key=lambda i: choices[i])
            max_chord_note = max(max_chord_note, max(chord_choice[chord[0]]))
            min_chord_note = min(min_chord_note, min(chord_choice[chord[0]]))

    augment = 0
    if min_note > max_chord_note:
        augment = ((min_note - max_chord_note) // 12) * 12

    for chord in chords:
        if chord[-1] != -1:
            for num in chord_choice[chord[0]]:
                demo.notes.add(
                    pitch=num + augment, start_time=chord[1] / 1000, end_time=chord[2] / 1000, velocity=80)

    fig = note_seq.plot_sequence(demo, show_figure=False)

    circle_y = min_chord_note - 6 + augment
    for chord in chords:
        if chord[-1] != -1:
            start = chord[1] / 1000
            end = chord[2] / 1000
            fig.circle(x=start, y=circle_y)
            fig.add_layout(Label(x=(start + end) / 2, y=circle_y, text=chord[0],
                                 text_font_size='10px', x_offset=-6))
            fig.circle(x=end, y=circle_y)
    bokeh.plotting.show(fig)

    note_seq.play_sequence(demo)

def _test():
    notes = [['65', 0, 2580], ['67', 2580, 3550], ['74', 3870, 4190], ['77', 4190, 4510], ['74', 4510, 4830], ['70', 4830, 5150], ['65', 5150, 6760], ['68', 6760, 7410], ['67', 7410, 9020], ['75', 9020, 9340], ['74', 9340, 9660], ['72', 9660, 9980], ['70', 9980, 13200], ['69', 13200, 13520], ['65', 13840, 14160], ['65', 14160, 14480], ['67', 14480, 15130], ['68', 15130, 15450], ['70', 15450, 16740], ['75', 17710, 18030], ['74', 18030, 18350], ['75', 18350, 19620], ['74', 19620, 19940], ['72', 19940, 20260], [
        '70', 20580, 21870], ['74', 21870, 22190], ['77', 22190, 22510], ['77', 22510, 22830], ['77', 22830, 23150], ['75', 23150, 23470], ['70', 23470, 24120], ['72', 24120, 24280], ['68', 24600, 24920], ['68', 24920, 25400], ['68', 25400, 25720], ['72', 25720, 26370], ['75', 27010, 27330], ['74', 27330, 27650], ['72', 27650, 27970], ['70', 28290, 28610], ['74', 28610, 28930], ['67', 29360, 30010], ['72', 30010, 30660], ['79', 31300, 31620], ['75', 31620, 31940], ['74', 31940, 32260], ['72', 32260, 32580], ['70', 32900, 33220]]
    chords = [['Bb', 0, 2580], ['gm', 2580, 5150], ['dm', 5150, 7730], ['Eb', 7730, 10300], ['Eb', 10300, 12880], ['Bb', 12880, 15450], ['gm', 15450, 17390], ['cm', 17390, 18350], [
        'gm', 18350, 20580], ['gm', 20580, 24280], ['Ab', 24600, 26690], ['gm', 26690, 28290], ['gm', 28290, 28930], ['cm', 29360, 30980], ['gm', 30980, 32900], ['gm', 32900, -1]]
    display_as_midi(notes, chords)


if __name__ == "__main__":
    _test()
