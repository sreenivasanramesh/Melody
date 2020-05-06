#import os
import music21
import numpy as np
from math import floor
from pathlib import Path


sampling_frequency = 4  # have to test with 16
note_range = 62  # 62 keys of piano instead of 88
note_offset = 33


class Decode(object):
    """
    Decodes a translated score to a midi stream.
    translate_to_midi - converts a translated score into a midi stream
    generate_midi - takes a translated score and generates a midi file
    """

    @staticmethod
    def translate_to_midi(score, sample_freq, note_offset):
        speed = 1.0 / sample_freq
        notes = list()
        time_offset = 0

        # convert to midi writable format and add to notes list
        for i in range(len(score)):
            if score[i] in ["", " "]:
                continue
            if score[i].startswith("end"):
                continue
            elif score[i].startswith("wait"):
                time_offset += int(score[i][4:])
                continue
            else:
                # Look ahead to see if another end was generated soon after
                duration = 1
                has_end = False
                note_string_len = len(score[i])
                for j in range(1, 200):
                    if i + j == len(score):
                        break
                    if score[i + j].startswith("wait"):
                        duration += int(score[i + j][4:])
                    if (score[i + j][:3 + note_string_len] == "end" + score[i]) or (score[i + j][:note_string_len] == score[i]):
                        has_end = True
                        break

                if not has_end:
                    duration = 12
                add_wait = 0

                new_note = music21.note.Note(int(score[i]) + note_offset)
                new_note.duration = music21.duration.Duration(duration * speed)
                new_note.offset = time_offset * speed
                notes.append(new_note)
                time_offset += add_wait

        piano = music21.instrument.fromString("Piano")
        notes.insert(0, piano)
        piano_stream = music21.stream.Stream(notes)
        stream = music21.stream.Stream([piano_stream])
        return stream

    @staticmethod
    def generate_midi(generated_vocab, filename):
        midi_stream = Decode().translate_to_midi(generated_vocab, sampling_frequency, note_offset)
        midi_stream.write('midi', filename)


class Encode(object):
    """
    Encodes a midi stream to a translated sequence of key presses.
    """

    @staticmethod
    def get_key_presses(stream):
        """
        Returns an array of arrays of keys that are pressed at a time step.
        A pressed key is encoded as a 1, a held key as a 2; else a 0
        :param stream: A midi stream
        :return:
        """
        # length of notes with sampling
        song_length = floor(stream.duration.quarterLength * sampling_frequency) + 1
        score_arr = np.zeros((song_length, note_range))  # [ [[key=1 if press, 2 if hold else 0]]   [[]]  [[]]  ....]

        notes = list()  # will be a list of lists [ [note, timestamp, duration of note] ... ]

        note_filter = music21.stream.filters.ClassFilter('Note')
        chord_filter = music21.stream.filters.ClassFilter('Chord')

        for note in stream.recurse().addFilter(note_filter):
            #  pitch.midi - piano key no  , note_offset - since we are restricting to 62 keys
            #  n.offset = position of the note from the start of the section
            #  n.offset * sampling_frequency = position on music sheet
            #  n.duration.quarterLength = "musical seconds" note is played
            #  multiplying quarter length by sampling frequency gives us the length the note should be played
            notes.append((note.pitch.midi - note_offset, floor(note.offset * sampling_frequency),
                          floor(note.duration.quarterLength * sampling_frequency)))

        for chord in stream.recurse().addFilter(chord_filter):
            #  chord.pitches - notes in a chord
            for note in chord.pitches:
                notes.append((note.midi - note_offset, floor(chord.offset * sampling_frequency),
                              floor(chord.duration.quarterLength * sampling_frequency)))

        for note in notes:
            key = note[0]
            #  reduce the range to 62 keys by move up or down two octaves
            while key < 0:
                key += 12
            while key >= note_range:
                key -= 12

            score_arr[note[1], key] = 1  # Strike note
            score_arr[note[1]+1: note[1]+note[2], key] = 2  # Continue holding note

        score_string = list()
        # converts array of keys pressed/held to string and puts p before
        # ['p00000000000000000000000000000000000000000000001000000000000000',
        # 'p00000000000000000000000000000000000000000000002000000000000000', ...]

        for time_step in score_arr:
            score_string.append(''.join(str(int(x)) for x in time_step))

        return score_string

    @staticmethod
    def augmentation(score_string_arr):
        """
        Move notes one octave up 12 times - increases length of ordered song 12 times
        :param: string - score_string_arr: output from get_key_presses
        :return: list - augmented_data: the augmented score
        """
        augmented_data = list()
        for i in range(0, 12):
            for chord in score_string_arr:
                padded = '000000' + chord + '000000'
                augmented_data.append(padded[i: i+note_range])
        return augmented_data

    @staticmethod
    def get_key_sequence(score):
        """
        Translates the augmented score to sequence of key presses and releases.
        A number indicates the piano key to be pressed,
        and end<number> represents the corresponding key release.
        Wait<num> are lengths of time nothing is being played.
        :param score: augmented_data from augmentation
        :return: string of key sequences
        """
        processed_score = list()
        for time_step in range(len(score)):
            chord = score[time_step]
            try:
                next_chord = score[time_step + 1]
            except IndexError:
                next_chord = ""

            prefix = chord[0]
            for i in range(len(chord)):
                if chord[i] == "0":
                    continue
                key_position = str(i)  # piano key position
                if chord[i] == "1":
                    processed_score.append(key_position)  # newly played key position
                # if chord[i]=="2" do nothing, we're continuing to hold the note
                # unless next_chord[i] is back to 0 then end note
                if next_chord == "" or next_chord[i] == "0":
                    processed_score.append("end" + key_position)

            if prefix.isdigit():
                processed_score.append("wait")
        # processed_score will be ike ['end49', 'wait', '52', 'wait', 'wait', 'wait', 'end52', 'wait', '44', 'wait'...

        i = 0
        translated_score = ""
        # merge multiple waits
        while i < len(processed_score):
            wait_count = 1
            if processed_score[i] == 'wait':
                while wait_count <= sampling_frequency * 2 and i + wait_count < len(processed_score) and processed_score[i + wait_count] == 'wait':
                    wait_count += 1
                processed_score[i] = 'wait' + str(wait_count)
            translated_score += processed_score[i] + " "
            i += wait_count
        return translated_score


def main():
    working_dir = Path.cwd()
    data_dir = working_dir/"data"
    result_dir = data_dir/"processed_data"

    for file1 in data_dir.glob("*.mid"):

        print("Processing File ", file1)
        midi_file = music21.midi.MidiFile()
        midi_file.open(file1)
        midi_file.read()
        midi_file.close()

        midi_stream = music21.midi.translate.midiFileToStream(midi_file)

        score_keys = Encode().get_key_presses(midi_stream)
        augmented_score = Encode().augmentation(score_keys)  # augments data by shifting it up an octave multiple times
        processed_score = Encode().get_key_sequence(augmented_score)

        file_name = file1.stem + ".txt"
        result_file = result_dir/file_name
        f = open(result_file, "w+")
        f.write(processed_score)
        f.close()
        print("Generated ", result_file)


if __name__ == "__main__":
    main()
