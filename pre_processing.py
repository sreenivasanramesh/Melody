from pathlib import Path
import pickle
from music21 import converter, instrument, note, chord


def convert_midi_to_notes(midi_file):
    """ Converts midi to a list of notes """
    notes = list()
    midi = converter.parse(midi_file)
    notes_to_parse = None

    try:
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notesAndRests

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
        elif isinstance(element, note.Rest):
            notes.append(element.name)
    return notes


def main():
    working_dir = Path.cwd()
    data_dir = working_dir/"data"
    result_dir = data_dir/"processed_data"
    test_dir = data_dir / "test_data"
    metadata_dir = result_dir / "metadata"

    # a list containing all the notes fromm all songs concatenated together
    notes = list()
    for file1 in data_dir.glob("*.mid"):
        print("Processing File ", file1)
        try:
            translated_score = convert_midi_to_notes(str(file1))
        except:
            print("Skippping {} as it is corrupted".format(file1))
            continue
        translated_score = convert_midi_to_notes(str(file1))
        # used for generating files for testing
        # with open(str(test_dir / "{}.pkl".format(str(file1.stem))), 'wb') as file_path:
        #   pickle.dump(translated_score, file_path)
        notes.extend(translated_score)

    with open(str(result_dir / "notes.pkl"), 'wb') as file_path:
        pickle.dump(notes, file_path)
   





    """
    # TODO: fix this mess later by using embedding layer
    # create vocab here itself since doing so later might give errors while playing test data
    pitch_names = sorted(set(item for item in notes))
    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))
    with open(metadata_dir / 'note_to_int.pkl', 'wb') as f:
        pickle.dump(note_to_int, f)
    with open(metadata_dir / 'int_to_note.pkl', 'wb') as f:
        pickle.dump(int_to_note, f)
    """


    print("Finished Pre-processing")


if __name__ == "__main__":
    main()