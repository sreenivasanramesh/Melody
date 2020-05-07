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
    metadata_dir = result_dir / "metadata"

    # a list containing all the notes fromm all songs concatenated together
    notes = list()
    for file1 in data_dir.glob("s*.mid"):
        print("Processing File ", file1)
        notes.extend(convert_midi_to_notes(str(file1)))

    with open(str(result_dir / "notes_sonata.pkl"), 'wb') as file_path:
        pickle.dump(notes, file_path)
    with open(str(metadata_dir / "vocab_size.pkl"), 'wb') as file_path:
        pickle.dump(len(set(notes)), file_path)
    print("Finished Pre-processing")


if __name__ == "__main__":
    main()