import argparse
from models import *
import pickle
from pathlib import Path
import numpy as np
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint



def read_metadata(metadata_file):
    """reads pickle to get back dictionary of vocab -> embeddings used during training"""
    with open(str(metadata_file), 'rb') as bin_file:
        obj = pickle.load(bin_file)
        return obj


def prepare_validation_song(file, sequence_length):
    notes = None
    sequence_in = list()
    sequence_out = list()
    note_to_int = read_metadata(metadata_dir / 'note_to_int.pkl')
    print(note_to_int)
    int_to_note = read_metadata(metadata_dir / 'int_to_note.pkl')
    print(len(note_to_int))

    with open(file) as f:
        notes = f.read().rstrip().split(' ')
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in.append(notes[i:i + sequence_length])
            sequence_out.append(notes[i + sequence_length])
        for i in range(0, len(sequence_in)):
            for j in range(0,len(sequence_in[i])):
                sequence_in[i][j] = note_to_int[sequence_in[i][j]]

    for i in range(0,len(sequence_out)):
        sequence_out[i] = note_to_int[sequence_out[i]]
    
    sequence_in = np.reshape(sequence_in, (len(sequence_in), len(sequence_in[0]), 1))
    sequence_in = sequence_in / float(len(note_to_int))
    sequence_out = np_utils.to_categorical(sequence_out, num_classes = len(note_to_int))

    #print(len(sequence_in))
    #print(len(sequence_out))
    return sequence_in,sequence_out






def train_network(model_name, batch_size=64, epochs=100, num_units=64, sequence_length=100):
    """ Trains the model and stores the weights in the output directory """
    network_input, network_output, vocab_size = get_data(sequence_length)
    network_input = np.reshape(network_input, (len(network_input), len(network_input[0]), 1))
    network_input = network_input / float(vocab_size)
    network_output = np_utils.to_categorical(network_output)

    # we have currently hard-coded these values as we ran different networks on different num_units
    # as we wanted to play around with num_units and see how the outputs change
    # we did not have enough resources to run them all on the same value for num_units




    test_in, test_out =  prepare_validation_song(str(test_dir / "sonat-9.txt"), sequence_length)




    if model_name == "lstm":
        model = SingleLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "bi-lstm":
        model = BiLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "lstm-attention":
        model = AttentionLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "bi-lstm-attention":
        model = AttentionBiLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)

    filename = model_name + "/model-{epoch:02d}-{loss:.4f}.hdf5"  # type(model).__name__ + 
    filepath = str(model_dir / filename)



    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')


    callbacks_list = [checkpoint]
    print(test_in.shape)
    print(test_out.shape)
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list, validation_data=(test_in, test_out))


def get_data(sequence_length):
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    vocab = set()
    sequence_in = list()
    sequence_out = list()

    all_notes = list()

    for file1 in data_dir.glob("simpl*.txt"):
        notes = None
        with open(file1, 'r') as f:
            notes = f.read().rstrip().split(' ')
        vocab.update(notes)
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in.append(notes[i: (i+sequence_length)])
            sequence_out.append(notes[i+sequence_length])

    vocab = sorted(vocab)
    print(len(vocab))
    note_to_int = dict((note, number) for number, note in enumerate(vocab))
    int_to_note = dict((number, note) for number, note in enumerate(vocab))

    for i in range(0, len(sequence_in)):
        for j in range(0, len(sequence_in[i])):
            sequence_in[i][j] = note_to_int[sequence_in[i][j]]
    for i in range(0, len(sequence_out)):
        sequence_out[i] = note_to_int[sequence_out[i]]

    with open(metadata_dir / 'sequence_in.pkl', 'wb') as f:
        pickle.dump(sequence_in, f)
    with open(metadata_dir / 'sequence_out.pkl', 'wb') as f:
        pickle.dump(sequence_out, f)
    with open(metadata_dir / 'note_to_int.pkl', 'wb') as f:
        pickle.dump(note_to_int, f)
    with open(metadata_dir / 'int_to_note.pkl', 'wb') as f:
        pickle.dump(int_to_note, f)
    return sequence_in, sequence_out, len(vocab)


working_dir = Path.cwd()
data_dir = working_dir / "data/processed_data_experimental"
test_dir = working_dir / "data/processed_test_data"
metadata_dir = data_dir / "metadata"
model_dir = working_dir / "models"


def main():
    """Main method to train selected network"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Model to train",
                        choices=["lstm", "bi-lstm", "lstm-attention", "bi-lstm-attention"], required=True)
    # parser.add_argument("--processing", dest="pre_processing_method", help="Which pre processing method to use",
    #                    choices=["old", "new"], default="new")
    args = parser.parse_args()
    train_network(args.model, batch_size=64, epochs=100, num_units=64, sequence_length=100)


if __name__ == "__main__":
    main()