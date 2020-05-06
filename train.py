import argparse
from models import *
import pickle
from pathlib import Path
import numpy as np
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint


def train_network(model_name, batch_size, epochs):
    """ Trains the model and stores the weights in the output directory """
    network_input, network_output, vocab_size = get_data()
    network_input = np.reshape(network_input, (len(network_input), len(network_input[0]), 1))
    network_input = network_input / float(vocab_size)
    network_output = np_utils.to_categorical(network_output)

    # we have currently hard-coded these values as we ran different networks on different num_units
    # as we wanted to play around with num_units and see how the outputs change
    # we did not have enough resources to run them all on the same value for num_units
    num_units = 256

    if model_name == "lstm":
        model = SingleLSTM(num_units, vocab_size).get_network()
    elif model_name == "bi-lstm":
        model = BiLSTM(num_units, vocab_size).get_network()
    elif model_name == "lstm-attention":
        model = AttentionLSTM(num_units, vocab_size).get_network()
    elif model_name == "bi-lstm-attention":
        model = AttentionBiLSTM(num_units, vocab_size).get_network()

    filename = "new_processing/" + type(model).__name__ + "/model-{epoch:02d}-{loss:.4f}.hdf5"
    filepath = model_dir / filename
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)


def get_data():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    vocab = set()
    sequence_in = list()
    sequence_out = list()
    sequence_length = 100

    for file1 in data_dir.glob("*.txt"):
        notes = None
        with open(file1, 'r') as f:
            notes = f.read().rstrip().split(' ')
        vocab.update(notes)
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in.append(notes[i: (i+sequence_length)])
            sequence_out.append(notes[i+sequence_length])

    vocab = sorted(vocab)
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
data_dir = working_dir / "data/processed_data"
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
    train_network(args.model, batch_size=64, epochs=50)


if __name__ == "__main__":
    main()