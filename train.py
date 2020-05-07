import argparse
from models import *
import pickle
from pathlib import Path
import numpy as np
import tensorflow.keras.utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import read_binary_file


def prepare_validation_data(files, sequence_length):
    """have to re-write"""
    notes = list()
    sequence_in = list()
    sequence_out = list()
    note_to_int = read_binary_file(metadata_dir / 'note_to_int.pkl')

    for file_name in files:
        data = read_binary_file(file_name) 
        notes.extend(data)

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in.append(notes[i:i + sequence_length])
        sequence_out.append(notes[i + sequence_length])
    for i in range(0, len(sequence_in)):
        for j in range(0, len(sequence_in[i])):
            sequence_in[i][j] = note_to_int[sequence_in[i][j]]
    for i in range(0, len(sequence_out)):
        sequence_out[i] = note_to_int[sequence_out[i]]

    sequence_in = np.reshape(sequence_in, (len(sequence_in), len(sequence_in[0]), 1))
    sequence_in = sequence_in / float(len(note_to_int))
    sequence_out = np_utils.to_categorical(sequence_out, num_classes=len(note_to_int))
    return sequence_in, sequence_out


def get_train_data(sequence_length=100):
    """ Prepare the input/output data for the Neural Network """

    network_input = list()
    network_output = list()
    notes = read_binary_file(str(data_dir / "notes.pkl"))

    # get all pitch names
    pitch_names = sorted(set(item for item in notes))
    # Embedding #TODO use keras Embedding layer instead
    note_to_int = read_binary_file(metadata_dir / 'note_to_int.pkl')
    int_to_note = read_binary_file(metadata_dir / 'int_to_note.pkl')
    vocab_size = len(set(note_to_int))

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(vocab_size)
    network_output = np_utils.to_categorical(network_output)

    with open(metadata_dir / 'sequence_in.pkl', 'wb') as f:
        pickle.dump(network_input, f)
    with open(metadata_dir / 'sequence_out.pkl', 'wb') as f:
        pickle.dump(network_output, f)
    # moved note to int and int to note to pre_precessing step
    # with open(metadata_dir / 'note_to_int.pkl', 'wb') as f:
    #    pickle.dump(note_to_int, f)
    # with open(metadata_dir / 'int_to_note.pkl', 'wb') as f:
    #    pickle.dump(int_to_note, f)
    return network_input, network_output, vocab_size


def train_network(model_name, batch_size=64, epochs=100, num_units=64, sequence_length=100):
    """ Trains the model and stores the weights in the output directory """
    network_input, network_output, vocab_size = get_train_data(sequence_length)
    print("vocab_size = ", vocab_size)

    test_files = list()
    for test_file in test_dir.glob("*.pkl"):
        if not test_file.stem.startswith("."):
            test_files.append(str(test_file))

    test_in, test_out = prepare_validation_data(test_files, sequence_length)

    # num_units = 256
    if model_name == "lstm":
        model = SingleLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "bi-lstm":
        model = BiLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "lstm-attention":
        model = AttentionLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "bi-lstm-attention":
        model = AttentionBiLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)

    filename = model_name + "/model-{epoch:02d}-{loss:.4f}.hdf5"  # type(model).__name__ +
    file_path = str(model_dir / filename)
    checkpoint = ModelCheckpoint(file_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], validation_data=(test_in, test_out))


working_dir = Path.cwd()
data_dir = working_dir / "data/processed_data"
test_dir = working_dir / "data/test_data"
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
    train_network(args.model, batch_size=64, epochs=200, num_units=128, sequence_length=100)


if __name__ == "__main__":
    main()