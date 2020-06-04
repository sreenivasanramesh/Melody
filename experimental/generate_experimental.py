import argparse
import tensorflow as tf
from pre_processing_experimental import Decode
import numpy
from pathlib import Path
from models import *
from utils import *
import random


def generate(
    model_name, weights_file, num_units=128, sequence_length=100, test_file=None
):
    int_to_note = read_binary_file(metadata_dir / "int_to_note.pkl")
    vocab_size = len(int_to_note)

    if model_name == "lstm":
        model = SingleLSTM(num_units, vocab_size).get_network(
            sequence_length=sequence_length
        )
    elif model_name == "bi-lstm":
        model = BiLSTM(num_units, vocab_size).get_network(
            sequence_length=sequence_length
        )
    elif model_name == "lstm-attention":
        model = AttentionLSTM(num_units, vocab_size).get_network(
            sequence_length=sequence_length
        )
    elif model_name == "bi-lstm-attention":
        model = AttentionBiLSTM(num_units, vocab_size).get_network(
            sequence_length=sequence_length
        )

    print(weights_file, num_units)
    model.load_weights(str(weights_file))
    prediction_output = generate_notes(
        model, sequence_length=sequence_length, test_file=test_file
    )
    print("Generated output, translating to midi...")

    filename = str(out_dir / "{}output.mid".format(model_name))
    Decode().generate_midi(prediction_output, str(filename))
    play_midi(str(filename))


def generate_notes(model, sequence_length, test_file=None):
    """ Generate notes from the neural network based on a sequence of notes """

    network_input = read_binary_file(metadata_dir / "sequence_in.pkl")
    note_to_int = read_binary_file(metadata_dir / "note_to_int.pkl")
    int_to_note = read_binary_file(metadata_dir / "int_to_note.pkl")

    pattern = list()
    prediction_output = list()

    if not test_file:
        start_pos = numpy.random.randint(0, len(network_input) - sequence_length - 1)
        network_input = network_input[start_pos : start_pos + sequence_length]
        for i in range(len(network_input)):
            pattern.append(note_to_int[i])
            # prediction_output.append(i)
    else:
        inp_pattern = ""
        with open(str(test_file), "r") as f:
            inp_pattern = f.read().rstrip().split(" ")

        inp_pattern = inp_pattern[0:sequence_length]
        prediction_output = inp_pattern
        for i in range(len(inp_pattern)):
            key = inp_pattern[i]
            inp_pattern[i] = note_to_int[key]
        pattern = inp_pattern

    print("Simulating the model...")
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(note_to_int))
        prediction = model.predict(prediction_input, verbose=0)

        # by default take the argmax
        index = numpy.argmax(prediction)

        # with some probability randomly sample from the top few probable notes
        if random.random() < 0.4:
            prediction = tf.constant(prediction)
            top, ind = tf.math.top_k(prediction, k=5)
            dist = tf.compat.v1.distributions.Multinomial(total_count=3.0, logits=top)
            multinomial_dist = dist.sample().numpy()[0]
            choice = tf.math.argmax(multinomial_dist).numpy()
            index = ind.numpy()[0][choice]

        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1 : len(pattern)]
    return prediction_output


def main():
    """Main method to train selected network"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        dest="model",
        help="Model to train",
        choices=["lstm", "bi-lstm", "lstm-attention"],
        required=True,
    )
    parser.add_argument(
        "--weights",
        dest="weights",
        help="Give hdf5 file on which the model was trained on",
        required=True,
    )
    parser.add_argument(
        "--units",
        dest="units",
        help="Num of LSTM units which was used to train the weights for given HDF5 file",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--test",
        dest="test_file",
        help="Give the processed file name for which you want to test the song on\
                        A random input is chosen if the file is not provided",
    )
    parser.add_argument(
        "--sequence-length",
        dest="sequence_length",
        help="Sequence length",
        default=100,
        type=int,
    )
    args = parser.parse_args()

    test_file = None
    if args.test_file:
        test_file = str(test_dir / args.test_file)
    weights_file = str(model_dir / args.model / args.weights)
    generate(args.model, weights_file, args.units, args.sequence_length, test_file)


working_dir = Path.cwd()
data_dir = working_dir / "data/processed_data_experimental"
test_dir = working_dir / "data/test_data"
metadata_dir = data_dir / "metadata"
model_dir = working_dir / "models"
out_dir = working_dir / "samples"

if __name__ == "__main__":
    main()
