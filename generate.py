import argparse
import pygame
import pickle
import tensorflow as tf
from pre_processing import Decode
import numpy
from pathlib import Path
from models import *


def read_metadata(metadata_file):
    """reads pickle to get back dictionary of vocab -> embeddings used during training"""
    with open(str(metadata_file), 'rb') as bin_file:
        obj = pickle.load(bin_file)
        return obj


def generate(model_name, test_data=None):
    sequence_in = read_metadata(metadata_dir/'sequence_in.pkl')
    note_to_int = read_metadata(metadata_dir/'note_to_int.pkl')
    int_to_note = read_metadata(metadata_dir/'int_to_note.pkl')
    vocab_size = len(int_to_note)

    network_input = numpy.reshape(sequence_in, (len(sequence_in), len(sequence_in[0]), 1))
    model = create_network(network_input, vocab_size)

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

    filename = "new_processing/" + type(model).__name__ + "/weights.hdf5"
    filename =  model_dir / filename

    model.load_weights(str(filename))
    prediction_output = generate_notes(model, sequence_in, note_to_int, int_to_note)
    print("Generated output, translating to midi...")

    filename = out_dir / "output.mid"
    Decode().generate_midi(prediction_output, str(filename))
    play_midi("yolo.mid")


def generate_notes(model, network_input, note_to_int, int_to_note, test_file=None):
    """ Generate notes from the neural network based on a sequence of notes """

    prediction_output = list()
    if not test_file:
        start_seq = numpy.random.randint(0, len(network_input) - 1)
        pattern = network_input[start_seq]
    else:
        input = ""
        with open(str(test_file), 'r') as f:
            input = f.read().rstrip().split(' ')

            input = input[0:100]  # TODO: 100 is sequence length, change to variable
        for i in range(0, len(input)):
            input[i] = note_to_int[i]
        prediction_output = prediction_output + input
        pattern = input

    print("Simulating the model...")
    for note_index in range(2000):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(note_to_int))

        prediction = model.predict(prediction_input, verbose=0)
        prediction = tf.constant(prediction)

        """sampling from output rather than taking argmax"""
        # get top k probable notes
        top, ind = tf.math.top_k(prediction, k=5)
        # convert them to a probability distribution
        topk_probabilities = tf.math.softmax(top).numpy()
        # select one of the notes by sampling them with the probability distribution generated above
        index = numpy.random.choice(ind.numpy()[0], p=topk_probabilities[0])

        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    return prediction_output


def play_music(music_file):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print ("Music file %s loaded!" % music_file)
    except pygame.error:
        print ("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)


def play_midi(file):
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024  # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # volume 0 to 1.0
    pygame.mixer.music.set_volume(0.6)
    try:
        play_music(file)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit


def main():
    """Main method to train selected network"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Model to train",
                        choices=["lstm", "bi-lstm", "lstm-attention", "bi-lstm-attention"], required=True)
    # parser.add_argument("--processing", dest="pre_processing_method", help="Which pre processing method to use",
    #                    choices=["old", "new"], default="new")
    args = parser.parse_args()
    generate(args.model)


working_dir = Path.cwd()
data_dir = working_dir / "data/processed_data"
metadata_dir = data_dir / "metadata"
model_dir = working_dir / "models"
out_dir = working_dir / "samples"

if __name__ == '__main__':
    main()






