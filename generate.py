import argparse
import pygame
import pickle
import tensorflow as tf
from pre_processing_experimental import Decode
import numpy
from pathlib import Path
from models import *
from utils import read_binary_file
import random
from tensorflow.compat.v1.distributions import Multinomial


import pickle
import numpy
from music21 import instrument, note, stream, chord,converter
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
import tensorflow.keras.utils as np_utils
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint



def generate():
    """ Generate a piano midi file """
    #load the notes used to train the model
    with open('/home/tanush/Desktop/APM Project/Repo/MuseGen_1/data/notes/notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)




def generate(model_name, test_file=None):
    int_to_note = read_binary_file(metadata_dir/'int_to_note.pkl')
    note_to_int = read_binary_file(metadata_dir/'note_to_int.pkl')
    notes = read_binary_file(data_dir / "notes.pkl")
    vocab_size = len(int_to_note)

    # we have currently hard-coded these values as we ran different networks on different num_units
    # as we wanted to play around with num_units and see how the outputs change
    # we did not have enough resources to run them all on the same value for num_units
    num_units = 128
    sequence_length = 100

    if model_name == "lstm":
        model = SingleLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "bi-lstm":
        model = BiLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "lstm-attention":
        model = AttentionLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)
    elif model_name == "bi-lstm-attention":
        model = AttentionBiLSTM(num_units, vocab_size).get_network(sequence_length=sequence_length)

    filename = model_name + "/weights.hdf5"  # type(model).__name__ + "/weights.hdf5"
    filename = model_dir / filename
    model.load_weights(str(filename))
    prediction_output = generate_notes(model, sequence_length=sequence_length, test_file=test_file)
    print("Generated output, translating to midi...")

    filename = str(out_dir / "output.mid")
    create_midi(prediction_output, filename)
    play_midi(filename)




def generate_notes(model, sequence_length, test_file=None):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction

    int_to_note = read_binary_file(metadata_dir/'int_to_note.pkl')
    note_to_int = read_binary_file(metadata_dir/'note_to_int.pkl')
    vocab_size = len(note_to_int)

    pattern = list()
    prediction_output = list()
    if not test_file:
        notes = read_binary_file(data_dir/'notes.pkl')
        start_pos = numpy.random.randint(0, len(notes)-sequence_length-1)
        notes = notes[start_pos: start_pos+sequence_length]
        for i in notes:
            pattern.append(note_to_int[i])
            prediction_output.append(i)
    else:
        notes = read_binary_file(data_dir/test_file)[:sequence_length]
        for i in notes:
            pattern.append(note_to_int[i])
            prediction_output.append(i)    

    # generate 1000 notes
    print("\n\nSimulating the network...\n\n")
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab_size)
        prediction = model.predict(prediction_input, verbose=0)

        # by default take the argmax
        index = numpy.argmax(prediction)

        # with some probability randomly sample from the top few probable notes
        if random.random() < 0.4:
            prediction = tf.constant(prediction)
            top, ind = tf.math.top_k(prediction, k=5)
            dist = Multinomial(total_count=3., logits=top)
            multinomial_dist = dist.sample().numpy()[0]
            iii = tf.math.argmax(multinomial_dist).numpy()
            index = ind.numpy()[0][iii]
        result = int_to_note[index]
        prediction_output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output



def create_midi(prediction_output, filename):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = list()

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a rest
        elif('rest' in pattern):
            new_rest = note.Rest(pattern)
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano() #???
            output_notes.append(new_rest)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', filename)





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
    generate(args.model, str(test_dir / "test_elise_format0.pkl"))  # , None)


working_dir = Path.cwd()
data_dir = working_dir / "data/processed_data"
test_dir = working_dir / "data/test_data"
metadata_dir = data_dir / "metadata"
model_dir = working_dir / "models"
out_dir = working_dir / "samples"

if __name__ == '__main__':
    main()