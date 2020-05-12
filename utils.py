"""
Contains functions used by others
"""
import pickle
import pygame
from pathlib import Path


def read_binary_file(file_name):
    """reads binary file and returns the content"""
    with open(str(file_name), 'rb') as bin_file:
        obj = pickle.load(bin_file)
        return obj


def play_music(music_file):
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


def play_midi(music_file):
    freq = 44100  # audio CD quality
    bitsize = -16  # unsigned 16 bit
    channels = 2  # 1 is mono, 2 is stereo
    buffer = 1024  # number of samples
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # volume 0 to 1.0
    pygame.mixer.music.set_volume(0.6)
    try:
        play_music(music_file)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit


def get_paths():
    """ returns a dictionary of directory paths """
    paths = dict()
    paths["working_dir"] = Path.cwd()
    paths["data_dir"] = paths["working_dir"] / "data/train_data"
    paths["test_dir"] = paths["working_dir"] / "data/test_data"
    paths["metadata_dir"] = paths["data_dir"] / "metadata"
    paths["model_dir"] = paths["working_dir"] / "models"
    return paths


