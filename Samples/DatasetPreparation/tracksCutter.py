import os
import librosa
import musdb
import numpy as np
from scipy.io.wavfile import write


def divide_audio_into_frames(song, frames=9) -> None:
    """
    Divide audio track into equal pieces and save them
    :param song: comes as MultiTrack representation from MUSDB
    :param frames: parts to divide track by
    """
    assert song is not None
    vocals_stft = librosa.stft(librosa.to_mono(song.targets['vocals'].audio.T))
    accompaniment_stft = librosa.stft(librosa.to_mono(song.targets['accompaniment'].audio.T))
    song_stft = librosa.stft(librosa.to_mono(song.audio.T))
    for i in range(song_stft.shape[1] - frames):
        # mask = binary_mask(accompaniment_stft[:, i * frames: i * frames + frames],
        #                    vocals_stft[:, i * frames: i * frames + frames])
        mask_for_middle_part = binary_mask(accompaniment_stft[:, (i + frames - 1) // 2],
                                           vocals_stft[:, (i + frames - 1) // 2])
        track_part = song_stft[:, i: i + frames]

        save_track_and_its_mask(track_part, mask_for_middle_part)
    return



def binary_mask(source, target):
    """
        :param source: Everything in the mix except for target (for example: for vocals it would be accompaniment)
        :param target: Target source that needs to be separated
        :return: Binary mask for separation the target from full mix
        """
    assert source is not None and target is not None
    assert source.shape == target.shape

    mask = np.zeros(source.shape)

    iterator = np.nditer(source, flags=['multi_index'])

    while not iterator.finished:
        source_item = iterator[0]
        target_item = target[iterator.multi_index]

        if np.abs(target_item) > np.abs(source_item):
            mask[iterator.multi_index] = 1

        iterator.iternext()

    return mask


def save_track_and_its_mask(track, mask, directory_to_save='dataset'):
    """
    Track will be saved as wav file, mask will be saved in .npy format (which is easy to work with using np.load(filename))
    :param track: track to save, should be ndarray data type
    :param mask: computed mask, should by ndarray data type
    """
    global TRACK_NAME
    if not os.path.exists(directory_to_save):
        os.mkdir(directory_to_save)
        os.mkdir(directory_to_save + '/tracks')
        os.mkdir(directory_to_save + '/masks')
    write(directory_to_save + '/tracks/' + str(TRACK_NAME) + '.wav', 44100, librosa.istft(track))
    np.save(directory_to_save + '/masks/' + str(TRACK_NAME), mask)
    TRACK_NAME += 1


TRACK_NAME = 0

db = musdb.DB(root='~/MUSDB18/MUSDB18-7')
for i in range(db.__len__()):
    divide_audio_into_frames(db[i])
