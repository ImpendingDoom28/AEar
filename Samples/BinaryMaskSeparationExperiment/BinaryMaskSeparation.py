import musdb
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io.wavfile import write


def binaryMask(source=None, target=None) -> np.ndarray:
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
        sourceItem = iterator[0]
        targetItem = target[iterator.multi_index]

        if np.abs(targetItem) > np.abs(sourceItem.real):
            mask[iterator.multi_index] = 1

        iterator.iternext()

    return mask


musicDatabase = musdb.DB(root="~/MUSDB18/MUSDB18-7")  # Если заменить аргумент на download=True, то датасет скачается

song = musicDatabase[2]

vocals = song.targets['vocals']  # STEM вокала
accompaniment = song.targets['accompaniment']  # STEM аккомпанимента

voxStft = librosa.stft(librosa.to_mono(vocals.audio.T))
accompanimentStft = librosa.stft(librosa.to_mono(accompaniment.audio.T))
fullStft = librosa.stft(librosa.to_mono(song.audio.T))

mask = binaryMask(source=accompanimentStft, target=voxStft)  # считаем маску

vocalsExtractedSTFT = np.multiply(fullStft, mask)  # умножаем исходный микс на маску
vocalsExtractedAudio = librosa.istft(vocalsExtractedSTFT)  # выполняем обратное преобразование

write("ExtractedVocals.wav", 44100, vocalsExtractedAudio)
