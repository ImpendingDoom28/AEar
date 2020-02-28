import musdb
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

musicDatabase = musdb.DB(root="~/MUSDB18/MUSDB18-7")  # Если заменить аргумент на download=True, то датасет скачается

song = musicDatabase[2]  # просто берем любой мультитрек

vocals = song.targets['vocals']  # STEM вокала
accompaniment = song.targets['accompaniment']  # STEM аккомпанимента

STFTOfTheWholeSong = librosa.stft(librosa.to_mono(song.audio.T))  # Преобразование Фурье от всего трека
# Отображение полной спектрограммы
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(STFTOfTheWholeSong), ref=np.max), y_axis='log', x_axis='time')
# plt.show()

STFTOfTheVocals = np.abs(librosa.stft(librosa.to_mono(vocals.audio.T)))  # Преобразование Фурье от дорожки вокала
# Отображение спектрограммы вокала
librosa.display.specshow(librosa.amplitude_to_db(np.abs(STFTOfTheVocals), ref=np.max), y_axis='log', x_axis='time',
                         sr=44100)
plt.savefig("./transparent/vox_spec.png", transparent=True)
plt.show()

STFTOfTheAccompaniment = np.abs(
    librosa.stft(librosa.to_mono(accompaniment.audio.T)))  # Преобразование Фурье от аккомпанимента
# Отображение спектрограммы аккомпанимента
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(STFTOfTheAccompaniment), ref=np.max), y_axis='log', x_axis='time')
# plt.show()

# Теперь нам нужно вычислить "маску" вокала: MaskVocals = STFTVocals / (STFTVocals + STFTAccompaniment + epsilon)
# epsilon нужен для того, чтобы случайно не поделить на 0 (на самом деле не очень ясно, когда такое может получиться, но вдруг)
epsilon = np.finfo(np.float).eps

fullSTFT = epsilon + np.abs(STFTOfTheVocals) + np.abs(STFTOfTheAccompaniment)
maskVocals = np.divide(np.abs(STFTOfTheVocals), fullSTFT)
# Отображение спектрограммы получившейся маски
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(maskVocals), ref=np.max), y_axis='log', x_axis='time')
# plt.show()

# Теперь мы можем получить вокал из оригинального трека. Для этого нужно умножить его STFT на STFT маски, после чего провести обратное преобразование
vocalsExtractedSTFT = np.multiply(STFTOfTheWholeSong, maskVocals)
vocalsExtractedAudio = librosa.istft(vocalsExtractedSTFT)

# До разделения
songAudio = librosa.istft(STFTOfTheWholeSong)

# Теперь сделаем то же самое, но с аккомпаниментом:
accompanimentMask = np.divide(STFTOfTheAccompaniment, epsilon + STFTOfTheAccompaniment + STFTOfTheVocals)
# Отображение спектрограммы маски
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(accompanimentMask), ref=np.max), y_axis='log', x_axis='time')
# plt.show()

accompanimentExtracted = np.multiply(accompanimentMask, fullSTFT)
accompanimentExtractedAudio = librosa.istft(accompanimentExtracted)

# Экспорт получившихся результатов:
# write("Before.wav", 44100, songAudio)
# write("ExtractedVocals.wav", 44100, vocalsExtractedAudio)
# write("ExtractedAccompaniment.wav", 44100, accompanimentExtractedAudio)
