import pandas as pd
import librosa
import numpy as np


def generate_data(df: pd.DataFrame, batch_size=256):
    """
    :param df: dataframe that contains paths to tracks and to corresponding masks
    :param batch_size: the size of batch
    :return: yields the batch of features and labels
    """

    offset = 0
    limit = batch_size
    while True:
        batch_feature = []
        batch_label = []
        batch_df = df.iloc[offset:limit]  # берем слайс датафрейма, из которого формируем батч

        batch_df = batch_df.sample(frac=1)  # перемешиваем

        for index, row in batch_df.iterrows():
            track = librosa.load(row['track'], sr=44100)[0]  # загружаем трек
            stft = np.abs(librosa.stft(track))  # вычисляем оконное преобразование и поэлементно находим модуль
            stft = np.expand_dims(stft, axis=2)  # добавляем дополнительное измерение
            mask = np.load(row['mask'])
            batch_feature.append(stft)
            batch_label.append(mask)

        offset = limit
        limit += batch_size

        if limit > int(
                df.shape[0]):  # если прошлись до конца, то сбрасываем лимит и оффсет и снова перемешиваем датафрейм
            offset = 0
            limit = batch_size
            df = df.sample(frac=1)

        yield np.array(batch_feature), np.array(batch_label)
