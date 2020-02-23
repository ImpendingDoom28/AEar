import numpy as np
import librosa
import tensorflow as tf
import pandas as pd

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_dataset(df: pd.DataFrame) -> tf.data.Dataset:
    stfts = []
    masks = []
    i = 0
    for index, row in df.iterrows():
        stft = np.abs(librosa.stft(librosa.load(row['track'], sr=44100)[0]))
        stft = stft[..., np.newaxis]
        stfts.append(stft)
        masks.append(np.load(row['mask']))
        print("retrieved the stft and mask for ", i, "index")
        i += 1
    dataset = tf.data.Dataset.from_tensor_slices((stfts, masks))
    print("dataset ready")
    return dataset


def split_dataset(ds: tf.data.Dataset):
    val_num = 500
    test_num = 500

    train_dataset = ds.skip(val_num + test_num)

    test_val_ds = ds.take(val_num + test_num)

    test_dataset = test_val_ds.take(test_num)
    val_dataset = test_val_ds.skip(test_num)

    print("dataset splitted")
    return train_dataset, test_dataset, val_dataset


def prepare_for_training(ds, shuffle_buffer_size=1024, batch_size=32):
    # Randomly shuffle (file_path, label) dataset
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat dataset forever
    # ds = ds.repeat()
    # Prepare batches
    ds = ds.batch(batch_size)
    # Prefetch
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    print("dataset prepared for training")
    return ds


def main():
    dataframe = pd.read_csv("../dataset\\meta.csv", names=['track', 'mask'], header=None)

    dataset = get_dataset(dataframe)

    train_dataset, test_dataset, val_dataset = split_dataset(dataset)

    test_dataset = test_dataset.batch(32)
    val_dataset = val_dataset.batch(32)

    train_dataset = prepare_for_training(train_dataset)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), padding='same', input_shape=(1025, 9, 1)),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(16, (3, 3), padding='same'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1025)
    ])

    print(model.summary())

    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    for test_track, test_mask in test_dataset.take(1):
        print(model.predict(test_track)[0])
        print(test_mask[0])


if __name__ == '__main__':
    main()
