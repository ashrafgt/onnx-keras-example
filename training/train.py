import sys
import numpy as np
from keras import layers, models
from keras.datasets import imdb

# declare constants
MAX_UNIQUE_WORDS = 20000
BATCH_SIZE = 5000
EPOCHS = 10
MODEL_PATH = sys.argv[1]


def load_prepare_split_data():
    # load dataset
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(
        num_words=MAX_UNIQUE_WORDS
    )
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    # prepare dataset
    vectorized_data = np.zeros((len(data), MAX_UNIQUE_WORDS))
    for i, sequence in enumerate(data):
        vectorized_data[i, sequence] = 1
    data = vectorized_data
    targets = np.array(targets).astype("float32")

    # split datset (leave first 1000 rows for validation testing)
    test_x = data[1000:10000]
    test_y = targets[1000:10000]
    train_x = data[10000:]
    train_y = targets[10000:]

    return train_x, train_y, test_x, test_y


# get training data
train_x, train_y, test_x, test_y = load_prepare_split_data()

# build model
model = models.Sequential()
model.add(layers.Dense(50, activation="relu", input_shape=(MAX_UNIQUE_WORDS,)))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# fit model
results = model.fit(
    train_x,
    train_y,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(test_x, test_y),
)
print(np.mean(results.history["val_accuracy"]))

# save model
model.save(MODEL_PATH)
