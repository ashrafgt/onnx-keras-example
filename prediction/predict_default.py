import sys
from time import perf_counter
import numpy as np
from keras.datasets import imdb
from keras.models import load_model

# declare constants
MAX_UNIQUE_WORDS = 20000
MODEL_PATH = sys.argv[1]


def load_prepare_data():
    # load data
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(
        num_words=MAX_UNIQUE_WORDS
    )
    prediction_x = np.concatenate((training_data, testing_data), axis=0)[:1000]

    # prepare data
    vectorized_data = np.zeros((len(prediction_x), MAX_UNIQUE_WORDS))
    for i, sequence in enumerate(prediction_x):
        vectorized_data[i, sequence] = 1
    prediction_x = vectorized_data

    return prediction_x


# get prediction data
prediction_x = load_prepare_data()

# load model
model = load_model(MODEL_PATH)

start = perf_counter()
predictions = model.predict(prediction_x)
end = perf_counter()

print(predictions)
print(len(predictions))
print(end - start)
