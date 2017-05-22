# Michael A. Alcorn (airalcorn2@gmail.com)

import gensim
import numpy as np
import random

from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, LSTM, Masking, TimeDistributed
from keras.models import Sequential
from matplotlib import pyplot as plt

# Inspired by: http://lewisandquark.tumblr.com/post/160776374467/new-paint-colors-invented-by-neural-network.
# Colors downloaded from: https://images.sherwin-williams.com/content_images/sw-colors-name-csp-acb.acb.
# Word vectors downloaded from: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing.


def read_colors():
    """Read in the colors from an Adobe Photoshop Color Book (ACB) file.

    :return:
    """
    f = open("sw-colors-name-csp-acb.acb", "r")
    rgbs = {}
    line = f.readline()
    while line != "":
        if "colorName" in line:
            name = line.split(">")[1].split("(")[0].strip()
            rgb = {}
            line = f.readline()
            for color in ["r", "g", "b"]:
                line = f.readline()
                val = int(line.split(">")[1].split("<")[0])
                rgb[color] = val

            rgbs[name] = rgb

        line = f.readline()

    f.close()
    return rgbs


def build_words2color_model(max_tokens, dim):
    """Build a model that learns to generate colors from words.

    :param max_tokens:
    :param dim:
    :return:
    """
    model = Sequential()
    model.add(Conv1D(128, 1, input_shape = (max_tokens, dim), activation = "tanh"))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(3))

    model.compile(loss = "mse", optimizer = "sgd")
    return model


def words_to_color(rgbs, word2vec):
    """Train a model that learns to generate colors from words.

    :param rgbs:
    :param word2vec:
    :return:
    """
    max_tokens = max([len(name.split()) for name in rgbs])
    (total_tokens, dim) = word2vec.syn0.shape
    avg_vec = np.mean(word2vec.syn0, axis = 0)
    empty_vec = np.zeros(dim)
    (X_train, Y_train) = ([], [])

    for name in rgbs:
        tokens = name.lower().split()
        X = []
        i = 0

        for token in tokens:
            if token in word2vec:
                X.append(word2vec[token])
            else:
                X.append(avg_vec)

            i += 1

        while i < max_tokens:
            X.append(empty_vec)
            i += 1

        X_train.append(np.array(X))
        y = [rgbs[name][c] for c in ["r", "g", "b"]]
        Y_train.append(np.array(y))

    idxs = list(range(len(X_train)))
    random.shuffle(idxs)
    (final_X_train, final_Y_train) = ([], [])
    for idx in idxs:
        final_X_train.append(X_train[idx])
        final_Y_train.append(Y_train[idx])

    (final_X_train, final_Y_train) = (np.array(final_X_train), np.array(final_Y_train))

    model = build_words2color_model(max_tokens, dim)
    checkpoint = ModelCheckpoint("words2color_params.h5", monitor = "val_loss", verbose = 1, save_best_only = True)
    model.fit(final_X_train, final_Y_train, epochs = 500, validation_split = 0.1, callbacks = [checkpoint])
    model.load_weights("words2color_params.h5")
    return (model, max_tokens, dim)


def build_color2words_model(max_tokens, dim):
    """Build a model that learns to generate words from colors.

    :param max_tokens:
    :param dim:
    :return:
    """
    model = Sequential()
    model.add(Masking(mask_value = -1, input_shape = (max_tokens, 3)))
    model.add(LSTM(256, return_sequences = True))
    model.add(TimeDistributed(Dense(dim)))

    model.compile(loss = "mse", optimizer = "adam")
    return model


def color_to_words(rgbs, word2vec):
    """Train a model that learns to generate words from colors.

    :param rgbs:
    :param word2vec:
    :return:
    """
    max_tokens = max([len(name.split()) for name in rgbs])
    (total_tokens, dim) = word2vec.syn0.shape
    avg_vec = np.mean(word2vec.syn0, axis = 0)
    empty_vec = np.zeros(dim)
    (X_train, Y_train) = ([], [])
    empty_rgb_vec = 3 * [0]
    mask_rgb_vec = 3 * [-1]

    for name in rgbs:
        tokens = name.lower().split()
        num_tokens = len(tokens)
        i = 0
        Y = []
        for token in tokens:
            if token in word2vec:
                Y.append(word2vec[token])
            else:
                Y.append(avg_vec)

            i += 1

        while i < max_tokens:
            Y.append(empty_vec)
            i += 1

        rgb = [rgbs[name][c] for c in ["r", "g", "b"]]
        X = [rgb]
        i = 1
        while i < num_tokens:
            X.append(empty_rgb_vec)
            i += 1

        while i < max_tokens:
            X.append(mask_rgb_vec)
            i += 1

        X_train.append(np.array(X))
        Y_train.append(np.array(Y))

    idxs = list(range(len(X_train)))
    random.shuffle(idxs)
    (final_X_train, final_Y_train) = ([], [])
    for idx in idxs:
        final_X_train.append(X_train[idx])
        final_Y_train.append(Y_train[idx])

    (final_X_train, final_Y_train) = (np.array(final_X_train), np.array(final_Y_train))

    model = build_color2words_model(max_tokens, dim)
    checkpoint = ModelCheckpoint("color2words_params.h5", monitor = "val_loss", verbose = 1, save_best_only = True)
    model.fit(final_X_train, final_Y_train, epochs = 100, validation_split = 0.1, callbacks = [checkpoint])
    model.load_weights("color2words_params.h5")
    return (model, max_tokens, dim)


def display_color(rgb):
    """Display a color from RGB values.

    :param rgb:
    :return:
    """
    print(rgb)
    rgb /= 255
    (h, w) = (64, 64)
    color_swatch = np.zeros((h, w, 3))
    for row in range(h):
        for col in range(w):
            color_swatch[row, col] = rgb

    plt.imshow(color_swatch)
    plt.show()


def generate_color(words, max_tokens, dim, word2vec, words2color):
    """Generate color from a name.

    :param words:
    :param max_tokens:
    :param dim:
    :param word2vec:
    :param words2color:
    :return:
    """
    X_test = np.zeros((max_tokens, dim))
    for (i, word) in enumerate(words):
        X_test[i] = word2vec[word]

    rgb = words2color.predict(np.array([X_test]))[0]
    display_color(rgb)


def generate_name(rgb, max_tokens, word2vec, color2words):
    """Generate name from a color.

    :param rgb:
    :param max_tokens:
    :param word2vec:
    :param color2words:
    :return:
    """
    X_test = np.zeros((max_tokens, 3))
    X_test[0] = rgb
    neighbors = 10
    word_vecs = color2words.predict(np.array([X_test]))[0]
    color_name = []
    for word in range(max_tokens):

        word_vec = word_vecs[word]
        word_mag = np.sqrt(np.dot(word_vec, word_vec))
        dotted = np.dot(word_vec, word2vec.syn0.T)
        mags = []
        for i in range(word2vec.syn0.shape[0]):
            mags.append(np.sqrt(np.dot(word2vec.syn0[i], word2vec.syn0[i])))

        mags = np.array(mags)
        cosine = dotted / (word_mag * mags)

        keeps = []
        for idx in cosine.argsort()[-neighbors:][::-1]:
            print(word2vec.index2word[idx])
            keeps.append(idx)

        print()

        exp = np.exp(cosine[keeps])
        probs = exp / exp.sum()
        idx = np.random.choice(keeps, p = probs)
        color_word = word2vec.index2word[idx]
        color_name.append(color_word)

    print(" ".join(color_name))


def main():
    rgbs = read_colors()
    word2vec = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary = True)

    (words2color, max_tokens, dim) = words_to_color(rgbs, word2vec)

    generate_color(["purple"], max_tokens, dim, word2vec, words2color)
    generate_color(["deep", "purple"], max_tokens, dim, word2vec, words2color)

    generate_color(["ocean"], max_tokens, dim, word2vec, words2color)
    generate_color(["calm", "ocean"], max_tokens, dim, word2vec, words2color)
    generate_color(["stormy", "ocean"], max_tokens, dim, word2vec, words2color)

    generate_color(["red"], max_tokens, dim, word2vec, words2color)
    generate_color(["blue"], max_tokens, dim, word2vec, words2color)
    generate_color(["red", "blue"], max_tokens, dim, word2vec, words2color)

    (color2words, max_tokens, dim) = color_to_words(rgbs, word2vec)

    rgb = np.array([161, 85, 130], dtype = "float32")
    display_color(rgb)
    generate_name(rgb, max_tokens, word2vec, color2words)

    rgb = np.array([1, 159, 11], dtype = "float32")
    display_color(rgb)
    generate_name(rgb, max_tokens, word2vec, color2words)


if __name__ == "__main__":
    main()