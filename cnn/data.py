import numpy as np
from tqdm import tqdm
from .consts import IMAGE_SIZE
import pickle
import random
import os
import cv2

MIN_WHITES = {
    "Trouser": 0.40,
    "Shirt": 0.30,
}


# there are closeup images mostly to see the fabric material
# or maybe the pattern which we don't need
# TODO: there are helper images for sizes which should be removed too
def is_clean(data, data_type):
    return np.count_nonzero(data == 255) / np.count_nonzero(data >= 0) > MIN_WHITES[data_type]


def create_data():
    DATADIR = "images"

    CATEGORIES = ["Trouser", "Shirt"]
    trousersPath = os.path.join(DATADIR, "Trousers")
    shirtsPath = os.path.join(DATADIR, "Shirts")

    trousers = os.listdir(trousersPath)
    shirts = os.listdir(shirtsPath)
    shirts = random.choices(shirts, k=len(trousers))

    training_data = []

    # shirts
    for img in tqdm(shirts):  # iterate over each image
        try:
            img_array = cv2.imread(os.path.join(shirtsPath, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))  # resize to normalize data size
            if is_clean(new_array, "Shirt"):
                training_data.append([new_array, CATEGORIES.index("Shirt")])
        except Exception as e:  # in case some error happened with few images
            pass

    # trousers
    for img in tqdm(trousers):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(trousersPath, img), cv2.IMREAD_GRAYSCALE)  # convert to array
            new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))  # resize to normalize data size
            if is_clean(new_array, "Trouser"):
                training_data.append([new_array, CATEGORIES.index("Trouser")])
        except Exception as e:
            pass

    # shuffle the data
    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    pickle_out = open("shirt_or_trouser_X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("shirt_or_trouser_y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
