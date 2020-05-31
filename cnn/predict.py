from keras.models import load_model
from .consts import IMAGE_SIZE
import numpy as np
import cv2


CATEGORIES = ["Trouser", "Shirt"]


def predict(image_path):
    model = load_model('trouser_vs_shirt_v1.h5py')
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    data = np.array(new_array).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    return CATEGORIES[int(model.predict(data)[0][0])]
