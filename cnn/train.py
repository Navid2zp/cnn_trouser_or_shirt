from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from .consts import IMAGE_SIZE
import pickle


def train():
    # Model the data
    batch_size = 30
    epochs = 20

    pickle_in = open("shirt_or_trouser_X.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("shirt_or_trouser_y.pickle", "rb")
    y = pickle.load(pickle_in)
    pickle_in.close()

    # comment this if you already reshaped your data
    X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

    X = X / 255.0

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(32))
    model.add(Dropout(0.35))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.3)
    # save the model
    model.save("trouser_vs_shirt_v1.h5py")
