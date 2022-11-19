import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras import Model

from trainingplot import TrainingPlot


class VGG_Utils:
    def __init__(self):
        self.model = None
        self.img_size = 224
        self.plot_losses = TrainingPlot()

    # Loading the model
    def load_model(self, path):
        if os.path.exists(path):
            self.model = load_model(filepath=path)
        else:
            return 0

    def start_training(self, images, classes, epochs, learning_rate, batch_size, test_size):
        # Making the labels in to categorical form
        classes = np_utils.to_categorical(classes)

        # Train and test split
        X_train, X_test, y_train, y_test = train_test_split(
            images, classes, test_size=test_size, random_state=10)

        # Display the dataset splition amount for train and test
        print("X_train", np.shape(X_train))
        print("X_test", np.shape(X_test))

        # Input and output size
        img_input = Input(shape=(self.img_size, self.img_size, 3))

        # Model Initialization
        self.model = VGG16(
            include_top=True,
            weights="imagenet",
            input_tensor=img_input,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax",
        )

        # Adding the last layer as ouput layer
        last_layer = self.model.get_layer('block5_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dense(64, activation='relu', name='fc2')(x)
        out = Dense(len(os.listdir("Captured Images")),
                    activation='sigmoid', name='output')(x)
        self.model = Model(img_input, out)
        for layer in self.model.layers[:-3]:
            layer.trainable = False

        opt = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compiling the model for the multi classes
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['acc'])

        self.model.summary()

        # Callback Function to save the model at each epoch
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        my_callbacks = [
            EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True),
            ModelCheckpoint(filepath='Model/vgg16_model.h5.h5',
                            save_best_only=True),
            self.plot_losses,
        ]

        # Starting the training
        history = self.model.fit(np.array(X_train), np.array(y_train),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_data=(
                                     np.array(X_test), np.array(y_test)),
                                 callbacks=my_callbacks)

    def predict(self):
        pass
