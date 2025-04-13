# --- train.py ---

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
from keras import models, layers
from keras.optimizers import Adam
from preprocess import preprocess_data
from augment import data_augmentation

def build_model():
    net = models.Sequential([
        layers.Rescaling(1./1, input_shape=(66, 200, 3)),
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])
    return net

def optimize_model(X_train, X_test, y_train, y_test, net, aug):
    opt = Adam(learning_rate=0.0001)
    net.compile(optimizer=opt, loss='mse', metrics=['mae'])

    H = net.fit(aug.flow(X_train, y_train, batch_size=64),
                epochs=20,
                validation_data=(X_test, y_test))

    plt.plot(H.history['loss'], label='Train Loss')
    plt.plot(H.history['val_loss'], label='Val Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return net

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("CVI-Project/driving_log.csv")
    X_train_aug, y_train_aug, aug = data_augmentation(X_train, y_train)
    net = build_model()
    net = optimize_model(X_train_aug, X_test, y_train_aug, y_test, net, aug)
    net.save('CVI-Project/self_driving_model.keras')
