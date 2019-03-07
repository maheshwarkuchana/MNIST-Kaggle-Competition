import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.iloc[:, 1:].values.astype('float32') / 255
y_train = train.iloc[:, :1].values.astype('int32')
X_test = test.values.astype('float32') / 255

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model = Sequential([
    Conv2D(32, (5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)),
    Conv2D(32, (5, 5), padding='Same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='Same', activation='relu'),
    Conv2D(64, (3, 3), padding='Same', activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(lr=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                              factor=0.5,
                              patience=3,
                              verbose=1,
                              min_lr=0.00001)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.3)

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=128),
                              epochs=50,
                              validation_data=datagen.flow(X_val, y_val, batch_size=128),
                              validation_steps=20,
                              verbose=1,
                              steps_per_epoch=X_train.shape[0] // 64,
                              callbacks=[reduce_lr])

predictions = model.predict_classes(X_test, verbose=0)

submissions = pd.DataFrame({"ImageId" : list(range(1, len(predictions)+1)), "Label" : predictions})
submissions.to_csv("digit_recognizer.csv", index=False, header=True)