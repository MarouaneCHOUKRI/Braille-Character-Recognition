import oss
from shutil import copyfile
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


os.mkdir('./images/')
alpha = 'a'
for i in range(0, 26):
    os.mkdir('./images/' + alpha)
    alpha = chr(ord(alpha) + 1)

rootdir = './Braille Dataset/'
for file in os.listdir(rootdir):
    letter = file[0]
    copyfile(rootdir+file, './images/' + letter + '/' + file)

datagen = ImageDataGenerator(rotation_range=20,
                             shear_range=10,
                             validation_split=0.05)

train_generator = datagen.flow_from_directory('./images/',
                                              target_size=(28, 28),
                                              subset='training')

val_generator = datagen.flow_from_directory('./images/',
                                            target_size=(28, 28),
                                            subset='validation')


model_ckpt = ModelCheckpoint('braille.h5',save_best_only=True)
braille_model = Sequential()
braille_model.add(Conv2D(16, kernel_size=(5, 5), activation='linear', input_shape=(28, 28, 3), padding='same', strides=1))
braille_model.add(LeakyReLU(alpha=0.1))
braille_model.add(MaxPooling2D((2, 2)))
braille_model.add(Conv2D(32, kernel_size=(5, 5), activation='linear', padding='same', strides=1))
braille_model.add(LeakyReLU(alpha=0.1))
braille_model.add(MaxPooling2D(pool_size=(2, 2)))
braille_model.add(Conv2D(64, kernel_size=(5, 5), activation='linear', padding='same', strides=1))
braille_model.add(LeakyReLU(alpha=0.1))
braille_model.add(MaxPooling2D(pool_size=(2, 2)))
braille_model.add(Conv2D(128, kernel_size=(5, 5), activation='linear', padding='same', strides=1))
braille_model.add(LeakyReLU(alpha=0.1))
braille_model.add(MaxPooling2D(pool_size=(2, 2)))
braille_model.add(Flatten())
braille_model.add(Dense(256, activation='linear'))
braille_model.add(LeakyReLU(alpha=0.1))
braille_model.add(Dense(26, activation='softmax'))
braille_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
braille_model.summary()

braille_train = braille_model.fit(train_generator, batch_size=32, epochs=45, verbose=1,
                                  validation_data=(val_generator),
                                  callbacks=[model_ckpt])


plt.plot(braille_train.history['loss'], label='train loss')
plt.plot(braille_train.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(braille_train.history['accuracy'], label='train acc')
plt.plot(braille_train.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')
