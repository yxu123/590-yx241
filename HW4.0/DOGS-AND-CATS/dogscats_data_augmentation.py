from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

# ln -s /home/xx/cats_and_dogs_small ./cats_and_dogs_small
# The training data set is divided into training set, verification set and test set. The directory structure like this:
# 
# - train
# --- dogs
# --- cats
# - validation
# --- dogs
# --- cats
# - test
# --- dogs
# --- cats
train_dir = './cats_and_dogs_small/train'
validation_dir = './cats_and_dogs_small/validation'

# build model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3))) # conv2d_1 (Conv2D) (None, 148, 148, 32) 896

model.add(layers.MaxPool2D((2, 2))) # maxpooling2d_1 (MaxPooling2D) (None, 74, 74, 32) 0

model.add(layers.Conv2D(64, (3, 3), activation='relu')) # conv2d_2 (Conv2D) (None, 72, 72, 64) 18496
model.add(layers.MaxPool2D((2, 2))) # maxpooling2d_2 (MaxPooling2D) (None, 36, 36, 64) 0

model.add(layers.Conv2D(128, (3, 3), activation='relu')) # 
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# Rescales all images by 1/255
# Read picture files;
# Decoding JPG into RGB pixels;
# Converting these pixels into floating-point tensors;
# Reduce the pixel value in the [0,255] interval to the [0,1] interval. CNN prefers to deal with small input values.

# By randomly changing the pictures that have been "remembered" by the model, the model will not see the same picture twice by scaling, cropping and stretching the picture
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
    
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# Use fit_generator populates the model with data
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

# save model to file 
model.save('cats_and_dogs_small_2.h5')

# Show the curve of loss and ACC in training
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
