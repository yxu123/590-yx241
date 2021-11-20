import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers

title = "HW6.3"
#GET DATASET
from keras.datasets import cifar10
(X, Y), (test_images, test_labels) = cifar10.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000, 28, 28, 1); 

input_img = layers.Input(shape=(28, 28, 1))
enc_conv1 = layers.Conv2D(12, (3, 3), activation='relu', padding='same')(input_img)
enc_pool1 = layers.MaxPooling2D((2, 2), padding='same')(enc_conv1)
enc_conv2 = layers.Conv2D(8, (4, 4), activation='relu', padding='same')(enc_pool1)
enc_ouput = layers.MaxPooling2D((4, 4), padding='same')(enc_conv2)

dec_conv2 = layers.Conv2D(8, (4, 4), activation='relu', padding='same')(enc_ouput)
dec_upsample2 = layers.UpSampling2D((4, 4))(dec_conv2)
dec_conv3 = layers.Conv2D(12, (3, 3), activation='relu')(dec_upsample2)
dec_upsample3 = layers.UpSampling2D((2, 2))(dec_conv3)
dec_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(dec_upsample3)

model = models.Model(input_img, dec_output)
model.compile(optimizer='rmsprop', loss='binary_crossentropy') 
 
model.summary()

history = model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)

epochs = range(1, len(history.history['loss']) + 1)
plt.figure()
plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

plt.title(title)
plt.legend()
# plt.show()

plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
plt.close()


from keras.datasets import cifar100
(x_cifar100, y_cifar100), (x_cifar100_test, y__cifar100test) = cifar100.load_data()


#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(x_cifar100) 

#RESHAPE
x_cifar100=x_cifar100.reshape(60000,28,28,1); #print(X[0])
X1=X1.reshape(60000,28,28,1); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(x_cifar100[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(x_cifar100[I2])
ax[3].imshow(X1[I2])
plt.savefig("hw6.3-anomaly-detection.png")

