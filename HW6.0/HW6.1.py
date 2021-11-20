import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from keras import models
from keras import layers

#GET DATASET
from keras.datasets import mnist
(X, Y), (test_images, test_labels) = mnist.load_data()

#NORMALIZE AND RESHAPE
X=X/np.max(X) 
X=X.reshape(60000,28*28); 


n_bottleneck=10 

model = models.Sequential()
model.add(layers.Dense(100, input_shape=(28*28,), activation="relu"))
# model.add(layers.Dense(n_bottleneck, activation="relu"))
# model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(28*28, activation="linear"))


#COMPILE AND FIT
model.compile(optimizer='rmsprop',
                loss='mean_squared_error')
model.summary()
history = model.fit(X, X, epochs=10, batch_size=1000,validation_split=0.2)



def report(history,title='',I_PLOT=True):
    
    # print(title+": TEST METRIC (loss,accuracy):",model.evaluate(test_images,test_labels,batch_size=50000,verbose=1))

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

        # plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        # plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

        plt.title(title)
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.close()

report(history,'HW6.1')

from keras.datasets import fashion_mnist
(x_fashion, y_fashion), (x_fashion_test, y__fashiontest) = fashion_mnist.load_data()


x_fashion=x_fashion/np.max(x_fashion) 
x_fashion=x_fashion.reshape(60000,28*28); 

# encoded_imgs = model.encoder(x_fashion).numpy()
# decoded_imgs = model.decoder(encoded_imgs).numpy()

# plt.plot(x_fashion[0],'b')
# plt.plot(decoded_imgs[0],'r')
# plt.fill_between(np.arange(140), decoded_imgs[0], x_fashion[0], color='lightcoral' )
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()

# (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
#EXTRACT MIDDLE LAYER (REDUCED REPRESENTATION)
from keras import Model 
extract = Model(model.inputs, model.layers[-2].output) # Dense(128,...)

extract.summary()
X1 = extract.predict(x_fashion)
print(X1.shape)

#2D PLOT
plt.scatter(X1[:,0], X1[:,1], c=Y, cmap='tab10')
# plt.show()
plt.savefig("hw6.1-2D.png")
#3D PLOT
ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=X1[:,0], 
    ys=X1[:,1], 
    zs=X1[:,2], 
    c=Y, 
    cmap='tab10'
)
# plt.show()
plt.savefig("hw6.1-3d.png")
#PLOT ORIGINAL AND RECONSTRUCTED 
X1=model.predict(x_fashion) 

#RESHAPE
x_fashion=x_fashion.reshape(60000,28,28); #print(X[0])
X1=X1.reshape(60000,28,28); #print(X[0])

#COMPARE ORIGINAL 
f, ax = plt.subplots(4,1)
I1=11; I2=46
ax[0].imshow(x_fashion[I1])
ax[1].imshow(X1[I1])
ax[2].imshow(x_fashion[I2])
ax[3].imshow(X1[I2])
plt.show()

plt.savefig("hw6.1-anomaly-detection.png")

