from pickle import NONE
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import random
##  hyper-param define
# MNIST, MNIST-Fashion, CIFAR-10
dataset = "MNIST" 
model_type = "CNN"
IMG_CHANNEL = 1
IMG_WIDTH = 28
IMG_HEIGHT = 28
network_layers = [{"layer_type":"Conv2D","params":{"filters":32,"kernel_size":(3, 3),"activation":'relu', "input_shape":(IMG_WIDTH,IMG_HEIGHT, IMG_CHANNEL)}},
                {"layer_type":"MaxPooling2D","params":{"pool_size":(2, 2)}},
                {"layer_type":"Conv2D","params":{"filters":64,"kernel_size":(3, 3),"activation":'relu', "input_shape":None}},
                {"layer_type":"MaxPooling2D","params":{"pool_size":(2, 2)}},
                {"layer_type":"Conv2D","params":{"filters":64,"kernel_size":(3, 3),"activation":'relu', "input_shape":None}},
                {"layer_type":"Flatten"},
                {"layer_type":"Dense","params":{"units":64,"activation":'relu', "input_shape":None}},
                {"layer_type":"Dense","params":{"units":10,"activation":'softmax', "input_shape":None}},
                ]

# CIFAR-10 network_layers
# IMG_CHANNEL = 3
# IMG_WIDTH = 32
# IMG_HEIGHT = 32
# network_layers = [{"layer_type":"Conv2D","params":{"filters":32,"kernel_size":(3, 3),"activation":'relu', "input_shape":(32,32, 3)}},
#                 {"layer_type":"MaxPooling2D","params":{"pool_size":(2, 2)}},
#                 {"layer_type":"Conv2D","params":{"filters":64,"kernel_size":(3, 3),"activation":'relu', "input_shape":None}},
#                 {"layer_type":"MaxPooling2D","params":{"pool_size":(2, 2)}},
#                 {"layer_type":"Conv2D","params":{"filters":64,"kernel_size":(3, 3),"activation":'relu', "input_shape":None}},
#                 {"layer_type":"Flatten"},
#                 {"layer_type":"Dense","params":{"units":64,"activation":'relu', "input_shape":None}},
#                 {"layer_type":"Dense","params":{"units":10,"activation":'softmax', "input_shape":None}},
#                 ]
# model_type = "CNN"

# network_layers = [{"layer_type":layers.Dense,"params":{"units":512,"activation":'relu',"input_shape":(28*28, )}},
#                 {"layer_type":layers.Dense,"params":{"units":10,"activation":'softmax',"input_shape":None}},
#                 ]


data_augmentation = False
optimizer = "rmsprop"
loss = "categorical_crossentropy"
metrics = ['accuracy']



model = models.Sequential()


def buid_model():
    for layer in network_layers:
        print(layer)
        if layer["layer_type"] == "Conv2D":
            if layer["params"]["input_shape"] != None:
                model.add(layers.Conv2D(layer["params"]["filters"], layer["params"]["kernel_size"], 
                    activation=layer["params"]["activation"] , input_shape=layer["params"]["input_shape"]))
            else:
                model.add(layers.Conv2D(layer["params"]["filters"], layer["params"]["kernel_size"], 
                    activation=layer["params"]["activation"]))
        elif layer["layer_type"] == "MaxPooling2D":
            model.add(layers.MaxPooling2D(layer["params"]["pool_size"]))
        elif layer["layer_type"] == "Flatten":
            model.add(layers.Flatten())
        elif layer["layer_type"] == "Dense":
            if layer["params"]["input_shape"] != None:
                model.add(layers.Dense(layer["params"]["units"], activation=layer["params"]["activation"] , input_shape=layer["params"]["input_shape"]))
            else:
                model.add(layers.Dense(layer["params"]["units"], activation=layer["params"]["activation"]))


buid_model()

model.summary()






from tensorflow.keras.utils import to_categorical

if dataset == "MNIST":  
    from keras.datasets import mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if dataset == "MNIST-Fashion":  
    from keras.datasets import fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
if dataset == "CIFAR-10":  
    from keras.datasets import cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()



# Add function to visualize a random (or specified) image in the dataset

def visaulize_random():
    index = random.randint(0,len(train_images))
    plt.imshow(train_images[index], cmap=plt.cm.gray); plt.show()

visaulize_random()
train_images = train_images.reshape((60000, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer=optimizer,
    loss=loss,
    metrics=metrics)

if data_augmentation:
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    datagen.fit(train_images)

model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_loss, test_acc)

# save model
def save_model(filename):
    model.save(filename)

# read a model from a file
def load_model(filename):
    from keras.models import load_model
    return load_model(filename)
