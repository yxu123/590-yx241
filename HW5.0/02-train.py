from keras import preprocessing
import numpy as np
from keras.models import Sequential 
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import RMSprop
import os

npzfile=np.load(os.path.join('novels.npz'))
#---------------------------
#USER PARAM
#---------------------------
max_features = 10000    #DEFINES SIZE OF VOCBULARY TO USE
maxlen       = 250      #CUTOFF REVIEWS maxlen 20 WORDS)
epochs       = 8
batch_size   = 1000
verbose      = 1
embed_dim    = 8        #DIMENSION OF EMBEDING SPACE (SIZE OF VECTOR FOR EACH WORD)
lr           = 0.001    #LEARNING RATE

#---------------------------
#GET AND SETUP DATA
#---------------------------

x_train = npzfile['x_train']
y_train = npzfile['y_train']
x_test  = npzfile['x_val']
y_test  = npzfile['y_val']
print(x_train[0]) # ,y_train.shape)
print(y_train[0:10]) # ,y_train.shape)

#truncating='pre' --> KEEPS THE LAST 20 WORDS
#truncating='post' --> KEEPS THE FIRST 20 WORDS
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen,truncating='post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,truncating='post')
# print('input_train shape:', x_train.shape)
print(x_train[0][0:10]) # ,y_train.shape)
# print('input_train shape:', x_train.shape)

#PARTITION DATA
rand_indices = np.random.permutation(x_train.shape[0])
CUT=int(0.8*x_train.shape[0]); 
train_idx, val_idx = rand_indices[:CUT], rand_indices[CUT:]
x_val=x_train[val_idx]; y_val=y_train[val_idx]
x_train=x_train[train_idx]; y_train=y_train[train_idx]
print('input_train shape:', x_train.shape)


#---------------------------
#plotting function
#---------------------------
def report(history,title='',I_PLOT=True):

    print(title+": TEST METRIC (loss,accuracy):",model.evaluate(x_test,y_test,batch_size=50000,verbose=verbose))

    if(I_PLOT):
        #PLOT HISTORY
        epochs = range(1, len(history.history['loss']) + 1)
        plt.figure()
        plt.plot(epochs, history.history['loss'], 'bo', label='Training loss')
        plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')

        plt.plot(epochs, history.history['acc'], 'ro', label='Training acc')
        plt.plot(epochs, history.history['val_acc'], 'r', label='Validation acc')

        plt.title(title)
        plt.legend()
        # plt.show()

        plt.savefig('HISTORY-'+title+'.png')   # save the figure to file
        plt.close()

print("---------------------------")
print("DFF (MLP)")  
print("---------------------------")

model = Sequential()
#learn 8-dimensional embeddings for each of the 10,000 words
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="DFF")
model.save('DFF_model.h5')


print("---------------------------")
print("SimpleRNN")  
print("---------------------------")

model = Sequential() 
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.SimpleRNN(32)) 
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 
model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="SimpleRNN")
model.save('SimpleRNN_model.h5')



print("---------------------------")
print("1D-CNN")  
print("---------------------------")

model = Sequential()
model.add(layers.Embedding(max_features, embed_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(lr=lr), loss='binary_crossentropy', metrics=['acc']) 

model.summary()
history = model.fit(x_train, y_train, epochs=epochs,batch_size=batch_size, validation_data=(x_val,y_val),verbose=verbose)
report(history,title="CNN")
model.save('1D-CNN_model.h5')