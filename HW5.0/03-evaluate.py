
import numpy as np
import os
from keras.models import load_model
from keras import preprocessing
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

x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen,truncating='post')

models = ['DFF_model.h5','SimpleRNN_model.h5','1D-CNN_model.h5']

for model_file in models:
    model = load_model(model_file)
    print(model_file +": TEST METRIC (loss,accuracy):",model.evaluate(x_test,y_test))