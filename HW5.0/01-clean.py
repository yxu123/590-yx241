
import os
import requests

books = {
    "Australasia Triumphant": 'https://www.gutenberg.org/ebooks/66658.txt.utf-8',
    "A New Story Book for Children": 'https://www.gutenberg.org/ebooks/66655.txt.utf-8',
    "La petite femme de la mer by Camille Lemonnier": 'https://www.gutenberg.org/ebooks/66652.txt.utf-8',
}




def break_the_novels(content):
    leave_content = content
    index = 0
    lines = 0

    broken = []
    start = 0
    end = 0
    try:
        while True:
            index = content.find('\n',index+1)

            if index < 0:
                break
            lines += 1
            if lines < 50: # jump head
                start = index
                end = index
                continue
            end = index

            if end - start > 1000:
                broken.append(content[start:end])
                start = end
    except:
        pass

    return broken
        


labels = []
texts = []
for label, book in enumerate(books.keys()):
    print("download " + book + "(" + books[book] + ")")
    # novels = requests.get(books[book],verify=False)
    f = open(os.path.join(book + ".txt"), encoding='utf-8')
    novels = f.read()
    for text in break_the_novels(novels):
        texts.append(text)
        labels.append(label)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen = 100
training_samples = int(len(labels) * 0.8)
validation_samples = len(labels) - training_samples
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

print(x_train[0:10])
print(y_train[0:10])


import numpy as np

np.savez("novels.npz",x_train=x_train,y_train=y_train,x_val=x_val,y_val=y_val)

