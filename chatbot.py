import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
#keras va permettre d'acceder aux differentes couches dans un reseaux de neronne
from keras.layers import Dense, Activation, Dropout 
from keras.optimizers import SGD
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
#We load our json file
data_file = open('kenya_facts.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print (len(documents), "documents", documents)
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)
#We save the model of words obtained
pickle.dump(words,open('words.pkl','wb'))
#We generate and save the model of classes. Classes here are the tag element in our json file
pickle.dump(classes,open('classes.pkl','wb')) 
# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
# create train and test lists. X - Patterns, Y - answers
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Data have been trained")
# Create model white 4 layers. 
#First layer 128 neurons, second layer 64 neurons
#The 3rd comprises 32 neurons 
#4th output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model using SGD. 
#Stochastic gradient descent with Nesterov accelerated gradient
# gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the compiled model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=5, verbose=1)
model.save('chat_bot_model.h5', hist)
print("model succefully created")
