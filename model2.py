# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 13:08:33 2019

@author: intel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 23:28:46 2019

@author: intel
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
#import numpy as np
#training_label = []
'''with open('D:/Foodproject/food-101/meta/test.txt') as f:
    lines = f.readlines()
    print('starting')
    for line in lines:
        training_label.append(line)
f.close()
tl = np.array(training_label)'''
#print(training_label)

classifier = Sequential()

classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu',padding='same')) #306,512
classifier.add(Convolution2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(64,(3,3),activation='relu',padding='same')) #306,512
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Convolution2D(64,(3,3),activation='relu',padding='same')) #306,512
classifier.add(Convolution2D(64,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(output_dim=512,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(output_dim=101,activation='softmax'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('D:/Foodproject/food-101/train/',color_mode='rgb',target_size=(64,64),batch_size=32)
test_set = test_datagen.flow_from_directory('D:/Foodproject/food-101/test/',color_mode='rgb',target_size=(64,64),batch_size=32)

i = 0
img_list = []
#for batch in test_set: #.flow(x, batch_size=1)
#    img_list.append(batch)
#    i += 1
#    print(batch.shape)
#    if i > 5:
#        break

print(test_set)

from IPython.display import display
from PIL import Image
print('trianing began')
classifier.fit_generator(training_set,steps_per_epoch=1000,epochs=10,validation_data=test_set,validation_steps=800)
print('trianing done')









