import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras import layers
import pickle


dataset = ImageDataGenerator(rescale=1./255, validation_split=0.25)

training_dataset = dataset.flow_from_directory(directory="part_3/training_data/image_dataset", 
                                              shuffle=True,
                                              target_size=(50,50), 
                                              batch_size=32, 
                                              subset='training')
validation_dataset = dataset.flow_from_directory(directory="part_3/training_data/image_dataset", 
                                              shuffle=True,
                                              target_size=(50,50), 
                                              batch_size=32, 
                                              subset='validation')

        
values = list(training_dataset.class_indices.values())
keys = list(training_dataset.class_indices.keys())
print([[values[i], keys[i]] for i in range(len(values))])
num_of_classes = len(values)

machine = Sequential()
machine.add(layers.Conv2D(filters=32, 
            kernel_size=(3,3), 
            padding='same', 
            activation='relu', 
            input_shape=(50,50,3)))
machine.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
machine.add(layers.MaxPooling2D(pool_size=(2,2)))
machine.add(layers.Dropout(0.25))

machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.MaxPooling2D(pool_size=(2,2)))
machine.add(layers.Dropout(0.25))
          
machine.add(layers.Flatten())
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dropout(0.25))
machine.add(layers.Dense(units=num_of_classes, activation='softmax'))
  
machine.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics='accuracy')

machine.fit(x=training_dataset, validation_data=validation_dataset, epochs=10)

pickle.dump(machine, open('cnn_image_machine.pickle', 'wb'))
