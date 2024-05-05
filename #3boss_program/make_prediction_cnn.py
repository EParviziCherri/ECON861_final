import pickle
import pandas
from keras.preprocessing.image import ImageDataGenerator

import numpy
import glob

machine = pickle.load(open('cnn_image_machine.pickle', 'rb'))

new_data = ImageDataGenerator(rescale=1./255)
new_data = new_data.flow_from_directory(directory="part_3/sample_new_data",
                            shuffle=False, 
                            target_size=(50,50), 
                            batch_size=1)

new_data.reset()

new_data_length = len([i for i in glob.glob('part_3/sample_new_data/sample_profile_pictures/*.jpg')])

prediction = numpy.argmax(machine.predict(new_data, steps=new_data_length), axis=1)
print(prediction)

results = [[new_data.filenames[i], prediction[i]] for i in range(new_data_length)]
results_dataframe = pandas.DataFrame(results, columns=['image', 'prediction_photo'])
results_dataframe.to_csv('predictions.csv', index=False)


dataset = pandas.read_csv('predictions.csv')
def extract_number(filename):
    filename_part = filename.split('/')[-1]
    number = filename_part.split('.')[0]
    return int(number)

dataset['image_number'] = dataset['image'].apply(extract_number)
dataset_sorted = dataset.sort_values(by='image_number')
dataset_sorted.drop(columns='image_number', inplace=True)
print(dataset_sorted)
dataset_sorted.to_csv('predictions_sorted.csv', index=False)

dataset1 = pandas.read_csv('predictions_sorted.csv')
dataset2 = pandas.read_csv('sample_new.csv')

dataset1_subset = dataset1.iloc[:, [1]]  
dataset2_subset = dataset2.iloc[:, [0, 1]]  

combined_subset = pandas.concat([dataset2_subset, dataset1_subset], axis=1)

combined_subset.to_csv('sample_new.csv', index=False)

df = pandas.read_csv('sample_new.csv')

mapping_dict = {0: 'building', 1: 'dog', 2: 'face'}

# Replace values in the 'prediction_photo' column using the mapping dictionary
df['prediction_photo'] = df['prediction_photo'].map(mapping_dict)

# Save the updated DataFrame back to 'sample_new.csv' file
df.to_csv('sample_new.csv', index=False)
