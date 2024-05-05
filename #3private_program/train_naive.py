import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn import metrics
import dill as pickle
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

review_stars = []
review_profile = []

with open("dataset.csv", encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        review_stars.append(row["stars"])
        review_profile.append(row["profile"]) # Convert to integer

dataset = pd.DataFrame(data={"prediction_photo": review_profile, "stars": review_stars})

dataset = dataset[(dataset['stars'] == '1') | (dataset['stars'] == '2') | (dataset['stars'] == '3')]


print(dataset)

data = dataset["prediction_photo"]
target = dataset["stars"]

lemmatizer = WordNetLemmatizer()

def pre_processing(text):
  text_processed = text.translate(str.maketrans("","", string.punctuation))
  text_processed = text_processed.split()
  result = []
  for word in text_processed:
    word_processed = word.lower()
    if word_processed not in stopwords.words("english"):
      word_processed = lemmatizer.lemmatize(word_processed)
      result.append(word_processed)
  return result
  
count_vectorize_transformer = CountVectorizer(analyzer=pre_processing).fit(data)

data = count_vectorize_transformer.transform(data)

machine = MultinomialNB()
def run_kfold(machine, data, target, n):
    kfold_object = KFold(n_splits=n)
    all_return_values = []
    for train_index, test_index in kfold_object.split(data):
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        
        machine.fit(data_train, target_train)
        prediction = machine.predict(data_test)
        
        accuracy = metrics.accuracy_score(target_test, prediction)
        all_return_values.append(accuracy)
        
    return all_return_values


return_values = run_kfold(machine, data, target, 4)
print(return_values)

machine = MultinomialNB()
machine.fit(data,target)


with open("text_analysis_machine.pickle", "wb") as f:
  pickle.dump(machine, f)
  pickle.dump(count_vectorize_transformer, f)
  pickle.dump(lemmatizer, f)
  pickle.dump(stopwords, f)
  pickle.dump(string, f)
  pickle.dump(pre_processing, f)

