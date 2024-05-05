import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import json
import pandas as pd
import csv
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import dill as pickle
from sklearn.metrics import accuracy_score
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import os

nltk.download("stopwords")
nltk.download("wordnet")

dataset = pd.read_csv('dataset.csv')

dataset_80_percent, dataset_20_percent = train_test_split(dataset, test_size=0.2, random_state=42)

dataset_80_percent.to_csv('dataset_80_percent.csv', index=False)
dataset_20_percent.to_csv('dataset_20_percent.csv', index=False)


review_stars = []
review_text = []
with open("dataset_80_percent.csv", encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        review_stars.append(row["stars"])
        review_text.append(row["reviewtext"])

dataset = pd.DataFrame(data={"text": review_text, "stars": review_stars})

#dataset = dataset.iloc[0:]

dataset = dataset[(dataset['stars'] == '1') | (dataset['stars'] == '2') | (dataset['stars'] == '3')]

print(dataset)

data = dataset["text"]
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


with open("text_analysis_machine_naive_bayesian80%.pickle", "wb") as f:
  pickle.dump(machine, f)
  pickle.dump(count_vectorize_transformer, f)
  pickle.dump(lemmatizer, f)
  pickle.dump(stopwords, f)
  pickle.dump(string, f)
  pickle.dump(pre_processing, f)



load_dotenv()
client = OpenAI()
dataset = pd.read_csv('dataset_20_percent.csv')

data = dataset['reviewtext']
target = dataset['stars']

def evaluate_review(reviewtext):
    prompt = "Evaluate this review on a scale of 1 (bad), 2 (average), or 3 (good) based on the quality of the programmer described: \"" + reviewtext + "\", answer in one number."
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.1,
        top_p=1
    )
    result = completions.choices[0].message.content.strip()  # Remove leading/trailing whitespace
    if result in ['1', '2', '3']:
        return int(result)  # Convert valid responses to integers
    else:
        return None  

def run_kfold(data, target, n_splits=5):
    kfold_object = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_return_values = []
    
    for train_index, test_index in kfold_object.split(data):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        target_train, target_test = target.iloc[train_index], target.iloc[test_index]
        y_pred_openai = [evaluate_review(review) for review in data_test]
        y_pred_openai = [x for x in y_pred_openai if x is not None]
        accuracy = accuracy_score(target_test[:len(y_pred_openai)], y_pred_openai)
        all_return_values.append(accuracy)
    return all_return_values

return_values_reviewtext = run_kfold(data, target, n_splits=4)
print("Accuracy scores using reviewtext (original features):", return_values_reviewtext)

dataset['predicted_stars'] = data.apply(lambda x: evaluate_review(x))
dataset.to_csv('dataset_processed_ataset_20_percent.csv', index=False)

