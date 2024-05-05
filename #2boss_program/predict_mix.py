import pandas 
import dill as pickle
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split


dataset = pandas.read_csv('sample_new.csv')

dataset_80_percent, dataset_20_percent = train_test_split(dataset, test_size=0.2, random_state=42)
dataset_80_percent.to_csv('dataset_80_percent_predict.csv', index=False)
dataset_20_percent.to_csv('dataset_20_percent_predict.csv', index=False)

with open("text_analysis_machine_naive_bayesian80%.pickle", "rb") as f:
    machine = pickle.load(f)
    count_vectorize_transformer = pickle.load(f)
    lemmatizer = pickle.load(f)
    stopwords = pickle.load(f)
    string = pickle.load(f)
    pre_processing = pickle.load(f)



new_reviews = pandas.read_csv("dataset_80_percent_predict.csv") 
new_reviews_tranformed = count_vectorize_transformer.transform(new_reviews.iloc[:,1])
print(new_reviews_tranformed)


prediction = machine.predict(new_reviews_tranformed)
prediction_prob = machine.predict_proba(new_reviews_tranformed)
print(prediction)
print(prediction_prob)

new_reviews['prediction'] = prediction
prediction_prob_dataframe = pandas.DataFrame(prediction_prob)


prediction_prob_dataframe = prediction_prob_dataframe.rename(columns={
  prediction_prob_dataframe.columns[0]: "prediction_prob_1",
  prediction_prob_dataframe.columns[1]: "prediction_prob_2",
  prediction_prob_dataframe.columns[2]: "prediction_prob_3"
  })


new_reviews = pandas.concat([new_reviews,prediction_prob_dataframe], axis=1)
print(new_reviews)
new_reviews = new_reviews.rename(columns={
 new_reviews.columns[1]: "text"
  })

new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_prob_1'] = round(new_reviews['prediction_prob_1'],4)
new_reviews['prediction_prob_2'] = round(new_reviews['prediction_prob_2'],4)
new_reviews['prediction_prob_3'] = round(new_reviews['prediction_prob_3'],4)

new_reviews.to_csv("sample_new_with_stars_80_percent.csv", index=False)

data = pandas.read_csv('dataset_20_percent_predict.csv')

load_dotenv()
client = OpenAI()

def simple_call(prompt):
  completions = client.chat.completions.create(model="gpt-3.5-turbo", 
            messages=[
              {"role": "user", "content": prompt},
            ], max_tokens=200
            , temperature=0.1
            , top_p=1
            )
  return completions.choices[0].message.content
  

data['predicted_stars'] = data['reviewtext'].apply(lambda x: simple_call("evaluate each review on a scale of 1 (bad), 2 (average), or 3 (good) based on the quality of the programmer described in the review: \"" + x + "\", answer in one number."   ))
data.to_csv("sample_new_with_stars_20_percent.csv", index=False)


