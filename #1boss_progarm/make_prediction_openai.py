import pandas
from openai import OpenAI
from dotenv import load_dotenv
import os

dataset = pandas.read_csv('sample_new.csv')

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
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
  

dataset['predicted_stars'] = dataset['reviewtext'].apply(lambda x: simple_call("evaluate each review on a scale of 1 (bad), 2 (average), or 3 (good) based on the quality of the programmer described in the review: \"" + x + "\", answer in one number."   ))
dataset.to_csv('dataset_predict_openai.csv', index=False)
