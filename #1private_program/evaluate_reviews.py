import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI()
dataset = pd.read_csv('dataset.csv')

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
dataset.to_csv('dataset_processed.csv', index=False)
