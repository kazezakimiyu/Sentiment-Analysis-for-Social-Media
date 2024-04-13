import pandas as pd
import re
import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('./sentiment_data.csv', encoding='ISO-8859-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df['text_clean'] = df['text'].apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))
df['target'] = df['target'].map({0: 0, 2: 1, 4: 2})

# Preprocess your data (assuming it's already preprocessed as `text_clean` and `target` columns)
X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['target'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vect, y_train)

# Predictions
predictions = model.predict(X_test_vect)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Save the model and vectorizer
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Extract metrics
metrics = {
    'accuracy': report['accuracy'],
    'precision': report['weighted avg']['precision'],
    'recall': report['weighted avg']['recall'],
    'f1-score': report['weighted avg']['f1-score']
}
'''
# Print and save metrics
with open('logistic_regression_metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f'{key}: {value}\n')
'''
with open('logistic_regression_classification_report.json', 'w') as f:
    json.dump(report, f)