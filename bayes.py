import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

df = pd.read_csv('../sentiment_data.csv', encoding='ISO-8859-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

df['text_clean'] = df['text'].apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))

df['target'] = df['target'].map({0: 0, 4: 2})

X_train, X_test, y_train, y_test = train_test_split(df['text_clean'], df['target'], test_size=0.2, random_state=42)
# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predictions
predictions = model.predict(X_test_vect)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True)
print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))

# Save the model and metrics
joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Extract metrics
metrics = {
    'accuracy': report['accuracy'],
    'weighted avg': report['weighted avg']
}

with open('bayes_metrics.json', 'w') as f:
    json.dump(metrics, f)

# Save the report as JSON
with open('bayes_classification_report.json', 'w') as f:
    json.dump(report, f)

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)
class_labels = ['Negative', 'Positive']  # Assuming 0 for Negative, 2 for Positive based on your target mapping

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()