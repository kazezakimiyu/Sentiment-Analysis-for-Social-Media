import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_report(file_path, model_name):
    with open(file_path, 'r') as file:
        data = json.load(file)
    rows = []
    for class_label, metrics in data.items():
        if class_label in ['Negative', 'Neutral', 'Positive']:
            row = {
                'Model': model_name,
                'Class': class_label,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1-score']
            }
            rows.append(row)
    return pd.DataFrame(rows)

def load_overall_metrics(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {
        'Model': file_path.split('_')[0],  # Adjust this extraction as necessary
        'Accuracy': data['accuracy'],
        'Precision': data['weighted avg']['precision'],
        'Recall': data['weighted avg']['recall'],
        'F1-Score': data['weighted avg']['f1-score']
    }

# Load and concatenate classification reports
df_lstm = load_report('lstm_classification_report.json', 'LSTM')
df_nb = load_report('naive_bayes_classification_report.json', 'Naive Bayes')
df_logreg = load_report('logistic_regression_classification_report.json', 'Logistic Regression')
df_class_metrics = pd.concat([df_lstm, df_nb, df_logreg], ignore_index=True)

# Load overall metrics
overall_metrics = [
    load_overall_metrics('lstm_metrics.json'),
    load_overall_metrics('naive_bayes_metrics.json'),
    load_overall_metrics('logistic_regression_metrics.json')
]
df_overall_metrics = pd.DataFrame(overall_metrics)

# Plot detailed analysis per class
plt.figure(figsize=(12, 8))
df_class_melted = df_class_metrics.melt(['Model', 'Class'], var_name='Metrics', value_name='Values')
sns.barplot(x='Class', y='Values', hue='Metrics', data=df_class_melted, palette='deep', ci=None)
plt.title('Precision, Recall, and F1-Score per Class for Each Model')
plt.ylabel('Score')
plt.show()

# Plot overall model comparison
plt.figure(figsize=(10, 6))
df_overall_melted = df_overall_metrics.melt('Model', var_name='Metrics', value_name='Values')
sns.barplot(x='Model', y='Values', hue='Metrics', data=df_overall_melted)
plt.title('Overall Comparison of Model Performance')
plt.ylabel('Score')
plt.show()
