import pandas as pd
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import json
import numpy as np

df = pd.read_csv('./sentiment_data.csv', encoding='ISO-8859-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

df['text_clean'] = df['text'].apply(lambda x: ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", x).split()))

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text_clean'])
sequences = tokenizer.texts_to_sequences(df['text_clean'])

max_length = max([len(x) for x in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

print("Unique target labels before mapping:", df[0].unique()) #suppose to return 3

df['target'] = df['target'].map({0: 0, 2: 1, 4: 2})
print("Unique target labels after mapping:", df['target'].unique())

targets = to_categorical(df['target'], num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, targets, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Convert the probabilities to class labels
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate classification report
unique_classes = np.unique(y_true)
class_names = ['Negative', 'Neutral', 'Positive']
active_class_names = [class_names[i] for i in unique_classes]
report = classification_report(y_true, y_pred, target_names=active_class_names, output_dict=True)
print('\nTest accuracy:', test_acc)
print('\nClassification Report:\n', classification_report(y_true, y_pred, target_names=active_class_names))

# Save the LSTM model
model.save('sentiment_lstm_model.h5')

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the metrics
metrics = {
    'accuracy': report['accuracy'],
    'precision': report['weighted avg']['precision'],
    'recall': report['weighted avg']['recall'],
    'f1-score': report['weighted avg']['f1-score']
}
'''
with open('lstm_metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f'{key}: {value}\n')
'''

# Save the report as JSON
with open('lstm_classification_report.json', 'w') as f:
    json.dump(report, f)