import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Load Titanic dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train, y_eval = dftrain.pop('survived'), dfeval.pop('survived')

# Define feature columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for col in CATEGORICAL_COLUMNS:
    vocab = dftrain[col].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(col, vocab))

for col in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(col, dtype=tf.float32))

# Input function
def make_input_fn(data, labels, shuffle=True, batch_size=32, num_epochs=10):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data), labels))
        if shuffle: ds = ds.shuffle(1000)
        return ds.batch(batch_size).repeat(num_epochs)
    return input_function

# Prepare input functions
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, num_epochs=1)

# Create Linear Classifier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

# Print results
print("Evaluation Accuracy:", result['accuracy'])

# Make Predictions
pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

# Plot predicted probabilities
probs.plot(kind='hist', bins=20, title='Predicted Probabilities')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_eval, probs)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
