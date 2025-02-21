import tensorflow as tf
import pandas as pd
import numpy as np

# Load the dataset
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

# Separate features and labels
train_y = train.pop('Species')
test_y = test.pop('Species')

# Define Input Function
def input_fn(features, labels=None, training=True, batch_size=32):
    """Input function for training/evaluation."""
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels)) if labels is not None else tf.data.Dataset.from_tensor_slices(dict(features))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

# Define Feature Columns
feature_columns = [tf.feature_column.numeric_column(key=key) for key in train.keys()]

# Build DNN Classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 10],  # Two hidden layers
    n_classes=3
)

# Train the Model
classifier.train(input_fn=lambda: input_fn(train, train_y, training=True), steps=5000)

# Evaluate the Model
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print(f'\nTest Accuracy: {eval_result["accuracy"]:.3f}\n')

# Prediction Function
def predict():
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    expected = ['Setosa', 'Versicolor', 'Virginica']

    predictions = classifier.predict(input_fn=lambda: input_fn(predict_x, training=False))

    for pred, exp in zip(predictions, expected):
        class_id = pred['class_ids'][0]
        probability = pred['probabilities'][class_id]
        print(f'Predicted: {SPECIES[class_id]} ({probability*100:.1f}%), Expected: {exp}')

# Run Predictions
predict()
