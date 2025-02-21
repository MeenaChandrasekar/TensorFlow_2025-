import tensorflow as tf
import tensorflow_datasets as tfds

# Define a simple compressed Dense layer
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units, name="dense"):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape[-1], self.units])
        self.bias = self.add_weight("bias", shape=[self.units])

    def call(self, inputs):
        return tf.nn.leaky_relu(tf.matmul(inputs, self.kernel) + self.bias)

# Load & preprocess MNIST dataset
def load_data():
    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255.0, label

    train_ds, test_ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)
    train_ds = train_ds.map(normalize).batch(128).prefetch(8)
    test_ds = test_ds.map(normalize).batch(128)
    return train_ds, test_ds

# Build a compressed model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        CustomDense(128),
        CustomDense(64),
        CustomDense(10)
    ])
    return model

# Train the model
def train_model(model, train_ds, test_ds):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=3, validation_data=test_ds)
    model.save("compressed_model.h5")

# Load & evaluate the compressed model
def evaluate_model():
    model = tf.keras.models.load_model("compressed_model.h5", custom_objects={"CustomDense": CustomDense})
    _, accuracy = model.evaluate(test_ds)
    print(f"✅ Accuracy after compression: {accuracy:.4f}")

if __name__ == "__main__":
    train_ds, test_ds = load_data()
    model = build_model()
    train_model(model, train_ds, test_ds)
    evaluate_model()
