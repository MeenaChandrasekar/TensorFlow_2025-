import tensorflow as tf
import tensorflow_compression as tfc
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

# Load MNIST dataset
(training_dataset, validation_dataset) = tfds.load(
    "mnist", split=["train", "test"], as_supervised=True, shuffle_files=True
)

def preprocess(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

training_dataset = training_dataset.map(preprocess).batch(32)
validation_dataset = validation_dataset.map(preprocess).batch(32)

# Define Encoder & Decoder

def make_analysis_transform(latent_dims):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(20, 5, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Conv2D(50, 5, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(latent_dims)
    ])

def make_synthesis_transform():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(2450, activation="relu"),
        tf.keras.layers.Reshape((7, 7, 50)),
        tf.keras.layers.Conv2DTranspose(20, 5, strides=2, activation="relu", padding="same"),
        tf.keras.layers.Conv2DTranspose(1, 5, strides=2, activation="sigmoid", padding="same"),
    ])

# Define Compression Trainer

class MNISTCompressionTrainer(tf.keras.Model):
    def __init__(self, latent_dims):
        super().__init__()
        self.analysis_transform = make_analysis_transform(latent_dims)
        self.synthesis_transform = make_synthesis_transform()
        self.prior_log_scales = tf.Variable(tf.zeros((latent_dims,)))

    @property
    def prior(self):
        return tfc.NoisyLogistic(loc=0., scale=tf.exp(self.prior_log_scales))

    def call(self, x, training):
        x = tf.reshape(x, (-1, 28, 28, 1))
        y = self.analysis_transform(x)
        entropy_model = tfc.ContinuousBatchedEntropyModel(self.prior, coding_rank=1, compression=False)
        y_tilde, rate = entropy_model(y, training=training)
        x_tilde = self.synthesis_transform(y_tilde)
        rate = tf.reduce_mean(rate)
        distortion = tf.reduce_mean(abs(x - x_tilde))
        return dict(rate=rate, distortion=distortion)

# Define Training Function

def pass_through_loss(_, x):
    return x

def train_model(lmbda, latent_dims=10, epochs=5):
    trainer = MNISTCompressionTrainer(latent_dims)
    trainer.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=dict(rate=pass_through_loss, distortion=pass_through_loss),
        metrics=dict(rate=pass_through_loss, distortion=pass_through_loss),
        loss_weights=dict(rate=1., distortion=lmbda)
    )
    trainer.fit(training_dataset, validation_data=validation_dataset, epochs=epochs)
    return trainer

# Train the model
trainer = train_model(lmbda=100, latent_dims=10, epochs=5)

# Generate Samples
compressor, decompressor = trainer, trainer
strings = tf.constant([os.urandom(8) for _ in range(16)])
samples = decompressor.synthesis_transform(strings)

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(tf.squeeze(samples[i]), cmap="gray")
    ax.axis("off")
plt.show()
