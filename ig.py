import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

# Load ImageNet Labels
def load_imagenet_labels(file_path):
    labels_file = tf.keras.utils.get_file('ImageNetLabels.txt', file_path)
    with open(labels_file) as reader:
        labels = reader.read().splitlines()
    return np.array(labels)

imagenet_labels = load_imagenet_labels('https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')

# Load Pretrained Model (Fixed)
inception_v1 = hub.KerasLayer(
    handle="https://tfhub.dev/google/imagenet/inception_v1/classification/4",
    trainable=False
)

# Define Input Shape
input_shape = (224, 224, 3)
inputs = tf.keras.Input(shape=input_shape)

# ✅ Fix: Explicitly call the layer instead of direct usage
outputs = inception_v1.call(inputs)

# Create Functional Model
model = tf.keras.Model(inputs, outputs)
print("✅ Model Loaded Successfully")

# Load & Preprocess Image
def read_image(file_name):
    image = tf.io.read_file(file_name)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image

# Download Example Images
img_url = {
    'Fireboat': 'http://storage.googleapis.com/download.tensorflow.org/example_images/San_Francisco_fireboat_showing_off.jpg',
    'Giant Panda': 'http://storage.googleapis.com/download.tensorflow.org/example_images/Giant_Panda_2.jpeg',
}

img_paths = {name: tf.keras.utils.get_file(name, url) for (name, url) in img_url.items()}
img_name_tensors = {name: read_image(img_path) for (name, img_path) in img_paths.items()}

# Display Images
plt.figure(figsize=(8, 8))
for n, (name, img_tensors) in enumerate(img_name_tensors.items()):
    ax = plt.subplot(1, 2, n+1)
    ax.imshow(img_tensors)
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.show()

# Function to Get Top-K Predictions
def top_k_predictions(img, k=3):
    image_batch = tf.expand_dims(img, 0)
    predictions = model(image_batch)
    probs = tf.nn.softmax(predictions, axis=-1)
    top_probs, top_idxs = tf.math.top_k(input=probs, k=k)
    top_labels = imagenet_labels[tuple(top_idxs)]
    return top_labels, top_probs[0]

# Display Predictions
for (name, img_tensor) in img_name_tensors.items():
    plt.imshow(img_tensor)
    plt.title(name, fontweight='bold')
    plt.axis('off')
    plt.show()

    pred_label, pred_prob = top_k_predictions(img_tensor)
    for label, prob in zip(pred_label, pred_prob):
        print(f'{label}: {prob:0.1%}')

# Baseline (Black Image)
baseline = tf.zeros(shape=(224, 224, 3))

# Interpolation Function
def interpolate_images(baseline, image, alphas):
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline_x = tf.expand_dims(baseline, axis=0)
    input_x = tf.expand_dims(image, axis=0)
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta
    return images

# Generate Alpha Values
m_steps = 50
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

# Generate Interpolated Images
interpolated_images = interpolate_images(baseline=baseline, image=img_name_tensors['Fireboat'], alphas=alphas)

# Compute Gradients
def compute_gradients(images, target_class_idx):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
    return tape.gradient(probs, images)

# Compute Gradients Along Path
target_class_idx = 555  # Fireboat class
path_gradients = compute_gradients(images=interpolated_images, target_class_idx=target_class_idx)

# Integral Approximation (Riemann Sum)
def integral_approximation(gradients):
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

# Compute Integrated Gradients
ig_attributions = integral_approximation(gradients=path_gradients)

# Scale IG Attributions
ig_attributions = (img_name_tensors['Fireboat'] - baseline) * ig_attributions

# Function to Plot IG Attributions
def plot_img_attributions(image, baseline, target_class_idx, m_steps=50, cmap=plt.cm.inferno, overlay_alpha=0.4):
    ig_attributions = integral_approximation(compute_gradients(interpolate_images(baseline, image, alphas), target_class_idx))
    attribution_mask = np.abs(ig_attributions).numpy().sum(axis=-1)
    plt.imshow(image)
    plt.imshow(attribution_mask, cmap=cmap, alpha=overlay_alpha)
    plt.axis('off')
    plt.show()

# Final Visualization
plot_img_attributions(image=img_name_tensors['Fireboat'], baseline=baseline, target_class_idx=target_class_idx, m_steps=240)

print("✅ Integrated Gradients Computation Completed Successfully!")
