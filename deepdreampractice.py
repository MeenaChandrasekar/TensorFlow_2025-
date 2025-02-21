import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# ✅ Step 1: Load TensorFlow and Check Version
print("✅ TensorFlow version:", tf.__version__)

# ✅ Step 2: Load an Image
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
image_path = tf.keras.utils.get_file("YellowLabrador.jpg", origin=url)
img = PIL.Image.open(image_path)

# ✅ Step 3: Display the Original Image
plt.imshow(img)
plt.axis('off')
plt.show()

# ✅ Step 4: Preprocess Image for DeepDream
def preprocess_image(image_path):
    img = PIL.Image.open(image_path)
    img = np.array(img).astype(np.float32) / 255.0
    return img

img = preprocess_image(image_path)

# ✅ Step 5: Load Pre-trained Model (InceptionV3)
base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

# ✅ Step 6: Select Layers for DeepDream Effect
dream_layers = ['mixed3', 'mixed5']
dream_model = tf.keras.Model(inputs=base_model.input,
                             outputs=[base_model.get_layer(name).output for name in dream_layers])

# ✅ Step 7: Define DeepDream Function
def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    losses = [tf.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

@tf.function
def deepdream_step(img, model, step_size=0.01):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = calc_loss(img, model)
    gradients = tape.gradient(loss, img)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    img = img + gradients * step_size
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img

# ✅ Step 8: Apply DeepDream Effect
def run_deep_dream(img, steps=50, step_size=0.01):
    img = tf.convert_to_tensor(img)
    for step in range(steps):
        img = deepdream_step(img, dream_model, step_size)
    return img.numpy()

dream_img = run_deep_dream(img)

# ✅ Step 9: Show Final Dreamy Image
plt.imshow(dream_img)
plt.axis('off')
plt.show()

# ✅ Step 10: Save the Output Image
final_image = PIL.Image.fromarray((dream_img * 255).astype(np.uint8))
final_image.save("deepdream_result.jpg")
print("✅ DeepDream image saved as deepdream_result.jpg")

file_name = "deepdream_image result.png"
tensor_to_image(img).save(file_name)

