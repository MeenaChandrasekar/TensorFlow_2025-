import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import numpy as np
import os

# 🚀 Step 1: Load MovieLens Dataset
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

# 🚀 Step 2: Prepare Dataset (Select User & Movie IDs)
ratings = ratings.map(lambda x: {
    "user_id": tf.strings.as_string(x["user_id"]),
    "movie_id": tf.strings.as_string(x["movie_id"])
})
movies = movies.map(lambda x: tf.strings.as_string(x["movie_id"]))

# 🚀 Step 3: Create Vocabulary (Unique Users & Movies)
user_ids = np.unique(np.concatenate([list(ratings.map(lambda x: x["user_id"]).as_numpy_iterator())]))
movie_ids = np.unique(np.concatenate([list(movies.as_numpy_iterator())]))

# 🚀 Step 4: Convert User & Movie IDs to TensorFlow Lookup Tables
user_id_lookup = tf.keras.layers.StringLookup(vocabulary=user_ids, mask_token=None)
movie_id_lookup = tf.keras.layers.StringLookup(vocabulary=movie_ids, mask_token=None)

# 🚀 Step 5: Build User & Movie Embedding Models
embedding_dim = 64

user_model = tf.keras.Sequential([
    user_id_lookup,
    tf.keras.layers.Embedding(len(user_ids) + 1, embedding_dim)
])

movie_model = tf.keras.Sequential([
    movie_id_lookup,
    tf.keras.layers.Embedding(len(movie_ids) + 1, embedding_dim)
])

# 🚀 Step 6: Define Recommendation Model using TFRS
class MovieRecommender(tfrs.Model):
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.user_model = user_model
        self.movie_model = movie_model
        self.task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(movie_model)
        ))

    def compute_loss(self, features, training=False):
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_id"])
        return self.task(user_embeddings, movie_embeddings)

# 🚀 Step 7: Compile & Train the Model
model = MovieRecommender(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Convert dataset to TensorFlow Dataset format
ratings_dataset = ratings.batch(4096).shuffle(100000).prefetch(tf.data.AUTOTUNE)

print("\n🔥 Training the Model...")
model.fit(ratings_dataset, epochs=5)

# 🚀 Step 8: Make Predictions (Recommend Movies for a User)
print("\n🎬 Getting Movie Recommendations...")

# Pick a sample user ID
sample_user_id = "42"

# Get user embedding
sample_user_embedding = user_model(tf.constant([sample_user_id]))

# Get scores for all movies
scores, recommended_movie_ids = model.task.factorized_top_k(sample_user_embedding)

# Convert movie IDs back to original format
recommended_movie_ids = movie_id_lookup.get_vocabulary()[recommended_movie_ids.numpy()[0][:5]]

# Print recommended movies
print(f"\n🎥 Recommended Movies for User {sample_user_id}: {recommended_movie_ids}")

# 🚀 Model Training & Recommendations Done!
