import numpy as np
from vae_topic import VAETopic
from preprocess import load_and_preprocess_data, vectorize_data
from sklearn.cluster import KMeans
from bayes_opt import BayesianOptimization
import os


def train_vae_topic(data_path, dataset_name, num_topics=5):
	"""Train VAE-TOPIC model."""
	# Load and preprocess data
	data = load_and_preprocess_data(data_path, dataset_name)
	X, vectorizer = vectorize_data(data['processed_text'])

	# Initialize VAE-TOPIC
	vae_topic = VAETopic(input_dim=X.shape[1], latent_dim=50)

	# Get BERT embeddings
	embeddings = vae_topic.get_bert_embeddings(data['processed_text'].values)

	# Dimensionality reduction with PHATE
	reduced_embeddings = vae_topic.reduce_dimensionality(embeddings)

	# Cluster documents
	kmeans = KMeans(n_clusters=num_topics, random_state=42)
	cluster_labels = kmeans.fit_predict(reduced_embeddings)

	# Train VAE
	X_dense = X.toarray()
	vae_topic.model.fit(X_dense, epochs=50, batch_size=128, verbose=1)

	# Get latent representations
	encoder = vae_topic.model.get_layer('lambda').input[0]  # z_mean
	latent_vectors = encoder.predict(X_dense)

	# Compute topic-word distributions
	topic_word_dist, vocab = vae_topic.compute_topic_word_distribution(X, latent_vectors, vectorizer)

	return topic_word_dist, vocab, cluster_labels, reduced_embeddings


def optimize_hyperparameters(data_path, dataset_name):
	"""Optimize VAE-TOPIC hyperparameters using Bayesian optimization."""

	def objective(latent_dim, learning_rate, kl_weight):
		latent_dim = int(latent_dim)
		vae_topic = VAETopic(input_dim=1000, latent_dim=latent_dim)  # Simplified input_dim
		vae_topic.model.compile(optimizer='adam', learning_rate=learning_rate)
		# Placeholder: Compute coherence score on validation set
		coherence = np.random.random()  # Replace with actual coherence computation
		return coherence

	pbounds = {'latent_dim': (10, 100), 'learning_rate': (0.0001, 0.01), 'kl_weight': (0.1, 2.0)}
	optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
	optimizer.maximize(init_points=5, n_iter=10)
	return optimizer.max['params']


if __name__ == "__main__":
	dataset_name = '20_newsgroups'
	data_path = '../data/20_newsgroups/'
	num_topics = 21 if dataset_name == '20_newsgroups' else 4 if dataset_name == 'ag_news' else 5
	topic_word_dist, vocab, cluster_labels, reduced_embeddings = train_vae_topic(data_path, dataset_name, num_topics)
	print("Topic-Word Distributions:", topic_word_dist.shape)