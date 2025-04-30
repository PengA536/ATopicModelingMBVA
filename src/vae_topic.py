import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import KLDivergence
from transformers import BertTokenizer, TFBertModel
import phate


class VAETopic:
	def __init__(self, input_dim, latent_dim=50, hidden_dim=128):
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.hidden_dim = hidden_dim
		self.model = self._build_vae()
		self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')

	def _build_vae(self):
		"""Build VAE model with encoder and decoder."""
		# Encoder
		inputs = layers.Input(shape=(self.input_dim,))
		h = layers.Dense(self.hidden_dim, activation='relu')(inputs)
		h = layers.Dense(self.hidden_dim, activation='relu')(h)
		z_mean = layers.Dense(self.latent_dim)(h)
		z_log_var = layers.Dense(self.latent_dim)(h)

		# Sampling layer
		def sampling(args):
			z_mean, z_log_var = args
			epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_dim))
			return z_mean + tf.exp(0.5 * z_log_var) * epsilon

		z = layers.Lambda(sampling)([z_mean, z_log_var])

		# Decoder
		decoder_h = layers.Dense(self.hidden_dim, activation='relu')(z)
		outputs = layers.Dense(self.input_dim, activation='sigmoid')(decoder_h)

		# VAE model
		vae = Model(inputs, outputs)

		# Loss: Reconstruction + KL Divergence
		reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, outputs)) * self.input_dim
		kl_loss = KLDivergence()(tf.random.normal(shape=(self.input_dim, self.latent_dim)),
			tf.random.normal(shape=(self.input_dim, self.latent_dim)))
		vae_loss = reconstruction_loss + kl_loss
		vae.add_loss(vae_loss)
		vae.compile(optimizer='adam', learning_rate=0.001)
		return vae

	def get_bert_embeddings(self, texts):
		"""Generate BERT embeddings for documents."""
		embeddings = []
		for text in texts:
			inputs = self.bert_tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=512)
			outputs = self.bert_model(**inputs)
			embedding = outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token
			embeddings.append(embedding)
		return np.vstack(embeddings)

	def reduce_dimensionality(self, embeddings):
		"""Apply PHATE for dimensionality reduction."""
		phate_op = phate.PHATE(n_components=2, verbose=False)
		reduced = phate_op.fit_transform(embeddings)
		return reduced

	def compute_topic_word_distribution(self, X, latent_vectors, vectorizer):
		"""Compute topic-word distributions."""
		vocab = vectorizer.get_feature_names_out()
		topic_word_dist = []
		for c in range(np.unique(latent_vectors).shape[0]):
			class_indices = np.where(latent_vectors == c)[0]
			S_c = np.zeros(len(vocab))
			for idx in class_indices:
				for w_idx, freq in zip(X[idx].indices, X[idx].data):
					S_c[w_idx] += freq * latent_vectors[idx, w_idx % latent_vectors.shape[1]]
			theta_c = np.exp(S_c) / np.sum(np.exp(S_c))
			topic_word_dist.append(theta_c)
		return np.array(topic_word_dist), vocab