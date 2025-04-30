import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


def plot_tsne(embeddings, labels, dataset_name):
	"""Plot t-SNE visualization of document clusters."""
	plt.figure(figsize=(10, 8))
	sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=labels, palette='viridis')
	plt.title(f't-SNE Visualization of {dataset_name} Documents')
	plt.savefig(f'../results/tsne_{dataset_name}.png')
	plt.close()


def plot_wordcloud(topic_word_dist, vocab, topic_id, dataset_name):
	"""Plot word cloud for a topic."""
	word_freq = dict(zip(vocab, topic_word_dist[topic_id]))
	wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
	plt.figure(figsize=(10, 5))
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis('off')
	plt.title(f'Topic {topic_id} Word Cloud ({dataset_name})')
	plt.savefig(f'../results/wordcloud_{dataset_name}_topic{topic_id}.png')
	plt.close()


if __name__ == "__main__":
	# Placeholder: Load data
	embeddings = np.random.rand(1000, 2)
	labels = np.random.randint(0, 5, 1000)
	topic_word_dist = np.random.rand(5, 1000)
	vocab = [f'word{i}' for i in range(1000)]
	dataset_name = 'patent'

	plot_tsne(embeddings, labels, dataset_name)
	for i in range(5):
		plot_wordcloud(topic_word_dist, vocab, i, dataset_name)