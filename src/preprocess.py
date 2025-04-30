import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_text(text):
	"""Preprocess text: tokenize, remove stopwords, stem, and lemmatize."""
	# Lowercase
	text = text.lower()
	# Remove punctuation
	text = re.sub(r'[^\w\s]', '', text)
	# Tokenize
	tokens = word_tokenize(text)
	# Remove stopwords
	stop_words = set(stopwords.words('english'))
	tokens = [t for t in tokens if t not in stop_words]
	# Stemming and Lemmatization
	stemmer = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	tokens = [stemmer.stem(lemmatizer.lemmatize(t)) for t in tokens]
	return ' '.join(tokens)


def load_and_preprocess_data(data_path, dataset_name):
	"""Load and preprocess dataset."""
	if dataset_name == 'patent':
		# Placeholder: Load patent data from CSV or text files
		# Expected format: {'text': document, 'label': IPC_code}
		data = pd.read_csv(data_path)  # Update with actual file path
		data['processed_text'] = data['text'].apply(preprocess_text)
	elif dataset_name == '20_newsgroups':
		from sklearn.datasets import fetch_20newsgroups
		newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
		data = pd.DataFrame({'text': newsgroups.data, 'label': newsgroups.target})
		data['processed_text'] = data['text'].apply(preprocess_text)
	elif dataset_name == 'ag_news':
		# AG News CSV format: {'class': label, 'title': text, 'description': text}
		data = pd.read_csv(data_path)
		data['text'] = data['title'] + ' ' + data['description']
		data['processed_text'] = data['text'].apply(preprocess_text)

	# Remove short documents
	data = data[data['processed_text'].str.split().str.len() >= 5]
	# Deduplicate
	data = data.drop_duplicates(subset='processed_text')
	return data


def vectorize_data(texts):
	"""Convert texts to bag-of-words representation."""
	vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
	X = vectorizer.fit_transform(texts)
	return X, vectorizer


if __name__ == "__main__":
	dataset_name = '20_newsgroups'  # Example dataset
	data_path = '../data/20_newsgroups/'  # Update for patent or AG News
	data = load_and_preprocess_data(data_path, dataset_name)
	X, vectorizer = vectorize_data(data['processed_text'])
	print(f"Processed {dataset_name} with {X.shape[0]} documents and {X.shape[1]} features.")