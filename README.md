# VAE-TOPIC: A Topic Modeling Method based on Variational Autoencoder

This repository contains the implementation of VAE-TOPIC, a novel topic modeling approach that integrates Variational Autoencoders (VAE) with Bayesian inference, as described in the paper *"A Topic Modeling Method based on Variational Autoencoder"* by Yanhua Wang and Chunfang Min.

## Overview
VAE-TOPIC combines deep learning and Bayesian inference to discover latent topics in document collections. It leverages BERT for document embedding, PHATE for dimensionality reduction, and a VAE architecture to learn low-dimensional representations, followed by Bayesian inference to infer topic-word distributions. The model outperforms traditional methods like LDA on sparse and long-tailed data.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/VAE-TOPIC.git
   cd VAE-TOPIC
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Download NLTK data:
   ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
## Datasets
The model is evaluated on three publicly available datasets:
1. Patent Data:
   * Source: China National Intellectual Property Administration (CNIPA)
   * Access: [https://english.cnipa.gov.cn](https://english.cnipa.gov.cn)
   * Instructions: Search for patents with IPC codes A61, C04, D03, G07, and H03, within the time frame [2022–2023]. Export as CSV or text files and place in data/patent_data/
2. 20 Newsgroups:
   * Source: [http://qwone.com/~jason/20Newsgroups/](http://qwone.com/~jason/20Newsgroups/)
   * Access: Automatically downloaded via sklearn.datasets.fetch_20newsgroups
3. AG News:
   * Source: [https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv)
   * Access: Download the CSV files and place them in data/ag_news/
## Usage
1. Preprocess Data:
   ```bash
   python src/preprocess.py
Update the dataset name and path in preprocess.py as needed.
2. Train VAE-TOPIC:
   ```bash
   python src/train.py
Specify the dataset name and number of topics in train.py.
3. Evaluate Model:
   ```bash
   python src/evaluate.py
Provide true and predicted labels for evaluation.
4. Visualize Results:
   ```bash
   python src/visualize.py
Generates t-SNE plots and word clouds saved in the results/ directory.
## Code Structure
* data/: Directory for storing datasets.
* src/preprocess.py: Data preprocessing (tokenization, stop word removal, vectorization).
* src/vae_topic.py: VAE-TOPIC model implementation (BERT embedding, PHATE, VAE).
* src/train.py: Training script with Bayesian optimization for hyperparameters.
* src/evaluate.py: Evaluation script for Precision, Recall, and F-measure.
* src/visualize.py: Visualization script for t-SNE and word clouds.
* requirements.txt: List of dependencies.
* LICENSE: MIT License.



