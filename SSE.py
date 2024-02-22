
from gensim.models import Word2Vec
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import enchant
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from math import log


class SpreadIndexCalculator:
    def __init__(self, models):
        self.models = models
    
    def spread_index_per_year(self, target_word):
        """
        Constructor for the SpreadIndexCalculator class.

        :param models: A dictionary where the key is the year and the value is the word embedding model for that year.
        """
        spread_indices = {}
        for year, model in self.models.items():
            if target_word in model.wv.key_to_index: 
                similar_words = model.wv.most_similar(target_word, topn=15)
                word_list = [word for word, _ in similar_words]
                word_vectors = [model.wv[word] for word in word_list if word in model.wv.key_to_index]
                
                if word_vectors: 
                    mean_vector = np.mean(word_vectors, axis=0)
                    spread_index = np.std([metrics.pairwise.euclidean_distances([vector], [mean_vector])[0][0] for vector in word_vectors])
                    spread_indices[year] = spread_index
        return spread_indices
    
    def spread_index_in_time(self, target_word):
        """
        Calculates the spread index for a target word for each year/model.

        :param target_word: The word for which to calculate the spread index.
        :return: A dictionary with years as keys and spread indices as values.
        """
        target_word_vectors = []
        for year, model in self.models.items():
            if target_word in model.wv.key_to_index:
                target_word_vectors.append(model.wv[target_word])
        
        if not target_word_vectors:
            return None  
        
        mean_vector = np.mean(target_word_vectors, axis=0)
        spread_index = np.std([metrics.pairwise.euclidean_distances([vector], [mean_vector])[0][0] for vector in target_word_vectors])
        
        return spread_index

def tfidf(tokens):
    frequency_per_year = {}
    for year, lists_of_tokens in tokens.items():
        document = [token for sublist in lists_of_tokens for token in sublist] 
        frequency_per_year[year] = {}
        document_len = len(document)
        for token in document:
            token = token.lower()
            frequency_per_year[year][token] = frequency_per_year[year].get(token, 0) + 1 / document_len # questa da capire
    
    # Compute doc frequency, as the frequency over the whole set
    doc_freq = {}
    for year in frequency_per_year:
        for token in frequency_per_year[year]:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    
    total_docs = len(frequency_per_year)
    
    # Compute IDF for each year
    idf = {}
    for token, df in doc_freq.items():
        idf[token] = log(total_docs / df)
    
    tfidf_per_year = {year: {} for year in frequency_per_year}
    for year in frequency_per_year:
        for token, tf in frequency_per_year[year].items():
            tfidf_per_year[year][token] = tf * idf[token]
    
    return tfidf_per_year


class WordFrequencyCalculator:
    def __init__(self, tokens):
        """
        Initializes the WordFrequencyCalculator with a given NSC (Narrow, Slow, Close) vocabulary.

        :param nsc_vocab: A dictionary with categories ('Narrow', 'Slow', 'Close') as keys and lists of words as values.
        """
        self.tokens = tokens

    def calculate_word_frequencies_cat(self, nsc_vocab):
        """
        Calculates the total word frequencies per category for each year.

        :param tokens: A dictionary where the key is the year and the value is a list of lists of tokens.
        :return: A dictionary with years as keys, and values as dictionaries of categories with total word count.
        """
        frequencies = {year: {'Narrow': 0, 'Slow': 0, 'Close': 0} for year in self.tokens}

        for year, lists_of_tokens in self.tokens.items():
            all_tokens_of_year = [token for sublist in lists_of_tokens for token in sublist]
            for category, words in nsc_vocab.items():
                frequencies[year][category] += sum(all_tokens_of_year.count(word) for word in words)

        return frequencies


    def calculate_detailed_word_frequencies(self, nsc_vocab):
        """
        Calculates detailed word frequencies for each category, providing counts for individual words per category for each year.

        :param tokens: A dictionary where the key is the year and the value is a list of lists of tokens.
        :return: A dictionary with years as keys, and for each year, dictionaries of categories with individual word counts.
        """
        frequencies = {year: {category: {word: 0 for word in words} for category, words in nsc_vocab.items()} for year in self.tokens}

        for year, lists_of_tokens in self.tokens.items():
            all_tokens_of_year = [token for sublist in lists_of_tokens for token in sublist]
            for category, words in nsc_vocab.items():
                for word in words:
                    frequencies[year][category][word] = all_tokens_of_year.count(word)

        return frequencies

    
    
def find_most_similar_words_per_year(target_word, tfidf_per_year, yearly_embeddings, words_to_include=None, words_to_exclude=None, top_n=10):
    similar_words_per_year = {}

    for year, tfidf_dict in tqdm(tfidf_per_year.items()):
        model = yearly_embeddings[year]
        # Check if the target word is in the model's vocabulary
        if target_word in model.key_to_index:
            target_embedding = model[target_word]

            word_embeddings = {}
            for word in tfidf_dict:
                if (word in model.key_to_index 
                    and word != target_word 
                    and (words_to_include is None or word in words_to_include) 
                    and (not words_to_exclude or word not in words_to_exclude)):
                    word_embeddings[word] = model[word]

            # Select words based on TF-IDF > 0 and that have embeddings
            filtered_words = {word: tfidf for word, tfidf in tfidf_dict.items() if word in word_embeddings and tfidf > 0}

            # Calculate similarities between target word and other words
            if word_embeddings:
                similarities = {}
                for word, embedding in word_embeddings.items():
                    similarities[word] = cosine_similarity([target_embedding], [embedding])[0][0]

                # Sort words by similarity and take the top n
                top_similar_words = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
                similar_words_per_year[year] = [(word, similarities[word]) for word in top_similar_words]

    return similar_words_per_year