from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
import gensim
import os
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')


class WordEmbedding:
    def __init__(self):
        self.length_longest_sentence = 1
        self.tokenized_sentences = []
        self.embedding_index = {}
        self.word_index = {}
        self.num_words = 0

    def token_sentences(self, clean_dataset):
        """
        :param clean_dataset:
        :return: senetence tokenizing
        """

        for sentence in clean_dataset.loc[:, 'text']:
            tokenized_sentence = word_tokenize(sentence)
            if len(tokenized_sentence) > self.length_longest_sentence:
                self.length_longest_sentence = len(tokenized_sentence)
            self.tokenized_sentences.append(tokenized_sentence)

        return self

    def word_2_vector(self):
        """
        :return: word to vector embedding
        """
        model = gensim.models.Word2Vec(sentences=self.tokenized_sentences,
                                       size=100, window=5, min_count=1)
        words = list(model.wv.vocab)
        print('vocabulary size {}'.format(len(words)))

        filename = 'embedding_word2vec.txt'
        model.wv.save_word2vec_format(filename, binary=False)

    def read_embedding_word2vec_file(self):
        """
        :return: reading word to vector embedding from file
        """
        self.embedding_index = {}
        with open(os.path.join('', 'embedding_word2vec.txt'), 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:])
                self.embedding_index[word] = vector

    def pad_sentences(self, clean_dataset):

        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(self.tokenized_sentences)
        sequences = tokenizer_obj.texts_to_sequences(self.tokenized_sentences)

        # pad sequences
        self.word_index = tokenizer_obj.word_index
        print('found unique tokens {}'.format(len(self.word_index)))

        sentences_padding = pad_sequences(sequences, maxlen=self.length_longest_sentence)
        score_groups = clean_dataset['y'].values

        return sentences_padding, score_groups

    def embedding_matrix(self):
        """
        :return: embedding matrix calculation - vector of 100 dim for each word
        """
        self.num_words = len(self.word_index) + 1
        embedding_matrix = np.zeros((self.num_words, 100))

        for word, i in self.word_index.items():
            if i > self.num_words:
                continue
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    @staticmethod
    def count_vectorizer(clean_dataset):
        """
        :param clean_dataset: clean text set
        :return: count of each word in a sentence
        """

        # To extract max 100 feature
        cv = CountVectorizer(max_features=100)

        # X contains corpus
        x_word_counter = cv.fit_transform(clean_dataset.loc[:, 'text']).toarray()

        return x_word_counter





