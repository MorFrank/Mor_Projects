from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.linear_model import LinearRegression
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class EDA:

    @staticmethod
    def group_scores(df_scores_levels: pd.DataFrame) -> np.ndarray:
        """

        :param df_scores_levels:
        :return: scores grouped by levels
        """

        my_array = np.array(df_scores_levels)
        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=800, n_init=10, random_state=111)
        y_kmeans = kmeans.fit_predict(my_array)

        plt.figure()

        plt.scatter(my_array[y_kmeans == 0, 0], my_array[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
        plt.scatter(my_array[y_kmeans == 1, 0], my_array[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
        # plt.scatter(my_array[y_kmeans==2, 0], my_array[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
        plt.xlabel('scores', fontsize=14)
        plt.ylabel('levels', fontsize=14)
        plt.legend(['0- good', '1- very good', '2- excellent', '3', '4'], fontsize=13)
        plt.savefig('kmeans.png')

        df_scores_levels['y'] = y_kmeans
        df_scores_levels.head()
        y_score_groups = np.array(df_scores_levels['y']).reshape(176, 1)

        return y_score_groups

    @staticmethod
    def score_level(scores, levels):
        """

        :param scores: score for each video
        :param levels: level for each video
        :return: visualization of scores by levels boxplots
        """

        df_scores_levels = pd.DataFrame([scores.ravel().tolist(), levels.ravel().tolist()],
                                        index=['scores', 'levels']).transpose()
        x_levels = df_scores_levels[['levels']]
        y_scores = df_scores_levels['scores']

        model = LinearRegression().fit(x_levels, y_scores)
        y_scores_pred = model.predict(x_levels)

        boxplot = df_scores_levels.boxplot(by='levels', figsize=(8, 5), fontsize=14)
        plt.plot(x_levels, y_scores_pred)
        plt.tight_layout()

        plt.xlabel('levels', fontsize=14)

        plt.legend(['scores-levels data', 'Linear regression'], loc='best')
        plt.savefig('boxplot_score_levels.png')
        plt.close()

        return EDA.group_scores(df_scores_levels)

    @staticmethod
    def display_images(x, scores, levels):
        fig = plt.figure(figsize=(10, 7))
        for i in range(0, 10):
            fig.add_subplot(2, 5, i + 1)
            plt.axis('off')
            num = np.random.randint(low=0, high=175, size=1)
            plt.imshow(x[num][0][50])
            plt.title(f'video : {num}\n score: {scores[num]}\n level: {levels[num]}', fontsize=14)
        plt.tight_layout()
        plt.savefig('sample_of_images.png')
        plt.close()

    @staticmethod
    def get_standardized_train_test_sets(x, y):
        """
        :param x: images
        :param y: score groups
        :return: standardized train-test sets
        """

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=121)

        # standardized X_train and X_test
        x_train_mean = np.mean(x_train)
        x_train_std_deviation = np.std(x_train)
        x_scaled_train = (x_train - x_train_mean) / x_train_std_deviation
        x_scaled_test = (x_test - x_train_mean) / x_train_std_deviation

        # clip pixel values to [-1,1]
        x_scaled_train = np.clip(x_scaled_train, -1.0, 1.0)
        x_scaled_test = np.clip(x_scaled_test, -1.0, 1.0)

        # shift from [-1,1] to [0,1] with 0.5 mean
        x_scaled_train = (x_scaled_train + 1.0) / 2.0
        x_scaled_test = (x_scaled_test + 1.0) / 2.0

        num_classes = 2
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return x_scaled_train, x_scaled_test, y_train, y_test

    @staticmethod
    def read_text(scores, scores_text):
        """
        :param scores: score for each video
        :param scores_text: text for each video
        :return: DataFrame with text and scores
        """
        dataset = pd.DataFrame(list(zip(scores_text, scores)), columns=['text', 'score'])
        dataset.head()

        return dataset

    @staticmethod
    def nlp_pre_processing(dataset, scores, y_score_groups):
        """
        :param y_score_groups:
        :param dataset: scores and text
        :param scores:
        :return: DataFrame with clean text and scores
        """

        # Initialize empty list
        # to append clean text
        clean_dataset = []

        # 176 text rows to clean
        for i in range(0, 176):
            # column : "test", row ith
            # clean_text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
            clean_text = dataset['text'][i]

            # convert text to lower cases
            clean_text = clean_text.lower()

            # split to list of words (default delimiter is " ")
            clean_text = clean_text.split()

            # rejoin all string words elements
            # to create back a string of a sentence
            clean_text = ' '.join(clean_text)

            # append each string of sentence to create
            # list of clean text
            clean_dataset.append(clean_text)

        clean_dataset = pd.DataFrame(list(zip(clean_dataset, scores)), columns=['text', 'score'])

        clean_dataset['y'] = y_score_groups
        clean_dataset.drop(['score'], axis=1, inplace=True)
        clean_dataset.head()

        return clean_dataset






