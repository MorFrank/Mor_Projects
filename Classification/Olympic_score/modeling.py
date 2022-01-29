import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LSTM, \
    TimeDistributed, Embedding
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam, Adadelta
from keras.utils import plot_model
from word_embedding import WordEmbedding
from tensorflow.keras.layers import Embedding
from keras.initializers import Constant
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import random
my_seed = 512
np.random.seed(my_seed)
random.seed(my_seed)


class Models:

    def __init__(self, embedding_matrix,
                 length_longest_sentence: WordEmbedding,
                 num_words: WordEmbedding):
        self.model1 = Sequential()
        model = Sequential()
        self.num_classes = 2
        self.embedding_matrix = embedding_matrix
        self.length_longest_sentence = length_longest_sentence
        self.num_words = num_words

    def image_modeling_architecture(self, x_train, y_train1, x_test, y_test, fold, img_height=128, img_width=128):
        """
        :return: neural network model architecture CNN + LSTM
        """
        # define CNN model
        model = Sequential()
        model.add((TimeDistributed(Conv2D(filters=9,
                                               kernel_size=(2, 2),
                                               padding='same',
                                               activation='relu',
                                               input_shape=(None, img_height, img_width, 3)))))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(Flatten()))

        # define LSTM model

        model.add(LSTM(60, return_sequences=False))
        model.add(Dense(units=self.num_classes,
                             activation='softmax'))

        model.compile(loss=categorical_crossentropy,
                           optimizer=Adadelta(),
                           metrics=['accuracy'])
        y_train1 = to_categorical(y_train1, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

        #my_history = model.fit(tf.convert_to_tensor(x_train, dtype=tf.float32),
                               #y_train1, batch_size=16,
                               #epochs=200, verbose=2)

        model = load_model('model1', custom_objects=None, compile=True)
        scores_test = model.evaluate(x_test, y_test, verbose=0)
        scores_train = model.evaluate(x_train, y_train1, verbose=0)
        test_predict = model.predict(x_test, verbose=0)
        train_predict = model.predict(x_train, verbose=0)


        # model summary
        input_shape = (None, 100, img_height, img_width, 3)
        #self.model.build(input_shape)
        #self.model.summary()
        # plot_model(model, show_layer_names=True, show_shapes=True)

        save_model(model,f'model{fold}')
        return my_history, scores_train, scores_test, train_predict, test_predict

    def run_image_model(self, x_scaled_train, y_train, x_scaled_test, y_test):
        fold_no = 1
        acc_per_fold_test = []
        loss_per_fold_test = []
        acc_per_fold_train = []
        loss_per_fold_train = []
        kf = KFold(n_splits=3)
        for train_index, test_index in kf.split(x_scaled_train, y_train):
            # my_history = self.model.fit(x_scaled_train[train], y_train[train], batch_size=16,
            # epochs=30, verbose=2, validation_split=0.2)
            x_train, x_test = x_scaled_train[train_index], x_scaled_train[test_index]
            y_train1, y_test = y_train[train_index], y_train[test_index]
            num_classes = 2
            #y_train1 = to_categorical(y_train1, num_classes)
            #y_test = to_categorical(y_test, num_classes)

            my_history, scores_train, scores_test, train_predict, test_predict = Models.image_modeling_architecture(self, x_train, y_train1, x_test,
                                                                                                                                  y_test, fold_no)

            n_epochs = len(my_history.history['loss'])

            #with open(f"image_model{fold_no}", 'wb') as fw:
                #pickle.dump(self.model, fw)

            with open(f"accuracy_history{fold_no}", 'wb') as fw:
                pickle.dump(my_history.history['accuracy'], fw)
            with open(f"loss_history{fold_no}", 'wb') as fw:
                pickle.dump(my_history.history['loss'], fw)

            with open(f"train_predict{fold_no}", 'wb') as fw:
                pickle.dump(train_predict, fw)
            with open(f"test_predict{fold_no}", 'wb') as fw:
                pickle.dump(f"test_predict{fold_no}", fw)

            plt.close()
            plt.plot([i for i in range(1, n_epochs+1)], my_history.history['accuracy'], "--ob")
            plt.plot([int(i) for i in range(1, n_epochs+1)], [0.5 for i in range(1, n_epochs+1)], "--g")
            plt.xlabel('epochs', fontsize=14)
            plt.ylabel('accuracy', fontsize=14)
            plt.legend(['train set'], loc='best')
            plt.tight_layout()
            plt.savefig(f'save_epochs_eccuracy{fold_no}.png')

            plt.close()

            plt.plot([i for i in range(1, n_epochs + 1)], my_history.history['loss'], "--ob")
            plt.xlabel('epochs', fontsize=14)
            plt.ylabel('loss', fontsize=14)
            plt.legend(['train set'], loc='best')
            plt.tight_layout()
            plt.savefig(f'save_epochs_loss{fold_no}.png')
            plt.close()

            #scores_test = self.model.evaluate(x_test, y_test, verbose=0)
            #scores_train = self.model.evaluate(x_train, y_train1, verbose=0)

            #print(
                #f'Score for fold {fold_no}: {self.model.metrics_names[0]} of {scores_test[0]}; {self.model.metrics_names[1]} of {scores_test[1] * 100}%')
            acc_per_fold_test.append(scores_test[1])
            loss_per_fold_test.append(scores_test[0])
            acc_per_fold_train.append(scores_train[1])
            loss_per_fold_train.append(scores_train[0])

            # Increase fold number
            fold_no = fold_no + 1

            # (pd.DataFrame(my_history.history)).plot()
            # plt.plot(my_history.history['accuracy'])

        with open("acc_per_fold_test", 'wb') as fw:
            pickle.dump(acc_per_fold_test, fw)

        with open("acc_per_fold_train", 'wb') as fw:
            pickle.dump(acc_per_fold_train, fw)

        with open("loss_per_fold_test", 'wb') as fw:
            pickle.dump(loss_per_fold_test, fw)

        with open("loss_per_fold_train", 'wb') as fw:
            pickle.dump(loss_per_fold_train, fw)

        plt.close()
        plt.plot([i for i in range(1, 6)], acc_per_fold_test, "--or")
        plt.plot([i for i in range(1, 6)], acc_per_fold_train, "--ob")
        plt.plot([int(i) for i in range(1, 6)], [0.5 for i in range(1, 6)], "--g")
        plt.xlabel('folds', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        plt.legend(['test', 'train'], loc='best')
        plt.tight_layout()
        plt.savefig(f'accuracy_video_images.png')
        plt.close()

        plt.plot([i for i in range(1, 6)], loss_per_fold_test, "--or")
        plt.plot([i for i in range(1, 6)], loss_per_fold_train, "--ob")
        plt.xlabel('folds', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(['test', 'train'], loc='best')
        plt.tight_layout()
        plt.savefig(f'loss_video_images.png')
        plt.close()

        return my_history

    def nlp_model_architecture(self):
        embedding_layer = Embedding(self.num_words, 100,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.length_longest_sentence,
                                    trainable=False)
        self.model1.add(embedding_layer)
        self.model1.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
        self.model1.add(Dense(units=1, activation='sigmoid'))

        self.model1.compile(loss=binary_crossentropy,
                            optimizer=Adam(),
                            metrics=['accuracy'])

    def run_nlp_network_model(self, sentences_padding, score_groups):
        """
        :param sentences_padding: padded sentences
        :param score_groups: group levels of scores
        :return: model history of nlp architecture
        """
        # split train - test sets using padded sentences
        x_train_pad, x_test_pad, y_train_pad, y_test = train_test_split(sentences_padding, score_groups,
                                                                        test_size=0.3,
                                                                        random_state=563)
        history = self.model1.fit(x_train_pad, y_train_pad, batch_size=8, epochs=300,
                                  verbose=2, validation_split=0.2)

    @staticmethod
    def run_nlp_random_forest(x, y):
        """
        :param x: word counter
        :param y: score groups
        :return: random forest classification
        """

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=563)

        model = RandomForestClassifier(n_estimators=200,
                                       criterion='entropy', max_depth=7, random_state=593)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        print(classification_report(y_test, y_pred, target_names=['good', 'very good']))

        scores = cross_val_score(model, x, y, cv=10)
        print("cross validation Scores : " + (10 * "{:.3f} ").format(*scores))
        print(sum(scores) / len(scores))
        print(np.std(scores))
