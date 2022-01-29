import numpy as np
import random
from edit_video_text import EditVideo
from EDA import EDA
from modeling import Models
from word_embedding import WordEmbedding
my_seed = 512
np.random.seed(my_seed)
random.seed(my_seed)
# project libraries


class Main:

    @staticmethod
    def run():
        """
        Enter point  to run the program
        """
        image_dir = 'C:\\Users\\1\PycharmProjects\pythonProject5\data\gym_vault_samples_len_100_lstm'\
                    '\gym_vault_samples_len_100_lstm'

        # 1. Get the image data and text
        images, scores, levels, score_text = EditVideo.create_data(image_dir)

        # 2. EDA + preprocessing
        # 2.1 images
        y_score_groups = EDA.score_level(scores, levels)
        EDA.display_images(images, scores, levels)
        x_scaled_train, x_scaled_test, y_train, y_test = EDA.get_standardized_train_test_sets(
            images[0:60], y_score_groups[0:60])
        # 2.2 text
        dataset = EDA.read_text(scores, score_text)
        clean_dataset = EDA.nlp_pre_processing(dataset, scores, y_score_groups)
        word_embedding = WordEmbedding()
        word_embedding.token_sentences(clean_dataset)
        word_embedding.word_2_vector()
        word_embedding.read_embedding_word2vec_file()
        sentences_padding, score_groups = word_embedding.pad_sentences(clean_dataset)
        embedding_matrix = word_embedding.embedding_matrix()
        x_word_counter = word_embedding.count_vectorizer(clean_dataset)

        # 3. modeling

        models = Models(embedding_matrix, word_embedding.length_longest_sentence,
                        word_embedding.num_words)
        # 3.1 images model
        #models.image_modeling_architecture()
        # history_images = models.run_image_model(x_scaled_train, y_train, x_scaled_test, y_test)
        history_images = models.run_image_model(images[146:177], y_score_groups[146:177], x_scaled_test, y_test)

        # 3.2 text model
        models.nlp_model_architecture()
        history_nlp = models.run_nlp_network_model(sentences_padding, score_groups)
        models.run_nlp_random_forest(x_word_counter, clean_dataset.loc[:, 'y'])






if __name__ == '__main__':
    Main.run()
