import cv2
import os
import numpy as np


class EditVideo:

    @staticmethod
    def get_frames(video_path: str, seq_len: int, img_height: int, img_width: int):
        """
        :param img_width: frame width
        :param img_height: frame height
        :param seq_len: number of frames in each video
        :param video_path: path for the video files
        :return: frames for each video
        """
        frames_list = []

        video_obj = cv2.VideoCapture(video_path)
        # Used as counter variable
        count = 1

        while count <= seq_len:

            success, image = video_obj.read()
            if success:
                image = cv2.resize(image, (img_height, img_width))
                frames_list.append(image)
                count += 1
            else:
                print("Defected frame")
                # break

        return frames_list

    @staticmethod
    def create_data(video_path: str):
        """
        :param video_path: path for video files
        :return: x-frames for each video, scores, levels, score_text
        """

        x = []
        scores = []
        score_text = []
        levels = []

        # list of all avi files for X
        files_list = sorted(
            [f for f in os.listdir(video_path) if f.endswith('.' + 'avi')])

        for k in range(0, len(files_list)):
            frames = EditVideo.get_frames(os.path.join(os.path.join(video_path, files_list[k])),
                                          seq_len=100, img_height=128, img_width=128)
            x.append(frames)

        # list of all scores for Y
        with open("scores.txt", 'r') as file:
            for line in file:
                scores.append(float(line.split(',')[0]))

        # list of all levels for Y1
        with open('levels.txt', 'r') as file:
            for line in file:
                levels.append(float(line.split('\n')[0]))

        # list of text score

        with open('scores.txt', 'r') as file:
            for line in file:
                score_text.append(line.strip().split(',')[1])

        x = np.array(x).reshape(176, 100, 128, 128, 3)
        scores = np.array(scores).reshape(176, 1)
        levels = np.array(levels).reshape(176, 1)

        return x, scores, levels, score_text

