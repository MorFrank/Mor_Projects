# library imports
import requests
import json
from bs4 import BeautifulSoup as bs
# project imports
from scientists_details import ScientistsDetails


class GetData:
    """
    What this class does
    """

    def __init__(self):
        pass

    # get scientists data in json format

    @staticmethod
    def get_data(url_scientists) -> dict:
        """
        :param url_scientists: the url to get the data from
        :return: the data from the the api call as json
        """
        # get data from api
        answer_json = json.loads(requests.get(url=url_scientists).text)

        # return the data
        return answer_json











