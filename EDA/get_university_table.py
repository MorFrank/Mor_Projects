# libraries import
import requests
import pandas as pd
from bs4 import BeautifulSoup as bs


class GetUniversityRank:
    """
    What this class does
    """

    @staticmethod
    def __init__(self):
        pass

    # get institution ranking

    @staticmethod
    def get_rank(url_universities) -> pd.DataFrame:
        """
        :param url_universities:
        :return: the data from the the api call as json
        """
        # get data from api
        page_html = requests.get(url_universities).text
        soup = bs(page_html, 'html.parser')

        # get institution name and ranking
        institution_name = []
        institution_ranking = []

        all_universities = soup.find('table', width="0", border="0").find_all('tr', class_=["bgfd", "bgf5"])
        for university in all_universities:

            # renaming of university names to fit the ones extracted from the api

            if 'The University of' in university.find('td', class_="left").a.text:
                institution_name.append(university.find('td', class_="left").a.text[18:] + ' University')
                institution_ranking.append(university.td.text.split('-')[0])

            elif 'University of' in university.find('td', class_="left").a.text:
                institution_name.append(university.find('td', class_="left").a.text[14:] + ' University')
                                        #split(' ')[0].replace(',', '') + ' University')
                institution_ranking.append(university.td.text.split('-')[0])

            else:
                institution_name.append(university.find('td', class_="left").a.text)
                institution_ranking.append(university.td.text.split('-')[0])

        institution_name_ranking = [institution_name, institution_ranking]

        return GetUniversityRank.get_university_rank_table(university_table=institution_name_ranking)

    @staticmethod
    def get_university_rank_table(university_table: list) -> pd.DataFrame:
        """
        :param city_rank:
        :param university_table:
        :return: table of university ranking as DataFrame
        """
        # get DataFrame table from the university table lists
        institution_name = pd.Series(university_table[0])
        institution_rank = pd.Series(university_table[1])

        university_data = {'institution': institution_name,
                           'institution_rank': institution_rank}

        scientists_table = pd.DataFrame(university_data)

        return scientists_table


