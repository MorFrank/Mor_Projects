# libraries imports
import pandas as pd
# project imports
from get_university_table import GetUniversityRank


class JoinTable:
    """
    What this class does
    """

    @staticmethod
    def __init__(self):
        pass

    @staticmethod
    def join_table(scientists_table, universities_table, year) -> pd.DataFrame:
        """
        :param year:
        :param universities_table:
        :param scientists_table:
        :return: joined tables of scientists_table and universities_table per year
        """
        scientists_table_year = scientists_table[scientists_table['year_prize'] == int(year)]
        # Left Join
        joined_table = pd.merge(scientists_table_year, universities_table, on='institution',
                                how='left')

        return joined_table

