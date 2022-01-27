# library imports
import pandas as pd
from datetime import datetime
# project imports
from scientists_details import ScientistsDetails


class ScientistsTable:
    """
    What this class does
    """
    def __init__(self):
        pass

    # create table all the scientists as dataframe
    @staticmethod
    def get_scientists_table(scientists_data) -> pd.DataFrame:
        scientist_id = pd.Series(scientists_data.scientist_id)
        firstname = pd.Series(scientists_data.firstname)
        surname = pd.Series(scientists_data.surname)
        born = pd.Series(scientists_data.born)
        gender = pd.Series(scientists_data.gender)
        age = pd.Series(scientists_data.age)
        year_prize = pd.Series(scientists_data.year_prize)
        institution = pd.Series(scientists_data.institution)
        category = pd.Series(scientists_data.category)
        city = pd.Series(scientists_data.city)
        country = pd.Series(scientists_data.country)

        data = {'scientist_id': scientist_id,
                'Firstname': firstname,
                'Surname': surname,
                'born': born,
                'gender': gender,
                'age': age,
                'year_prize': year_prize,
                'category': category,
                'institution': institution,
                'city': city, 'country': country}

        scientists_table = pd.DataFrame(data)

        return scientists_table


