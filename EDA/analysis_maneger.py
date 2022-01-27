# libraries imports
import json
import jsonpickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# project imports
from get_university_table import GetUniversityRank


class AnalysisManager:
    """
    Analyses calculations and statistics
    """
    def __init__(self, ages_year: [], age: [], year_rank: [], category_count: [], gender: [],
                 num_males: [], num_females: [], institutions: [], institutions_counts: [], institution_ranks: [],
                 category_male_count: [], category_female_count: [], year_rank_male: [], year_rank_female: [],
                 num_prizes: [], all_gender_country: [], all_country_category: []):
        self.ages_year = ages_year
        self.age = age
        self.year_rank = year_rank
        self.category_count = category_count
        self.gender = gender
        self.num_males = num_males
        self.num_females = num_females
        self.institutions = institutions
        self.institution_ranks = institution_ranks
        self.institutions_counts = institutions_counts
        self.category_male_count = category_male_count
        self.category_female_count = category_female_count
        self.year_rank_male = year_rank_male
        self.year_rank_female = year_rank_female
        self.num_prizes = num_prizes
        self.all_gender_country = all_gender_country
        self.all_country_category = all_country_category

    def age_statistics(self, age: pd.tseries, year_rank: str):
        """
        :param year_rank:
        :param age:
        :return: average age per year
        """

        self.ages_year.append(age)
        self.age.append([np.mean(age), np.std(age), year_rank])

    def category_counter(self, category: pd.tseries, gender: pd.tseries, year_rank: str):
        """
        :param gender: gender
        :param year_rank:
        :param category: research field
        :return: number of each category per year
        """

        df = pd.DataFrame(category)
        df1 = pd.DataFrame(gender)
        df_all = pd.concat([df, df1], axis=1)
        self.category_count.append(df.groupby(['category'])['category'].count())
        self.category_male_count.append(df_all[df_all.gender == 'male'].groupby(['category'])
                                        ['category'].count())
        self.year_rank_male.append(year_rank)

        try:
            self.category_female_count.append(df_all[df_all.gender == 'female'].groupby(['category'])
                                              ['category'].count())
            self.year_rank_female.append(year_rank)
        except:
            pass

    def gender_counter(self, gender: pd.tseries, year_rank: str):
        """
        :param num_males: append (number of males per year)
        :param gender: male/female
        :param year_rank:
        :return: extend (number of each gender type (male/female) per year)
        """

        df = pd.DataFrame(gender)
        #self.gender.append(df.groupby(['gender'])['gender'].count())
        self.gender.append(df.groupby('gender').size())
        self.num_males.extend(df.groupby('gender').get_group('male').count())
        try:
            self.num_females.extend(df.groupby('gender').get_group('female').count())
        except:
            self.num_females.extend([0])


        #self.gender.append([gender.value_counts().index.tolist(), gender.value_counts().values.tolist(),
                            #year_rank])

        return gender

    def number_prizes_per_year(self, year_prize: pd.tseries, years: list):
        """
        :param years: selected years
        :param year_prize: series of year (the same as year rank)
        :return:
        """
        self.num_prizes.append(pd.DataFrame(year_prize, columns=['year_prize']))

    def gender_in_country(self, gender: pd.tseries, country: pd.tseries):

        df_gender = pd.DataFrame(gender, columns=['gender'])
        df_country = pd.DataFrame(country, columns=['country'])
        df_gender_country = pd.concat([df_gender, df_country], axis=1)
        self.all_gender_country.append(df_gender_country)

    def category_in_country(self, country: pd.tseries, category: pd.tseries):
        df = pd.DataFrame(country)
        df1 = pd.DataFrame(category)
        df_all = pd.concat([df, df1], axis=1).replace("", 'N/A')
        self.all_country_category.append(df_all)

    def rank_university(self, institution: pd.tseries, institution_rank: pd.tseries, year_rank: str):
        """
        :param institution: university name
        :param institution_rank: world rank
        :param year_rank: year rank
        :return: list of institution names and rank for all years
        """
        self.institutions.extend(institution)
        self.institution_ranks.extend(institution_rank)

    def save_analyses_manager_json(self, path: str):
        self_json = jsonpickle.encode(self, unpicklable=False)
        with open(path, "w") as json_file:
            json_file.write(json.dumps(self_json))

    @staticmethod
    def load_analyses_manager_json(path: str):
        with open(path, "r") as json_file:
            json_obj = json.loads(json_file.read())

        return AnalysisManager(ages_year=json_obj["ages_year"],
                               age=json_obj["age"],
                               year_rank=json_obj["year_rank"],
                               category_count=json_obj["category_count"],
                               gender=json_obj["gender"],
                               num_males=json_obj["num_males"],
                               num_females=json_obj["num_females"],
                               institutions=json_obj["institutions"],
                               institutions_counts=json_obj["institutions_counts"],
                               institution_ranks=json_obj["institution_ranks"])












