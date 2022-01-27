# library imports
import json
import os
import pandas as pd
# project imports
from get_data import GetData
from data_process import DataProcess
from scientists_details import ScientistsDetails
from get_scientists_table import ScientistsTable
from get_university_table import GetUniversityRank
from join_table import JoinTable
from analysis_maneger import AnalysisManager
from plot_results import PlotResults


class Main:
    """
    Managing all the process
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        """
        Enter point  to run the program
        """

        # define api to get nobel scientist data
        domain = 'http://api.nobelprize.org'
        root = 'v1/laureate.json'
        url_scientists = f'{domain}/{root}'
        scientists_data_json = GetData.get_data(url_scientists=url_scientists)

        # use web scraping to get institution ranking
        domain_university = 'http://www.shanghairanking.com'
        #ranking_years = ['2010', '2011', '2013', '2015', '2016', '2018', '2019', '2020']
        ranking_years = ['2003', '2004', '2005', '2006', '2007', '2009', '2010', '2011', '2013',
                         '2014', '2015', '2016', '2017', '2018', '2019', '2020']

        # 1. get lists of scientists data
        scientists_data = DataProcess.process_scientists_data(data=scientists_data_json)
        # create scientists table as Dataframe
        scientists_table = ScientistsTable.get_scientists_table(scientists_data=scientists_data)

        statistics_all_years = AnalysisManager(ages_year=[], age=[], year_rank=[], category_count=[], gender=[],
                                               num_males=[], num_females=[],
                                               institutions=[], institutions_counts=[],
                                               institution_ranks=[], category_male_count=[],
                                               category_female_count=[], year_rank_male=[], year_rank_female=[],
                                               num_prizes=[], all_gender_country=[], all_country_category=[])

        if not os.path.exists("data/statistics_all_years.json"):
            for ranking_year in ranking_years:
                url_university_ranking = domain_university + f'/ARWU{ranking_year}.html'

                # 2. create university ranking table as DataFrame
                universities_table = GetUniversityRank.get_rank(url_universities=url_university_ranking)

                # 3. join scientists_table & university_table
                joined_table_per_year = JoinTable.join_table(scientists_table=scientists_table,
                                                             universities_table=universities_table, year=ranking_year)
                # 4. Parameters investigation
                statistics_all_years.age_statistics(joined_table_per_year.age, ranking_year)
                statistics_all_years.category_counter(joined_table_per_year.category, joined_table_per_year.gender,
                                                      ranking_year)
                statistics_all_years.gender_counter(joined_table_per_year.gender, ranking_year)
                statistics_all_years.rank_university(joined_table_per_year.institution,
                                                     joined_table_per_year.institution_rank,
                                                     ranking_year)
                statistics_all_years.number_prizes_per_year(joined_table_per_year.year_prize, ranking_years)
                statistics_all_years.gender_in_country(joined_table_per_year.gender, joined_table_per_year.country)
                statistics_all_years.category_in_country(joined_table_per_year.country, joined_table_per_year.category)

            statistics_all_years.save_analyses_manager_json(path="data/statistics_all_years.json")
        else:
            statistics_all_years.load_analyses_manager_json(path="data/statistics_all_years.json")

        # 5 plot results
        PlotResults.plot_age(statistics_all_years=statistics_all_years, years=ranking_years)
        PlotResults.plot_category(statistics_all_years=statistics_all_years, years=ranking_years)
        PlotResults.plot_gender(statistics_all_years=statistics_all_years, years=ranking_years)
        PlotResults.plot_institutions_score(statistics_all_years=statistics_all_years, years=ranking_years)
        PlotResults.plot_number_of_prices_per_year(statistics_all_years=statistics_all_years, years=ranking_years)
        PlotResults.plot_gender_in_country(statistics_all_years=statistics_all_years, years=ranking_years)
        PlotResults.plot_category_in_country(statistics_all_years=statistics_all_years, years=ranking_years)



        pd.set_option('display.max_columns', 20)
        pd.set_option('display.width', 2000)
        #print(scientists_table[scientists_table.year_prize == 2018])
        print(joined_table_per_year)


if __name__ == '__main__':
    Main.run()
