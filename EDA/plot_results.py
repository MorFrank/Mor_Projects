# libraries imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from colour import Color
# project imports
from analysis_maneger import AnalysisManager
from get_university_table import GetUniversityRank


class PlotResults:
    """
    What this class does
    """

    @staticmethod
    def __init__(self):
        pass

    @staticmethod
    def plot_age(statistics_all_years: AnalysisManager, years: list) -> None:
        """
        :param years: all selected years
        :param statistics_all_years:
        :return: plot of the average age over the years
        """
        x = [elem[2] for elem in statistics_all_years.age]
        yr = [elem[1] for elem in statistics_all_years.age]
        y = [elem[0] for elem in statistics_all_years.age]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca()
        ax.set_title('Average and Std of age over the years', fontsize=27)
        ax.set_ylabel('average age', fontsize=27)
        ax.set_xlabel('year', fontsize=27)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        ax.bar(x, y, yerr=yr, error_kw=dict(lw=3, capsize=5, capthick=1))
        plt.tight_layout()
        plt.savefig("plots/AverageAage.png")
        plt.close()

        # plot age histogram per year
        fig = plt.figure(figsize=(15, 8))
        for num, elem in enumerate(statistics_all_years.ages_year):
            ax = fig.add_subplot(2, 8, num+1)
            ax.hist(elem)
            plt.title('{}'.format(years[num]), fontsize=15)
            ax.set_xlim([20, 90])
            ax.set_ylim([0, 4.5])
            plt.xticks(fontsize=15)
            y_ticks = np.arange(0, 7, 1)
            plt.yticks(y_ticks, fontsize=15)
            plt.ylim(0, 5.5, 1)
            plt.xlabel('age', fontsize=14)
            plt.ylabel('counts', fontsize=14)
            #plt.yticks(fontsize=15)
        fig.tight_layout()
        fig.savefig("plots/age histogram.png")
        plt.close()

    @staticmethod
    def plot_category(statistics_all_years: AnalysisManager, years: list) -> None:
        fig = plt.figure(figsize=(20, 15))
        ax = fig.gca()
        ax.set_title('Counts of categories over the years', fontsize=43)

        # merge list of series
        all_df = pd.concat(statistics_all_years.category_count, axis=1)

        all_df.plot(kind="bar", ax=ax)
        ax.set_xlabel(xlabel='categories', fontsize=33)
        ax.set_ylabel(ylabel='counts', fontsize=40)
        ax.set_xticklabels(labels=all_df.index, fontsize=33)
        y_ticks = np.arange(0, 4, 1)
        plt.xticks(rotation=0)
        plt.yticks(y_ticks, fontsize=40)
        ax.legend(years,
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                  fontsize=29)
        plt.tight_layout()
        plt.savefig("plots/category.png")
        plt.close()

        # plot male and female on the same figure
        gender = ['males', 'females']
        legend_gender_years = [statistics_all_years.year_rank_male, statistics_all_years.year_rank_female]

        all_df_male = pd.concat(statistics_all_years.category_male_count, axis=1)
        all_df_female = pd.concat(statistics_all_years.category_female_count, axis=1)
        all_df = [all_df_male, all_df_female]

        fig = plt.figure(figsize=(15, 8))
        for num, gender in enumerate(gender):
            ax = fig.add_subplot(2, 1, num + 1)
            ax.set_title(f'Counts of categories over the years -  {gender}', fontsize=15)
            #all_df[num].plot(kind="bar", ax=ax)
            all_df[num].reindex(all_df[0].index.tolist()).plot(kind='bar', ax=ax)


            ax.set_xlabel(xlabel='categories', fontsize=15)
            ax.set_ylabel(ylabel='counts', fontsize=15)
            #ax.set_xticklabels(labels=all_df[num].index, fontsize=15)
            plt.xticks(rotation=0, fontsize=15)
            y_ticks = np.arange(0, 5, 1)
            plt.yticks(y_ticks, fontsize=15)
            plt.ylim(0, 4, 1)
            ax.legend(legend_gender_years[num],
                      bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                      fontsize=15)
        fig.tight_layout()
        fig.savefig("plots/category_SUB.png")
        plt.close()

    @staticmethod
    def plot_gender(statistics_all_years: AnalysisManager, years: list) -> None:

        # plot gender over the years
        fig = plt.figure(figsize=(20, 15))
        ax = fig.gca()
        ax.set_title('Counts of gender over the years', fontsize=40)

        # merge list of dataframes
        all_df = pd.concat(statistics_all_years.gender, axis=1)

        all_df.plot(kind="bar", ax=ax, rot=0)
        ax.set_xlabel(xlabel='gender', fontsize=40)
        ax.set_ylabel(ylabel='counts', fontsize=40)
        ax.set_xticklabels(labels=all_df.index, fontsize=40)
        plt.yticks(fontsize=40)
        ax.legend(years,
                  bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                  fontsize=29)
        plt.tight_layout()
        plt.savefig("plots/gender.png")
        plt.close()

        # plot gender per year

        for num, elem in enumerate(statistics_all_years.gender):
            elem.plot(kind='pie', label="", autopct='%1.1f%%')
            plt.title("Gender {}".format(years[num]))
            plt.tight_layout()
            plt.savefig("plots/gender {}.png".format(years[num]))
            plt.close()

        # plot line graph of percentage of gender over the years

        array_males = np.asarray(statistics_all_years.num_males)
        array_females = np.asarray(statistics_all_years.num_females)
        x_male = 100*(array_males/(array_males + array_females))
        x_female = 100*(array_females/(array_males + array_females))

        fig = plt.figure(figsize=(30, 15))
        ax = fig.gca()
        ax.set_title('Gender over the years', fontsize=43)
        ax.set_ylabel('[%]', fontsize=40)
        ax.set_xlabel('years', fontsize=40)
        ax.plot([int(i) for i in years], x_male, "--og", linewidth=5, markersize=20)
        ax.plot([int(i) for i in years], x_female, "--ob", linewidth=5, markersize=20)

        plt.xlim(2009, 2021, 1)
        plt.yticks(fontsize=40)
        ax.legend(['males', 'females'], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.,
                  fontsize=40)

        # plot average [%] for male and female
        avg_male = np.mean(x_male)
        avg_female = np.mean(x_female)
        ax.plot([int(i) for i in years], [avg_male for i in range(len(array_males))], c="k", linewidth=2.5)
        ax.plot([int(i) for i in years], [avg_female for i in range(len(array_males))], c="k", linewidth=2.5)
        #ax.set_xticklabels(labels=[i for i in range(2010, 2021)], fontsize=40)
        x_ticks = np.arange(2010, 2021, 1)
        plt.xticks(x_ticks, fontsize=40)

        plt.tight_layout()
        plt.savefig("plots/Gender_years.png")
        plt.close()

    @staticmethod
    def plot_number_of_prices_per_year(statistics_all_years: AnalysisManager, years: list) -> None:
        df_all_years = pd.concat(statistics_all_years.num_prizes, ignore_index=True)
        prices_per_year = df_all_years.groupby(['year_prize'])['year_prize'].count()

        fig = plt.figure(figsize=(14, 10))
        ax = fig.gca()
        prices_per_year.plot(kind='line', ax=ax,  color='blue', linestyle='dashed', marker='o',
                             linewidth='3', markersize='11', markerfacecolor='red')

        ax.set_title('Number of nobel prizes over the years', fontsize=27)

        ax.set_xlabel(xlabel='years', fontsize=25)
        ax.set_ylabel(ylabel='counts', fontsize=25)
        x_ticks = np.arange(2003, 2020, 1)
        plt.xticks(x_ticks, fontsize=23, rotation=45)
        #y_ticks = np.arange(9, 16, 1)
        y_ticks = np.arange(7, 16, 1)
        plt.yticks(y_ticks, fontsize=23)
        #plt.ylim(9.5, 14.5, 1)
        plt.ylim(7.5, 14.5, 1)

        avg_prizes = np.mean(prices_per_year)
        ax.plot([int(i) for i in years], [avg_prizes for i in range(len(years))], c="k", linewidth=2.5)

        std_prizes = np.std(prices_per_year)

        ax.plot([int(i) for i in years], [avg_prizes + std_prizes for i in range(len(years))], c="k", linewidth=2.5)
        ax.plot([int(i) for i in years], [avg_prizes - std_prizes for i in range(len(years))], c="k", linewidth=2.5)

        plt.tight_layout()
        plt.savefig("plots/number_prizes_years.png")
        plt.close()

    @staticmethod
    def plot_gender_in_country(statistics_all_years: AnalysisManager, years: list) -> None:
        """
        :param statistics_all_years:
        :param years:
        :return:
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.gca()
        df_gender_country = pd.concat(statistics_all_years.all_gender_country, ignore_index=True)

        # nan_value = float("NaN")
        df_gender_country.replace("", "N/A", inplace=True)
        # df_gender_country.dropna(inplace=True)

        df_gender_country.groupby(['country', 'gender'])['gender']\
            .count().unstack(1).plot.bar(ax=ax)

        ax.set_title('Gender distribution in countries', fontsize=33)

        ax.set_xlabel(xlabel='countries', fontsize=30)
        ax.set_ylabel(ylabel='counts', fontsize=30)
        plt.xticks(fontsize=25, rotation='45', rotation_mode='anchor', ha='right')
        plt.yticks(fontsize=25)
        plt.tight_layout()
        plt.legend(['female', 'male'], loc='best', fontsize=25)
        plt.figtext(-0.0003, 0.008, "** based on counting the following years:\n"
                                    "    {} ".format(years), ha="left", fontsize=14,
                    bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})
        plt.savefig("plots/number_gender_countries.png")
        plt.close()

        # plot countries per year
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle('Nobel prize origin country', fontsize=17)
        colors = ['blue', 'red', 'green', 'orange', 'violet', 'cyan', 'yellow']
        for num, elem in enumerate(statistics_all_years.all_gender_country):
            ax = fig.add_subplot(2, 8, num + 1)
            df_country = pd.DataFrame(elem['country']).replace("", 'N/A')
            df_country_sort = df_country.groupby('country').size().sort_values()
            ax.pie(df_country_sort.values.tolist(), labels=df_country_sort.index.tolist(),
                   autopct='%1.0f%%', startangle=90,  textprops={'fontsize': 12}, colors=colors)
            ax.set_title("countries {}".format(years[num]), fontsize=13)

        plt.tight_layout()
        plt.savefig("plots/countries_distribution.png")
        plt.close()

    @staticmethod
    def plot_category_in_country(statistics_all_years: AnalysisManager, years: list) -> None:
        df = pd.concat(statistics_all_years.all_country_category, ignore_index=True)

        # get country area
        country_table = pd.read_csv('countries.csv', keep_default_na=False)
        countries_in_table = country_table['Country'].str.lower().tolist()
        area_in_table = country_table.Area.tolist()
        country_names = list(df.groupby(['country', 'category'], as_index=False).groups.keys())
        unzipped_object = zip(*country_names)
        unzipped_list = list(unzipped_object)
        country_names = list(map(str.lower, unzipped_list[0]))
        country_index = countries_in_table
        area_list = [area_in_table[int(country_index.index(i))] for i in country_names
                     if i in countries_in_table]
        area_list_array = np.asarray(area_list)
        #area_list_array = pd.DataFrame(area_list_array)
        ######################################################

        fig = plt.figure(figsize=(17, 10))
        ax = fig.gca()
        #df.groupby(['country', 'category'])['category'].count().div(area_list_array).unstack(1).plot.\
            #barh(ax=ax, stacked=True).legend(fontsize=25,
                                             #bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        df.groupby(['country', 'category'])['category'].count().div(area_list_array).unstack(1).plot.\
            barh(ax=ax, stacked=True).legend(fontsize=18,
                                             bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        ax.set_title('Categories in countries', fontsize=24)
        #ax.set_xlabel(xlabel='counts x 10^-6', fontsize=28)
        #ax.set_ylabel(ylabel='countries', fontsize=28)
        ax.set_xlabel(xlabel='counts/area[sq mi] x 10^-6', fontsize=24)
        ax.set_ylabel(ylabel='countries', fontsize=24)
        #x_ticks = np.arange(0, 48, 5)
        x_ticks = np.arange(0, 125, 10)
        #plt.xticks(x_ticks, fontsize=25)
        plt.xticks(x_ticks, fontsize=22)
        #plt.xlim(0, 49, 5)
        plt.xlim(0, 130, 10)

        plt.yticks(fontsize=22)

        plt.figtext(-0.01, 0.002, "** based on counting the following years:\n"
                                    "    {} ".format(years), ha="left", fontsize=14,
                    bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})

        plt.tight_layout()
        plt.savefig("plots/category_in_country.png")
        plt.close()

    @staticmethod
    def plot_institutions_score(statistics_all_years: AnalysisManager, years: list) -> None:
        """
        :param statistics_all_years:
        :param years:
        :return: institution score - the higher the better chances to get a nobel prize
        """
        df1 = pd.DataFrame(statistics_all_years.institutions, columns=['institutions'])
        df2 = pd.DataFrame(statistics_all_years.institution_ranks,  columns=['institution_ranks'])
        df2["institution_ranks"] = pd.to_numeric(df2["institution_ranks"])
        df3 = pd.concat([df1, df2], axis=1).dropna()

        names_institutions = df3.groupby(['institutions'])['institutions'].first().values
        count_institutions = df3.groupby(['institutions'])['institution_ranks'].count().values
        mean_rank = df3.groupby(['institutions'])['institution_ranks'].mean().values
        sum_of_ranks = df3['institution_ranks'].sum()

        institution_score = (1-(mean_rank/sum_of_ranks)) * count_institutions

        df4 = pd.DataFrame(data=names_institutions, columns=['institutions'])
        df5 = pd.DataFrame(data=institution_score, columns=['score'])
        df6 = pd.concat([df4, df5], axis=1)
        sorted_df = df6.sort_values(["score"], ascending=False)

        fig = plt.figure(figsize=(28, 20))
        ax = fig.gca()
        ax.set_title('Score of institutions', fontsize=43)
        plt.figtext(0, 0.008, "** score is based on averaging world rank of the following years:\n"
                             "    {} ".format(years), ha="left", fontsize=28,
                    bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 5})
        ax.set_ylabel('score', fontsize=40)
        ax.set_xlabel('institutions', fontsize=40)

        green = Color("green")
        colors = list(green.range_to(Color("red"), 26))
        colors = [color.rgb for color in colors]


        #idx = np.asarray([i for i in range(len(names_institutions))])
        ax.bar(sorted_df['institutions'], sorted_df['score'], width=0.5, color=colors)

        my_cmap = ListedColormap(colors)
        s = plt.cm.ScalarMappable(cmap=my_cmap.reversed())

        mn = int(np.floor(sorted_df['score'].min()))  # colorbar min value
        mx = int(np.ceil(sorted_df['score'].max()))  # colorbar max value
        md = (mx - mn) / 2
        cb = plt.colorbar(s, ticks=[0, 0.5, 1])
        cb.ax.set_yticklabels(['Low', 'Medium', 'High'])  # add the labels
        cb.ax.tick_params(labelsize=33)  # change font size

        #ax.set_xticks(names_institutions)
        #ax.set_xticklabels(names_institutions)

        plt.yticks(fontsize=40)
        plt.xticks(fontsize=30, rotation='45', rotation_mode='anchor', ha='right')
        plt.tight_layout()
        plt.savefig("plots/institutions_score.png")
        plt.close()


