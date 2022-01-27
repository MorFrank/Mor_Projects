# library imports
import pandas as pd
import logging
from datetime import datetime
# project imports
from scientists_details import ScientistsDetails


class DataProcess:
    """
    What this class does
    """
    def __init__(self):
        pass

    # get details of all the scientists
    @staticmethod
    def process_scientists_data(data) -> ScientistsDetails:

        # return the scientists data
        details = ScientistsDetails(scientist_id=[], firstname=[], surname=[], born=[], gender=[],
                                    year_prize=[], category=[], institution=[], city=[], country=[], age=[])

        for elem in data['laureates']:
            # check if affiliations not empty
            if elem['prizes'][0]['affiliations'][0]:
                try:
                    details.add_scientist(elem['id'], elem['firstname'], elem['surname'],
                                          datetime.strptime(elem['born'][:4], "%Y").year,
                                          elem['gender'],
                                          datetime.strptime(elem['prizes'][0]['year'], "%Y").year,
                                          elem['prizes'][0]['category'],
                                          elem['prizes'][0]['affiliations'][0]["name"],
                                          elem['prizes'][0]['affiliations'][0]["city"],
                                          elem['prizes'][0]['affiliations'][0]['country'])
                    details.get_age(born=datetime.strptime(elem['born'][:4], "%Y").year,
                                    year_prize=datetime.strptime(elem['prizes'][0]['year'], "%Y").year)
                except Exception as error:
                    logging.error(error, exc_info=True)
                    print(error)
            else:
                try:
                    details.add_scientist(elem['id'], firstname=elem['firstname'],
                                          surname=elem['surname'],
                                          born=datetime.strptime(elem['born'][:4], "%Y").year,
                                          gender=elem['gender'],
                                          year_prize=datetime.strptime(elem['prizes'][0]['year'], "%Y").year,
                                          category=elem['prizes'][0]['category'],
                                          institution='', city='', country='')
                    details.get_age(born=datetime.strptime(elem['born'][:4], "%Y").year,
                                    year_prize=datetime.strptime(elem['prizes'][0]['year'], "%Y").year)
                except Exception as error:
                    logging.error(error, exc_info=True)
                    print(error)

        return details
