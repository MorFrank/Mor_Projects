# library imports
from datetime import datetime, timedelta

# project imports


class ScientistsDetails:
    """
    What this class does
    """

    def __init__(self,
                 scientist_id: [], firstname: [], surname: [], born: [], gender: [], year_prize: [],
                 category: [], institution: [], city: [], country: [], age: []):
        self.scientist_id = scientist_id
        self.firstname = firstname
        self.surname = surname
        self.born = born
        self.gender = gender
        self.age = age
        self.year_prize = year_prize
        self.category = category
        self.institution = institution
        self.city = city
        self.country = country

    def add_scientist(self, scientist_id, firstname: str, surname: str, born: int,
                      gender: str, year_prize: int, category: str, institution: str,
                      city: str, country: str):
        self.scientist_id.append(scientist_id)
        self.firstname.append(firstname)
        self.surname.append(surname)
        self.born.append(born)
        self.gender.append(gender)
        self.year_prize.append(year_prize)
        self.category.append(category)

        if 'University of' in institution:
            self.institution.append(institution[14:] + ' University')
        else:
            self.institution.append(institution)
        self.city.append(city)
        self.country.append(country)

    # get age when winning the nobel prize
    def get_age(self, born: int, year_prize: int):
        self.age.append((year_prize-born))




