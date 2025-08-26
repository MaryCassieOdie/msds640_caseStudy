import pandas as pd
from datetime import datetime
from sklearn.utils import resample

#Setup class
class DataHandler:

    original_data = []
    cleaned_data = []
    
    def __init__(self, data):
        self.original_data = data


    def describe_data(self, flag):
        data = self.original_data if flag == 'O' else self.cleaned_data

        print("Info")
        print(data.info())
        print("Describe")
        print(data.describe())
        print("Samples")
        print(data.head())

    def calculate_age(self, row):
        return 2025 - datetime.strptime(row['AGE'], '%Y-%m-%d').year

    def calculate_score(self, row):
        if pd.isnull(row['RATING']):
            return -1
        elif row['RATING'] == '-1':
            return -1
        else:
            if row['RATING'][0] == 'A':
                return 500 + (4-int(row['RATING'][1]))*30
            elif row['RATING'][0] == 'B':
                return 400 + (4-int(row['RATING'][1]))*30
            elif row['RATING'][0] == 'C':
                return 300 + (4-int(row['RATING'][1]))*30
            elif row['RATING'][0] == 'D':
                return 200 + (4-int(row['RATING'][1]))*30
            else:
                return 100 + (4-int(row['RATING'][1]))*30

    def clean_data(self):
        data = self.original_data.rename(columns={'Задолженность': 'DEBT', 'Просрочка, дни': 'DAYS_OVERDUE', 
                            'Первоначльный лимит':'LIMIT', 'Рейтинг кредитной истории':'RATING',
                            'BIRTHDATE':'AGE'})

        # 1= MALE, 0=FEMALE
        data['SEX'] = data['SEX'].replace({'Мужской': 1, 'Женский': 0})

        # 1Высшее = College, 0Среднее = Highschool, 2Среднее специальное = Technical College, 3Неоконченное высшее = partial college, 4**Послевузовское = Graduate
        data['EDU'] = data['EDU'].replace({'Высшее': 1, 'Среднее': 0,
                                            'Среднее специальное': 2, 
                                            'Неоконченное высшее': 3,
                                            '**Послевузовское': 4})

        #Transform Birthdate to Age
        data['AGE'] = data.apply(self.calculate_age, axis=1)

        #Transform credit rating to numerical score
        data['RATING'] = data.apply(self.calculate_score, axis=1)

        self.cleaned_data = data.drop(columns=['SCORINGMARK', 'VELCOMSCORING','LV_AREA','LV_SETTLEMENTNAME','INDUSTRYNAME','TERM','CLIENTID'])

    def get_cleaned_data(self):
        return self.cleaned_data
    
    def over_sample(self):
        majority = self.cleaned_data[self.cleaned_data['SEX'] == 1]
        minority = self.cleaned_data[self.cleaned_data['SEX'] == 0]

        minority = resample(minority, replace=True, n_samples=len(majority), random_state=42)

        return pd.concat([minority, majority])
    
    def under_sample(self):
        majority = self.cleaned_data[self.cleaned_data['SEX'] == 1]
        minority = self.cleaned_data[self.cleaned_data['SEX'] == 0]

        majority = resample(majority, replace=False, n_samples=len(minority), random_state=42)

        return pd.concat([minority, majority])
    
