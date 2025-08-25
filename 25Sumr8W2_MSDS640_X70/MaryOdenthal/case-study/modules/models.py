from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


#Setup class
class ModelRunner:

    ridge_pipeline = []
    data = []
    lasso_pipeline = []
    f_test = []
    t_test = []
    f_train = []
    t_train = []


    def __init__(self, data):
        self.data = data
        self.ridge_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic_regression', Ridge())
        ])

        self.lasso_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic_regression', Lasso())
        ])



        

#create & train model
    def train(self, target):
        #split features and target
        target_data = self.data[target]
        feature_data = self.data.drop(target, axis=1)
        f_train, f_test, t_train, t_test = train_test_split(feature_data, target_data)

        self.f_test = f_test
        self.t_test = t_test
        self.f_train = f_train
        self.t_train = t_train

        self.ridge_pipeline.fit(f_train, t_train)
        self.lasso_pipeline.fit(f_train, t_train)

        

    def predict(self, model_flag):
        total_results = self.f_test.copy(deep=True)


        if model_flag == 'R':
            y_pred = self.ridge_pipeline.predict(self.f_test)
        elif model_flag == 'L':
             y_pred = self.lasso_pipeline.predict(self.f_test)
        else: 
            raise ValueError
        
        total_results['PREDICTIONS'] = y_pred
        total_results['TRUE_LIMIT'] = self.t_test
        return total_results

    def adhoc_predict(self, model_flag, test_case):
        total_results = test_case

        if model_flag == 'R':
            y_pred = self.ridge_pipeline.predict(total_results)
        elif model_flag == 'L':
             y_pred = self.lasso_pipeline.predict(total_results)
        else: 
            raise ValueError
        
        total_results['PREDICTIONS'] = y_pred
        return total_results
    
    def get_test_data(self):
        full_test_data = self.f_test
        full_test_data['TRUE_LIMIT'] = self.t_test
        return full_test_data
    
    def save_models(self, file_path):
        full_path = file_path+'ridge_pipe.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(self.ridge_pipeline, f)
        
        full_path = file_path+'lasso_pipe.pkl'
        with open(full_path, 'wb') as f:
            pickle.dump(self.lasso_pipeline, f)
        
        full_path = file_path+'test_data.pkl'
        test_data = self.f_test
        test_data['LIMIT'] = self.t_test
        with open(full_path, 'wb') as f:
            pickle.dump(test_data, f)

        full_path = file_path+'train_data.pkl'
        train_data = self.f_train
        train_data['LIMIT'] = self.t_train
        with open(full_path, 'wb') as f:
            pickle.dump(train_data, f)
        

    def load_models(self, file_path):
        full_path = file_path+'ridge_pipe.pkl'
        with open(full_path, 'rb') as f:
            self.ridge_pipeline = pickle.load(f)

        full_path = file_path+'lasso_pipe.pkl'
        with open(full_path, 'rb') as f:
            self.lasso_pipeline = pickle.load(f)

        full_path = file_path+'test_data.pkl'
        with open(full_path, 'rb') as f:
            test_data = pickle.load(f)
            self.f_test = test_data.drop('LIMIT', axis=1)
            self.t_test = test_data['LIMIT']

        full_path = file_path+'train_data.pkl'
        with open(full_path, 'rb') as f:
            train_data = pickle.load(f)
            self.f_train = train_data.drop('LIMIT', axis=1)
            self.t_train = train_data['LIMIT']