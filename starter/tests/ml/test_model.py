from starter.ml.model import train_model
import pytest
import pandas as pd
import sklearn
class TestModel():
    @pytest.fixture
    def input_Xy(self):
        input_dict = {
            'X1':[1,2,3,4,5],
            'X2':[6,7,8,9,10],
            'y':[0, 1, 0, 0, 1]
        }
        df = pd.DataFrame.from_dict(input_dict)
        return [df.drop('y', axis = 1).values, df['y'].values]
    def test_model_instance(self, input_Xy):
        X_train = input_Xy[0]
        y_train = input_Xy[1]
        model = train_model(X_train, y_train)
        assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)
       