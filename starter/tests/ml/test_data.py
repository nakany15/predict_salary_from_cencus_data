from starter.ml.data import process_data
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

class TestProcessData():
    @pytest.fixture
    def input_X(self):
        input_dict = {
            'X1':[1,2,3,4,5],
            'X2':[6,7,8,9,10],
            'category1': ['cate1', 'cate1', 'cate2', 'cate3', 'cate3'],
            'y':['red', 'red', 'blue', 'blue', 'blue']
        }
        df = pd.DataFrame.from_dict(input_dict)
        return df 

    @pytest.fixture
    def lb_fixt(self, input_X):
        lb = LabelBinarizer()
        y = input_X['y']
        y = lb.fit_transform(y.values).ravel()
        return lb
        
    @pytest.fixture
    def encoder_fixt(self, input_X):
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_categorical = encoder.fit(input_X[['category1']].values)
        return X_categorical
        
    def test_data_shape(self, input_X):
        cat_features = ['category1']
        X_train, y_train, encoder, lb = process_data(
            input_X, 
            categorical_features = cat_features,
            label = 'y'
        )
        assert X_train.shape == (5, 5)
        assert len(set(y_train)) == 2
        
    def test_data_shape_without_label(self, input_X, lb_fixt, encoder_fixt):
        cat_features = ['category1']
        X_train, y, encoder, lb = process_data(
            input_X, 
            training = False,
            categorical_features = cat_features,
            encoder = encoder_fixt,
            lb = lb_fixt
        )
        assert X_train.shape == (5, 6)
        assert len(y) == 0