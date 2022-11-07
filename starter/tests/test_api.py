import logging

from main import ModelParams, app
import pytest
from fastapi.testclient import TestClient

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_greet(test_app):
    response = test_app.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "This is an Income prediction API!!"

class TestPredict():
    @pytest.fixture(scope="module")
    def test_app():
        client = TestClient(app)
        return client
    @pytest.fixture
    def test_case_under_50k(self):
        body = ModelParams(
        **{
            "age": 39, 
            "workclass": "State-gov", 
            "fnlgt": 500, 
            "education": " HS-grad", 
            "education-num": 9,
           "marital-status": " Divorced", 
           "occupation": "Adm-clerical", 
           "relationship": "Not-in-family",
           "race": "White", 
           "sex": "Male",
           "capital-gain": 0, 
           "capital-loss": 0, 
           "hours-per-week": 40,
           "native-country": 
           "United-States"})
        return body
    def test_case_over_50k(self):
        body = ModelParams(
        **{
            "age": 39, 
            "workclass": "State-gov", 
            "fnlgt": 80000, 
            "education": " Masters", 
            "education-num": 14,
           "marital-status": " Married-civ-spouse", 
           "occupation": "Adm-clerical", 
           "relationship": "Not-in-family",
           "race": "White", 
           "sex": "Male",
           "capital-gain": 5000, 
           "capital-loss": 0, 
           "hours-per-week": 45,
           "native-country": 
           "United-States"})
        return body
    def test_predict_under_50k(self, test_case_under_50k, test_app):

        response = test_app.post("/predict", data=test_case_under_50k.json(by_alias=True))

        assert response.status_code == 200
        assert response.json()["prediction"] == "<=50K"

    def test_predict_under_50k(self, test_case_over_50k, test_app):

        response = test_app.post("/predict", data=test_case_over_50k.json(by_alias=True))

        assert response.status_code == 200
        assert response.json()["prediction"] == ">50K"