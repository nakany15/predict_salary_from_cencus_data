#!/usr/bin/env python3
"""Test script that invokes the deployed Heroku instance of the FastAPI for census """
import requests
from typing import Dict


def example_below(base_url: str):
    body = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    present_prediction(base_url, body)


def example_above(base_url):
    body = {
        "age": 31, 
        "workclass": "Private", 
        "fnlgt": 45781, 
        "education": "Masters", 
        "education_num": 14,
        "marital_status": "Never-married", 
        "occupation": "Prof-specialty", 
        "relationship": "Not-in-family",
        "race": "White", 
        "sex": "Female", 
        "capital_gain": 14084, 
        "capital_loss": 0, 
        "hours_per_week": 50,
        "native_country": "United-States"
        }

    present_prediction(base_url, body)


def present_prediction(base_url, body):
    response = make_prediction(f"{base_url}/predict", body)
    print(f"Expect person with {body['education']} at age {body['age']} to earn {response['prediction']}")


def make_prediction(url: str, body: Dict):
    response = requests.post(url, json=body)
    print(f"Response status code: {response.status_code}")
    return response.json()


if __name__ == '__main__':
    base_url = "https://nakany-income-prediction.herokuapp.com"
    example_below(base_url)
    example_above(base_url)