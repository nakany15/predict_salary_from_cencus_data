#!/usr/bin/env python3
"""Test script that invokes the deployed Heroku instance of the FastAPI for census """
import requests
from typing import Dict


def example_1(base_url: str):
    body = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 76532,
        "education": "Bachelors",
        "education-num": 12,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2222,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    post_prediction(base_url, body)


def example_2(base_url):
    body = {
        "age": 28, 
        "workclass": "Private", 
        "fnlgt": 54312, 
        "education": "Masters", 
        "education-num": 14,
        "marital-status": "Never-married", 
        "occupation": "Prof-specialty", 
        "relationship": "Not-in-family",
        "race": "White", 
        "sex": "Female", 
        "capital-gain": 14520, 
        "capital-loss": 2, 
        "hours-per-week": 48,
        "native-country": "United-States"
        }

    post_prediction(base_url, body)


def post_prediction(base_url, body):
    response = requests.post(f"{base_url}/predict", json=body)
    print(f"Response status code: {response.status_code}")
    return_json = response.json()
    print(f"Expect person with {body['education']} at age {body['age']} to earn {return_json['prediction']}")


if __name__ == '__main__':
    base_url = "https://nakany-income-prediction.herokuapp.com"
    example_1(base_url)
    example_2(base_url)