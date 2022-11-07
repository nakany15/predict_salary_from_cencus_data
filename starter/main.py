# Put the code for your API here.
import os
import logging
import pandas as pd
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
import pickle
import os

from starter.ml.data import process_data
from starter.ml.model import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

    
app = FastAPI(title="Income Prediction 🤖",
              description="Machine Learning model trained on the [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)")
with open('./starter/starter/model/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./starter/starter/model/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('./starter/starter/model/lb.pkl', 'rb') as f:
    lb = pickle.load(f)


def get_prediction(param1, param2):
    x = [[param1, param2]]
    y = model.predict(x)[0]  # just get single value
    prob = model.predict_proba(x)[0].tolist()  # send to list for return

    return {'prediction': int(y), 'probability': prob}


class ModelParams(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


@app.get("/")
def greet():
    return {"message": "This is an Income prediction API!!"}
@app.post("/predict")
def predict(params: ModelParams):
    df_input = pd.DataFrame.from_dict([params.dict(by_alias=True)])

    cat_features = [
        'workclass', 
        'education', 
        'marital-status', 
        'occupation',
        'relationship', 
        'race', 
        'sex', 
        'native-country']
    X, _, _, _ = process_data(
        df_input, 
        cat_features, 
        None, 
        training=False, 
        encoder=encoder, 
        lb=lb)
    pred = inference(model, X)
    return {"prediction": "<=50K" if pred <= 0.5 else ">50K"}