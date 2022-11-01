# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import pickle

# Add code to load in the data.
data = pd.read_csv('starter/data/processed/cencus_cleaned.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


# Train and save a model.
model = train_model(X_train, y_train)
with open('starter/model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

def slice_performance(df, category, label, cat_features, encoder, lb, clf):
    cat_value_list = df[category].unique()
    for cat_value in cat_value_list:
        df2 = df.loc[df[category] == cat_value]
        X_test, y_test, encoder, lb = process_data(
            df2, 
            label = label,
            categorical_features=cat_features, 
            training=False,
            encoder=encoder,
            lb = lb
        )

        print(
            cat_value, 
            compute_model_metrics(
                y_test, 
                clf.predict(X_test)
            )
        )

slice_performance(
    data, 
    'race', 
    'salary', 
    cat_features, 
    encoder, 
    lb, 
    model)