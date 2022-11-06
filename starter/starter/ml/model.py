from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
#from starter.ml.data import process_data
from starter.ml.data import process_data
import logging
# Optional: implement hyperparameter tuning.
logger = logging.getLogger(__name__)

def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


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

        precision, recall, fbeta = compute_model_metrics(
            y_test, 
            clf.predict(X_test)
        )
        logger.info(f'precision score for {cat_value}: {precision}') 
        logger.info(f'recall score for {cat_value}: {recall}') 
        logger.info(f'fbeta score for {cat_value}: {fbeta}') 
