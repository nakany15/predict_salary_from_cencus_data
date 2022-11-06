# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
import pickle
import logging
import hydra
import os
import json

# Add code to load in the data.
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def slice_test_performance(test, X_test, y_test, slice_cat_list, model):
    slice_scores = {}
    for slice in slice_cat_list:
        slice_scores[slice] = {}
        cat_value_list = test[slice].unique()
        for cat_value in cat_value_list:
            slice_scores[slice][cat_value] = {}
            test_sliced_idx = test.loc[test[slice] == cat_value].index
            X_test_sliced = X_test[test_sliced_idx]
            y_test_sliced = y_test[test_sliced_idx]
            precision, recall, fbeta = compute_model_metrics(
                y_test_sliced, 
                model.predict(X_test_sliced)
            )
            slice_scores[slice][cat_value]
            slice_scores[slice][cat_value]['precision'] = precision
            slice_scores[slice][cat_value]['recall'] = recall
            slice_scores[slice][cat_value]['fbeta'] = fbeta
            logger.info(f'precision score for {cat_value}: {precision}') 
            logger.info(f'recall score for {cat_value}: {recall}') 
            logger.info(f'fbeta score for {cat_value}: {fbeta}') 
    return slice_scores


@hydra.main(config_path= '../../',config_name='params')
def main(cfg):
    config = cfg.model
    logger.info(f"current directory: {os.getcwd()}")
    cat_features = config['categorical_features']
    target = config['target']
    slice_list = config['slices']
    model_file_path = os.path.join(
        config['output_dir'], 
        config['output_model']
    )
    encoder_file_path = os.path.join(
        config['output_dir'], 
        config['output_encoder']
    )
    lb_file_path = os.path.join(
        config['output_dir'], 
        config['output_lb']
    )
    slice_file_path = os.path.join(
        config['output_dir'], 
        config['output_slice']
    )

    data = pd.read_csv(os.path.join(
        config['input_dir'],
        config['input_csv']
    ))

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(
        data, 
        test_size=config['test_size'],
        random_state = config['random_state']
    )

    # Proces the test data with the process_data function.

    X_train, y_train, encoder, lb = process_data(
        train, 
        categorical_features=cat_features, 
        label = target, 
        training = True
    )
    test = test.reset_index().drop(columns = ['index'])
    X_test, y_test, encoder, lb = process_data(
        test, 
        label = target,
        categorical_features=cat_features, 
        training=False,
        encoder=encoder,
        lb = lb
    )
    # Train and save a model.
    model = train_model(X_train, y_train)
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

    with open(encoder_file_path, 'wb') as f:
        pickle.dump(encoder, f)
    
    with open(lb_file_path, 'wb') as f:
        pickle.dump(lb, f)

    precision, recall, fbeta = compute_model_metrics(
        y_test,
        model.predict(X_test)
    )
    logger.info(f'precision score : {precision}') 
    logger.info(f'recall score : {recall}') 
    logger.info(f'fbeta score : {fbeta}') 
    slice_scores = slice_test_performance(test, X_test, y_test, slice_list, model)
    logger.info(f'{slice_scores}')
    with open(slice_file_path, mode="w") as f:
        json.dump(slice_scores,f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()