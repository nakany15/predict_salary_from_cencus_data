import pandas as pd
import os
import logging
import hydra

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@hydra.main(config_path= '../../',config_name='params')
def clean_data(cfg):
    print(os.getcwd())

    df = pd.read_csv(os.path.join(
        cfg.clean_data['data_dir'],
        cfg.clean_data['input_csv']
    ))

    #remove leading spaces in column names
    df.columns = [col.strip() for col in df.columns]

    # column name list of non-numeric features
    cat_cols = df.select_dtypes(include='object').columns

    # remove leading spaces in the non-numeric features
    for col in cat_cols:
        df[col] = df[col].str.strip()

    df.to_csv(
        os.path.join(
            cfg.clean_data['output_csv'],
        ),
        index = False
    )

if __name__ == '__main__':
    clean_data()