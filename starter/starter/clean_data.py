import pandas as pd
import os
import logging
import hydra

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@hydra.main(config_path= '../../',config_name='params')
def main(cfg):
    config = cfg.clean_data

    df = pd.read_csv(os.path.join(
        config['input_dir'],
        config['input_csv']
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
            config['output_dir'],
            config['output_csv'],
        ),
        index = False
    )

if __name__ == '__main__':
    main()