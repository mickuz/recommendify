"""This script generates processed data for model training."""

import yaml
import pandas as pd

from .feature_engineering import (standardize_features, discretize_features,
                                  encode_features)

from sklearn.model_selection import train_test_split


def main():
    with open('src/configs/config.yaml', mode='r') as config_file:
        conf = yaml.load(config_file, Loader=yaml.FullLoader)

    raw_path = conf['data']['raw-dataset']
    train_path = conf['data']['processed-train']
    test_path = conf['data']['processed-test']
    interim_train_path = conf['data']['interim-train']
    interim_test_path = conf['data']['interim-test']

    seed = conf['random-seed']
    size = conf['processing']['split']['test-size']

    drop_cols = conf['processing']['drop']['columns']
    std_cols = conf['processing']['standardization']['columns']
    discr_cols = conf['processing']['discretization']['columns']
    categories = conf['processing']['discretization']['categories']
    enc_cols = conf['processing']['encoding']['columns']

    df = pd.read_csv(raw_path, index_col=0)

    df.drop(drop_cols, axis=1, inplace=True)
    df_train, df_test = train_test_split(df, test_size=size, random_state=seed)

    df_train.to_csv(interim_train_path, index=True)
    df_test.to_csv(interim_test_path, index=True)

    df_train, df_test = standardize_features(df_train, df_test, cols=std_cols)
    df_train, df_test = discretize_features(df_train, df_test, cols=discr_cols,
                                            cat_number=categories)
    df_train, df_test = encode_features(df_train, df_test, cols=enc_cols)

    df_train.to_pickle(train_path)
    df_test.to_pickle(test_path)


if __name__ == '__main__':
    main()
