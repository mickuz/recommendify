"""This script uses trained model to make final predictions."""

import os
import yaml
import pickle

import pandas as pd

from .data.data_fetcher import SpotifyDataFetcher
from .features.feature_engineering import (standardize_features,
                                           discretize_features, encode_features)


def main():
    with open('src/configs/config.yaml', mode='r') as config_file:
        conf = yaml.load(config_file, Loader=yaml.FullLoader)

    id = os.environ['CLIENT_ID']
    secret = os.environ['CLIENT_SECRET']

    username = conf['api']['username']
    scope = conf['api']['scope']
    uri = conf['api']['uri']

    drop_cols = conf['processing']['drop']['columns']
    std_cols = conf['processing']['standardization']['columns']
    discr_cols = conf['processing']['discretization']['columns']
    categories = conf['processing']['discretization']['categories']
    enc_cols = conf['processing']['encoding']['columns']

    to_predict = conf['playlists']['predictions']

    model_path = conf['model']['path']
    interim_train_path = conf['data']['interim-train']

    sdf = SpotifyDataFetcher(id, secret, username, scope, uri)
    df_pred = sdf.create_dataframe(username, to_predict)
    df_train = pd.read_csv(interim_train_path, index_col=0)

    df_pred.drop(drop_cols, axis=1, inplace=True)

    df_train, df_pred = standardize_features(df_train, df_pred, cols=std_cols)
    df_train, df_pred = discretize_features(df_train, df_pred, cols=discr_cols,
                                            cat_number=categories)
    df_train, df_pred = encode_features(df_train, df_pred, cols=enc_cols)

    with open(model_path, mode='rb') as model_file:
        classifier = pickle.load(model_file)

    y_pred = classifier.predict(df_pred).tolist()
    names = list(df_pred.index)

    for prediction, name in zip(y_pred, names):
        print('{}: {}'.format(name, prediction))


if __name__ == "__main__":
    main()
