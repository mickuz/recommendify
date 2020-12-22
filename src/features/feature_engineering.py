import re
import pandas as pd

from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   KBinsDiscretizer)


def standardize_features(df_train, df_test, cols=[]):
    df_train = df_train.copy()
    df_test = df_test.copy()

    scaler = StandardScaler()

    df_train[cols] = scaler.fit_transform(df_train[cols])
    df_test[cols] = scaler.transform(df_test[cols])

    return df_train, df_test


def discretize_features(df_train, df_test, cols=[], cat_number=5):
    df_train = df_train.copy()
    df_test = df_test.copy()

    discretizer = KBinsDiscretizer(strategy='uniform',
                                   n_bins=cat_number,
                                   encode='ordinal')

    df_train[cols] = discretizer.fit_transform(df_train[cols]).astype(int)
    df_test[cols] = discretizer.transform(df_test[cols]).astype(int)

    return df_train, df_test


def encode_features(df_train, df_test, cols=[]):
    encoder = OneHotEncoder(handle_unknown='ignore',
                            sparse=False)

    def add_dummy_vars(df, encoder, transformed_features):
        df = df.copy()

        df_transformed = pd.DataFrame(transformed_features,
                                      columns=encoder.get_feature_names(cols),
                                      index=df.index)

        df = pd.concat([df, df_transformed], axis=1)

        df.drop(cols, axis=1, inplace=True)
        for col in cols:
            regex = '{}.*_\d*$'.format(col)
            pattern = re.compile(regex)

            selected_cols = [col for col in list(df.columns)
                             if pattern.match(col)]

            df.drop(selected_cols[0], axis=1, inplace=True)

        return df

    train_transformed = encoder.fit_transform(df_train[cols]).astype(int)
    df_train = add_dummy_vars(df_train, encoder, train_transformed)

    test_transformed = encoder.transform(df_test[cols]).astype(int)
    df_test = add_dummy_vars(df_test, encoder, test_transformed)

    return df_train, df_test
