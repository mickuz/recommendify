import re
import pandas as pd

from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   KBinsDiscretizer)


def standardize_features(df, cols=[], if_train=True):
    df_copy = df.copy()

    scaler = StandardScaler()

    if if_train:
        df_copy[cols] = scaler.fit_transform(df[cols])
    else:
        df_copy[cols] = scaler.transform(df[cols])

    return df_copy


def discretize_features(df, cols=[], if_train=True, cat_number=5):
    df_copy = df.copy()

    discretizer = KBinsDiscretizer(strategy='uniform',
                                   n_bins=cat_number,
                                   encode='ordinal')

    if if_train:
        df_copy[cols] = discretizer.fit_transform(df[cols]).astype(int)
    else:
        df_copy[cols] = discretizer.transform(df[cols]).astype(int)

    return df_copy


def encode_features(df, cols=[], if_train=True):
    encoder = OneHotEncoder(handle_unknown='ignore',
                            sparse=False)

    def add_dummy_vars(encoder, transformed_features):
        df_copy = df.copy()

        df_transformed = pd.DataFrame(transformed_features,
                                      columns=encoder.get_feature_names(cols),
                                      index=df_copy.index)

        df_copy = pd.concat([df_copy, df_transformed], axis=1)

        df_copy.drop(cols, axis=1, inplace=True)
        for col in cols:
            regex = '{}.*_\d*$'.format(col)
            pattern = re.compile(regex)

            selected_cols = [col for col in list(df_copy.columns)
                             if pattern.match(col)]

            df_copy.drop(selected_cols[0], axis=1, inplace=True)

        return df_copy

    if if_train:
        transformed = encoder.fit_transform(df[cols]).astype(int)
        df = add_dummy_vars(encoder, transformed)
    else:
        transformed = encoder.transform(df[cols]).astype(int)
        df = add_dummy_vars(encoder, transformed)

    return df
