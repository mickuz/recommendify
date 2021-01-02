"""This module implements feature engineering techniques."""

import re
import pandas as pd

from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   KBinsDiscretizer)


def standardize_features(df_train, df_test, cols=[]):
    """Scale continuous features to unit variance and zero mean.

    Parameters
    ----------
    df_train : pandas.core.frame.DataFrame
        A subset of data intended for training.
    df_test : pandas.core.frame.DataFrame
        A subset of data intended for testing.
    cols : list of [str]
        Names of features to transform.

    Returns
    -------
    tuple of (pandas.core.frame.DataFrame)
        Transformed subsets of dataset.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    scaler = StandardScaler()

    df_train[cols] = scaler.fit_transform(df_train[cols])
    df_test[cols] = scaler.transform(df_test[cols])

    return df_train, df_test


def discretize_features(df_train, df_test, cols=[], cat_number=5):
    """Produce categories in form of integer numbers out of continuous features.

    Parameters
    ----------
    df_train : pandas.core.frame.DataFrame
        A subset of data intended for training.
    df_test : pandas.core.frame.DataFrame
        A subset of data intended for testing.
    cols : list of [str]
        Names of features to transform.
    cat_number : int
        Number of categories to produce.

    Returns
    -------
    tuple of (pandas.core.frame.DataFrame)
        Transformed subsets of dataset.
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    discretizer = KBinsDiscretizer(strategy='uniform',
                                   n_bins=cat_number,
                                   encode='ordinal')

    df_train[cols] = discretizer.fit_transform(df_train[cols]).astype(int)
    df_test[cols] = discretizer.transform(df_test[cols]).astype(int)

    return df_train, df_test


def encode_features(df_train, df_test, cols=[]):
    """Produce one-hot numeric features out of categorical or ordinal features.

    Parameters
    ----------
    df_train : pandas.core.frame.DataFrame
        A subset of data intended for training.
    df_test : pandas.core.frame.DataFrame
        A subset of data intended for testing.
    cols : list of [str]
        Names of features to transform.

    Returns
    -------
    tuple of (pandas.core.frame.DataFrame)
        Transformed subsets of dataset.
    """
    if not cols:
        raise ValueError('No features to transform.')

    encoder = OneHotEncoder(handle_unknown='ignore',
                            sparse=False)

    def add_dummy_vars(df, encoder, transformed_features):
        """Transform sparse matrix into data frame and drop redundant data."""
        df = df.copy()

        df_transformed = pd.DataFrame(transformed_features,
                                      columns=encoder.get_feature_names(cols),
                                      index=df.index)

        df = pd.concat([df, df_transformed], axis=1)

        df.drop(cols, axis=1, inplace=True)
        for col in cols:
            regex = '{}.*_\w*$'.format(col)
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
