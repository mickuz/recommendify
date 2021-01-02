import pytest
import pandas as pd

from src.features.feature_engineering import (standardize_features,
                                              discretize_features,
                                              encode_features)


@pytest.fixture(scope='function')
def dataframes():
    """Return a tuple of pandas DataFrame objects. The first element of the
    tuple is considered as a subset of a dataset intended for training and the
    second one - for testing.
    """
    in_train = pd.DataFrame({'col1': [1, 4, 3, 7, 2],
                             'col2': ['a', 'a', 'b', 'c', 'c']})
    in_test = pd.DataFrame({'col1': [4, 5, 3, 4, 10],
                            'col2': ['c', 'b', 'b', 'a', 'b']})

    return in_train, in_test


def test_standardization(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    in_train_mean = in_train['col1'].mean()
    in_train_std = in_train['col1'].std(ddof=0)

    out_train, out_test = standardize_features(in_train,
                                               in_test,
                                               cols=['col1'])

    expected_col1_train = (in_train['col1'] - in_train_mean) / in_train_std
    expected_col1_test = (in_test['col1'] - in_train_mean) / in_train_std

    expected_train = pd.DataFrame({'col1': expected_col1_train,
                                   'col2': in_train['col2']})
    expected_test = pd.DataFrame({'col1': expected_col1_test,
                                  'col2': in_test['col2']})

    assert out_train.equals(expected_train) and out_test.equals(expected_test)


def test_if_mean_zero(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    out_train, out_test = standardize_features(in_train,
                                               in_test,
                                               cols=['col1'])

    assert pytest.approx(out_train['col1'].mean()) == 0


def test_if_std_one(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    out_train, out_test = standardize_features(in_train,
                                               in_test,
                                               cols=['col1'])

    assert pytest.approx(out_train['col1'].std(ddof=0)) == 1


@pytest.mark.parametrize('function', [
    standardize_features,
    discretize_features,
    encode_features
])
def test_default_no_col(dataframes, function):
    in_train = dataframes[0]
    in_test = dataframes[1]

    with pytest.raises(ValueError):
        function(in_train, in_test)


@pytest.mark.parametrize('function', [
    standardize_features,
    discretize_features
])
def test_text_col(dataframes, function):
    in_train = dataframes[0]
    in_test = dataframes[1]

    with pytest.raises(ValueError):
        function(in_train, in_test, cols=['col2'])


def test_one_bin(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    with pytest.raises(ValueError):
        discretize_features(in_train, in_test, ['col1'], cat_number=1)


def test_discretization(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    out_train, out_test = discretize_features(in_train,
                                              in_test,
                                              cols=['col1'],
                                              cat_number=3)

    expected_train = pd.DataFrame({'col1': [0, 1, 1, 2, 0],
                                   'col2': in_train['col2']})
    expected_test = pd.DataFrame({'col1': [1, 2, 1, 1, 2],
                                  'col2': in_test['col2']})

    assert out_train.equals(expected_train) and out_test.equals(expected_test)


def test_encoding(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    out_train, out_test = encode_features(in_train,
                                          in_test,
                                          cols=['col1', 'col2'])

    expected_train = pd.DataFrame({'col1_2': [0, 0, 0, 0, 1],
                                   'col1_3': [0, 0, 1, 0, 0],
                                   'col1_4': [0, 1, 0, 0, 0],
                                   'col1_7': [0, 0, 0, 1, 0],
                                   'col2_b': [0, 0, 1, 0, 0],
                                   'col2_c': [0, 0, 0, 1, 1]})

    expected_test = pd.DataFrame({'col1_2': [0, 0, 0, 0, 0],
                                  'col1_3': [0, 0, 1, 0, 0],
                                  'col1_4': [1, 0, 0, 1, 0],
                                  'col1_7': [0, 0, 0, 0, 0],
                                  'col2_b': [0, 1, 1, 0, 1],
                                  'col2_c': [1, 0, 0, 0, 0]})

    assert out_train.equals(expected_train) and out_test.equals(expected_test)


def test_cols_number(dataframes):
    in_train = dataframes[0]
    in_test = dataframes[1]

    out_train, out_test = encode_features(in_train, in_test, ['col1'])
    out_cols = len(out_train.columns), len(out_test.columns)

    assert out_cols == (5, 5)
