from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer


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
        df_copy[cols] = discretizer.fit_transform(df[cols])
    else:
        df_copy[cols] = discretizer.transform(df[cols])

    return df_copy


def encode_features(df, cols=[], if_train=True):
    df_copy = df.copy()

    encoder = OneHotEncoder(drop='first',
                            handle_unknown='ignore')
    if if_train:
        df_copy[cols] = encoder.fit_transform(df[cols])
    else:
        df_copy[cols] = encoder.transform(df[cols])
