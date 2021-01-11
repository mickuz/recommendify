"""This script sets up training of the model."""

import yaml
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier


def main():
    with open('src/configs/config.yaml', mode='r') as config_file:
        conf = yaml.load(config_file, Loader=yaml.FullLoader)

    seed = conf['random-seed']

    dataset_path = conf['data']['processed-train']
    model_path = conf['model']['path']

    df = pd.read_pickle(dataset_path)
    X_train, y_train = df.drop(['if_liked'], axis=1), df['if_liked']

    classifier = RandomForestClassifier(n_estimators=700,
                                        criterion='entropy',
                                        max_depth=10,
                                        min_samples_leaf=3,
                                        min_samples_split=10,
                                        random_state=seed)
    classifier.fit(X_train, y_train)

    with open(model_path, mode='wb') as model_file:
        pickle.dump(classifier, model_file)


if __name__ == "__main__":
    main()
