import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DOWNLOAD_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
RED_URL = DOWNLOAD_ROOT + "winequality-red.csv"
WHITE_URL = DOWNLOAD_ROOT + "winequality-white.csv"
DATASETS_PATH = "./datasets"

def fetch_wine_data(red_url=RED_URL, white_url=WHITE_URL, datasets_path=DATASETS_PATH):
    if not os.path.isdir(datasets_path):
        os.mkdir(datasets_path)
    red_wine_path = os.path.join(datasets_path, "red_wine.csv")
    urllib.request.urlretrieve(red_url, red_wine_path)
    white_wine_path = os.path.join(datasets_path, "white_wine.csv")
    urllib.request.urlretrieve(white_url, white_wine_path)

def load_red_wine_data(datasets_path=DATASETS_PATH):
    red_csv_path = os.path.join(datasets_path, "red_wine.csv")
    return pd.read_csv(red_csv_path, sep=";")

def load_white_wine_data(datasets_path=DATASETS_PATH):
    white_csv_path = os.path.join(datasets_path, "white_wine.csv")
    return pd.read_csv(white_csv_path, sep=";")

def add_color_feature(red_df, white_df):
    red_df["color"] = 1
    white_df["color"] = 0

def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

def split_dataset(data, test_size=0.2, random_state=42):

    # randomly subdivide dataset, using stratified sampling to maintain color category 
    # proportions of the full dataset
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data["color"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    #split the quality labels off of both datasets
    train_labels = strat_train_set["quality"].copy()
    strat_train_set = strat_train_set.drop("quality", axis=1)
    test_labels = strat_test_set["quality"].copy()
    strat_test_set = strat_test_set.drop("quality", axis=1)

    #return split datasets and labels
    return strat_train_set, train_labels, strat_test_set, test_labels

class CustomAttributeTransformer (BaseEstimator, TransformerMixin):

    def __init__(self, remove_pH=False):
        self.remove_pH = remove_pH

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.values # convert dataframe to numpy array
        free_sulfur_dioxide_ix, total_sulfur_dioxide_ix = 5, 6
        free_sulfur_dioxide_percentage = X[:, free_sulfur_dioxide_ix] / X[:, total_sulfur_dioxide_ix]

        if self.remove_pH:
            # remove pH feature
            pH_ix = 8
            X = np.delete(X, pH_ix, 1)
            return np.c_[X, free_sulfur_dioxide_percentage]

        else: return np.c_[X, free_sulfur_dioxide_percentage]

def create_data_pipeline(data):

    pipeline = Pipeline([
        ("attrib_adder", CustomAttributeTransformer(remove_pH=True)),
        ("std_scaler", StandardScaler()),
    ])

    pipeline.fit(data)
    return pipeline

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

