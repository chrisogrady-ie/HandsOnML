import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def main():
    housing = pd.read_csv("datasets/housing.csv", encoding="ISO-8859-1")
    #housing.info()

    #print(housing.describe())

    #housing.hist(bins=50, figsize=(20,15))
    #plt.show()

    #train_set, test_set = split_train_test(housing, 0.2)

    #train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    #print(len(train_set), len(test_set))

    housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    housing["income_cat"].hist()
    #plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))



main()
#page 86