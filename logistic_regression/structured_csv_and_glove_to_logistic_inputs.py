"""
Gabriel Appleby
Working With Corpora
Logistic Regression

This file converts our structured csv and our glove vectors into standard matrices for logistic
regression. Should be set to work on balanced or unbalanced data.
"""

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constants import KEY, NAME, FILE, TEXT, UNBALANCED_DATA_BY_CHARACTER, \
    BALANCED_DATA_BY_CHARACTER, UNBALANCED_FINAL_DATA, BALANCED_FINAL_DATA

VECTOR: str = "Vector"
UNBALANCED_GLOVE_VECTORS = "vectors.txt"
BALANCED_GLOVE_VECTORS = "balanced_vectors.txt"

# SET ME
BALANCED = True


def main() -> None:
    """
    Reads in either the balanced or unbalanced dataframe and glove vectors. Then combines the
    dataframe and vectors into one matrix. Finally writes the matrix as Train / Test X and Y
    matrices in a npz.
    :return: None
    """
    glove_vectors = UNBALANCED_GLOVE_VECTORS
    data_by_character_file = UNBALANCED_DATA_BY_CHARACTER
    final_data = UNBALANCED_FINAL_DATA
    if BALANCED:
        glove_vectors = BALANCED_GLOVE_VECTORS
        data_by_character_file = BALANCED_DATA_BY_CHARACTER
        final_data = BALANCED_FINAL_DATA

    vector_representation_by_word: Dict[str, np.array] = {}
    with open(glove_vectors, "r") as f:
        for line in f.readlines():
            tokens = line.split()
            vector_representation_by_word[tokens[0]] = np.array(tokens[1:-1], dtype=np.float64)

    data_by_character: pd.DataFrame = pd.read_csv(data_by_character_file, index_col=0)

    vectors_by_character: Dict[str, np.array] = {}

    for label, row in data_by_character.iterrows():
        temp = np.array(0.0, dtype=np.float64)
        for idx, word in enumerate(row[TEXT]):
            if word in vector_representation_by_word:
                temp = np.add(temp, vector_representation_by_word[word])
        if idx != 0:
            vectors_by_character[label] = temp / idx

    vector_df: pd.DataFrame = pd.DataFrame(vectors_by_character).transpose()
    data_by_character.drop(columns=[NAME, KEY, FILE, TEXT], inplace=True)

    all_info: pd.DataFrame = data_by_character.merge(
        vector_df, how="inner", left_index=True, right_index=True)

    all_info = all_info.values
    X = np.array(all_info[:, 2:-1], dtype=np.float64)
    y = np.array(all_info[:, 0], dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=2019)
    np.savez(final_data, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


if __name__ == "__main__":
    main()
