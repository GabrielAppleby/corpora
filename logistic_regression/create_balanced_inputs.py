"""
Gabriel Appleby
Working With Corpora
Logistic Regression

This file takes the unbalanced data and creates a balanced csv and write all the balanced text
to a file for use with Glove.
"""

from typing import List

import pandas as pd
from nltk.tokenize import word_tokenize

from constants import UNBALANCED_DATA_BY_CHARACTER, SEX, BALANCED_DATA_BY_CHARACTER, TEXT

BALANCED_SHAKESPEARE_OUTPUT: str = "shakespeare_balanced_text.txt"
NUM_WORDS: int = 2000


def main() -> None:
    """
    Takes all of unbalanced data and creates a balanced dataset. Then writes it and balanced glove
    text to disk.
    :return: None.
    """
    data_by_character: pd.DataFrame = pd.read_csv(UNBALANCED_DATA_BY_CHARACTER, index_col=0)

    df_women = data_by_character[(data_by_character.Text.str.len() > NUM_WORDS) &
                                 (data_by_character[SEX] == 0)]
    df_men = data_by_character[(data_by_character.Text.str.len() > NUM_WORDS) &
                               (data_by_character[SEX] == 1)]

    df_men = df_men.sample(df_women.shape[0], random_state=2019)
    balanced_data_by_character = df_women.merge(df_men, how="outer")

    # Write balanced data frame to disk
    balanced_data_by_character.to_csv(BALANCED_DATA_BY_CHARACTER)

    # Create glove file
    texts = []
    for label, row in balanced_data_by_character.iterrows():
        tokens: List[str] = word_tokenize(row[TEXT])
        lower_alpha_tokens: List[str] = [word.lower() for word in tokens if word.isalpha()]
        lower_alpha_text: str = ' '.join(lower_alpha_tokens)
        texts.append(lower_alpha_text)

    full_text: str = ' '.join(texts)
    with open(BALANCED_SHAKESPEARE_OUTPUT, "w") as f:
        f.write(full_text)


if __name__ == "__main__":
    main()
