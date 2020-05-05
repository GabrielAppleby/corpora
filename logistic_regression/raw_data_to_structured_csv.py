"""
Gabriel Appleby
Working With Corpora
Logistic Regression

This file converts our various inputs to a single CSV file.
"""

import os
from collections import defaultdict
from typing import List, DefaultDict

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

from constants import CHARACTER_CSV, SHAKESPEARE_TEXT_DIR, XML_PARSER
from constants import KEY, NAME, FILE, TEXT, UNBALANCED_DATA_BY_CHARACTER, SEX

TO_DROP_ONE: str = "To_drop_one"
TO_DROP_TWO: str = "To_drop_two"
MALE: str = 'male'
FEMALE: str = 'female'
CLUDGE: str = "Cludge"

DF_HEADER: List[str] = [KEY, NAME, FILE, TO_DROP_ONE, SEX, TO_DROP_TWO]
COLUMNS_TO_DROP: List[str] = [TO_DROP_ONE, TO_DROP_TWO]

SPEAKER_TAG: str = "sp"
LINE_TAG: str = "l"
WHO_ATTRIBUTE: str = "who"


def main() -> None:
    """
    Uses the characters csv and shakespeare text to output a csv ready to use.
    :return: None.
    """
    df: pd.DataFrame = pd.read_csv(CHARACTER_CSV, names=DF_HEADER, index_col=None)
    df.drop(columns=COLUMNS_TO_DROP, inplace=True)
    df[SEX] = df[SEX].str.lower()

    text_by_name: DefaultDict[str] = defaultdict(str)
    for filename in os.listdir(SHAKESPEARE_TEXT_DIR):  # type: str
        file_path: str = os.path.join(SHAKESPEARE_TEXT_DIR, filename)
        with open(file_path, 'r') as tei_file:
            soup: BeautifulSoup = BeautifulSoup(tei_file, XML_PARSER)
            all_speaker_tags: List[Tag] = soup.findAll(SPEAKER_TAG)
            for speaker_tag in all_speaker_tags:  # type: Tag
                text: str = ' '
                for line_tag in speaker_tag.findAll(LINE_TAG):  # type: Tag
                    text += line_tag.text
                if WHO_ATTRIBUTE in speaker_tag.attrs:
                    text_by_name[file_path + '-' + speaker_tag.attrs[WHO_ATTRIBUTE]] += text

    # Add our newly found text as a column based on name key
    text: pd.DataFrame = pd.DataFrame.from_dict(text_by_name, orient="index", columns=[TEXT])
    df[CLUDGE] = df[FILE] + '-' + df[KEY]
    df = df.merge(text, how="inner", right_index=True, left_on=CLUDGE)

    # Drop rows with NA vals
    df.dropna(inplace=True)
    # Drop rows with no text
    idx_rows_no_text = df[df[TEXT] == ''].index
    df.drop(idx_rows_no_text, inplace=True)

    # Drop columns with bad sex data
    idx_rows_no_sex = df[(df[SEX] != MALE) & (df[SEX] != FEMALE)].index
    df.drop(idx_rows_no_sex, inplace=True)

    df[SEX].replace(MALE, 1, inplace=True)
    df[SEX].replace(FEMALE, 0, inplace=True)

    df.to_csv(UNBALANCED_DATA_BY_CHARACTER)


if __name__ == "__main__":
    main()
