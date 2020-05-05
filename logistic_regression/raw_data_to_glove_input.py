"""
Gabriel Appleby
Working With Corpora
Logistic Regression

This file converts the shakespeare data to a file that glove can read.
"""

import os
from typing import List

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

from constants import SHAKESPEARE_TEXT_DIR, XML_PARSER

FULL_SHAKESPEARE_OUTPUT: str = "shakespeare_full_text.txt"


def main() -> None:
    """
    Takes all of the text from the Shakespeare data and throws it into a format that glove can make
    vectors from. Writes this as a file in this format to disk.
    :return: None.
    """
    files: List[str] = []
    for filename in os.listdir(SHAKESPEARE_TEXT_DIR):  # type: str
        file_path: str = os.path.join(SHAKESPEARE_TEXT_DIR, filename)
        with open(file_path, 'r') as tei_file:
            soup: BeautifulSoup = BeautifulSoup(tei_file, XML_PARSER)
            tokens: List[str] = word_tokenize(soup.text)
            lower_alpha_tokens: List[str] = [word.lower() for word in tokens if word.isalpha()]
            lower_alpha_text: str = ' '.join(lower_alpha_tokens)
            files.append(lower_alpha_text)

    full_text: str = ' '.join(files)
    with open(FULL_SHAKESPEARE_OUTPUT, "w") as f:
        f.write(full_text)


if __name__ == "__main__":
    main()
