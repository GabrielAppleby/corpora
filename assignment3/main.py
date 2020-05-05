"""
Gabriel Appleby
Working With Corpora
Assignment 3
"""
import re
from typing import List
from urllib import request

from bs4 import BeautifulSoup
from nltk import Index, PorterStemmer
from nltk.corpus import brown
from nltk.corpus import gutenberg, words
from nltk.corpus import wordnet as wn
from tei_reader import TeiReader


def main() -> None:
    """
    Goes through all of the homework questions.
    :return: None.
    """

    # Question 21.
    # Probably goes well with some beautiful soup..
    bread_url: str = \
        "http://www.thefreshloaf.com/node/23821/baguette-tradition-after-phillip-gosselin"
    unknown_words = unknown(bread_url)
    print(unknown_words)
    print("Wow, look at all that javascript nonsense. Total length: {}".format(len(unknown_words)))

    # Questions 22.
    print("Lets remove some words using another regex.")
    unknown_words_somewhat_cleaned = remove_some_javascript_from_bread_tokens(unknown_words)
    print(unknown_words_somewhat_cleaned)
    print("Somewhat better. Total length: {}.".format(len(unknown_words_somewhat_cleaned)))

    # Question 29
    words_lore = brown.words(categories='lore')
    sents_lore = brown.sents(categories='lore')
    score_lore = readability(words_lore, sents_lore)
    print("Readability score of the lore section: {}.".format(score_lore))

    words_learned = brown.words(categories='learned')
    sents_learned = brown.sents(categories='learned')
    score_learned = readability(words_learned, sents_learned)
    print("Readability score of the learned section: {}.".format(score_learned))

    words_news = brown.words(categories='news')
    sents_news = brown.sents(categories='news')
    score_news = readability(words_news, sents_news)
    print("Readability score of the news section: {}.".format(score_news))

    # Question 42 + xml thing
    reader = TeiReader()
    seneca = reader.read_file('seneca_1900.xml')
    indexed_text = IndexedText(PorterStemmer(), seneca.text)

    # This is what I used to train the Glove vectors.
    # write_nltk_gutenberg_to_file()


def unknown(url: str) -> List[str]:
    """
    Gets the unknown words on a web page given by the URL.
    :param url: The url of the webpage to look for unknown words in. Cannot be None.
    :return: The list of unknown words. Never None.
    """
    word_list = words.words()
    html = request.urlopen(url).read().decode('utf8')
    raw = BeautifulSoup(html, 'html.parser').get_text()
    tokens = re.findall(r"[a-z]+", raw)
    unknown_words = [token for token in tokens if token not in word_list]
    return unknown_words


def remove_some_javascript_from_bread_tokens(break_tokens: List[str]):
    """
    Removes javascript and css terms from a list of words. Is very specific to the list of words
    generated in the previous question.

    IMPORTANT:  Do not use with an arbitrary list.

    :param break_tokens: The list of word tokens generated about the delicious baguettes. Cannot be
    None.
    :return: The cleaned list of break tokens (some javascript and css terms removed).
    """
    pattern = re.compile(r"^[a | c| i | j  p | r | s | t| v ]*$")
    cleaned_bread_tokens = [thing for thing in break_tokens if not re.match(pattern, thing)]
    return cleaned_bread_tokens


def readability(words, sents) -> float:
    """
    Gets the readability of a work given its words and sentences.
    :param words: A list of the words of the work. Cannot be None.
    :param sents: A list of the sents of the work. Cannot be None.
    :return: The readability score.
    """
    m_w = average_len(words)
    m_s = average_len(sents)
    score = 4.71 * m_w + .05 * (m_s - 21.43)
    return score


def average_len(items) -> float:
    """
    Gets the average length of a list of things.
    :param items: The list of items. Cannot be None.
    :return: The average len.
    """
    return sum([len(item) for item in items]) / len(items)


def write_nltk_gutenberg_to_file() -> None:
    """
    Writes everything in nltk's gutenberg collection to one file..
    :return: None.
    """
    file_ids = gutenberg.fileids()
    f = open("nltk_gutenberg.txt", "a")
    fencepost_im_lazy = False
    for id in file_ids:
        # if fencepost_im_lazy:
        #     f.write("\n")
        fl = " ".join(gutenberg.words(id))
        f.write(fl)
        # fencepost_im_lazy = True


class IndexedText(object):

    def __init__(self, stemmer, text):
        self._text = text
        self._stemmer = stemmer
        self._index = Index((self._stem(word), i) for (i, word) in enumerate(text))
        semantic_list = []
        for i, word in enumerate(text):
            stemmed_word = self._stem(word)
            word_synsets = wn.synsets(stemmed_word)
            if word_synsets:
                syn = word_synsets[0]
                semantic_list.append((syn, syn.offset()))
        self._semantic_index = Index(semantic_list)

    def concordance(self, word, width=40):
        key = self._stem(word)
        wc = int(width / 4)  # words of context
        for i in self._index[key]:
            lcontext = ' '.join(self._text[i - wc:i])
            rcontext = ' '.join(self._text[i:i + wc])
            ldisplay = '{:>{width}}'.format(lcontext[-width:], width=width)
            rdisplay = '{:{width}}'.format(rcontext[:width], width=width)
            print(ldisplay, rdisplay)

    def _stem(self, word):
        return self._stemmer.stem(word).lower()


if __name__ == "__main__":
    main()
