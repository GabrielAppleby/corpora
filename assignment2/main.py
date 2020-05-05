"""
Gabriel Appleby
Working With Corpora
Assignment 2
"""
from collections import defaultdict
from itertools import tee
from typing import List, Tuple, Callable, Dict, FrozenSet, Iterable, Iterator

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import gutenberg
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import StreamBackedCorpusView


def main() -> None:
    """
    Goes through all of the homework questions.
    :return: Nothing.
    """
    n_to_print: int = 10
    sense_and_sensibility: StreamBackedCorpusView = gutenberg.words('austen-sense.txt')
    colocations(sense_and_sensibility, n_to_print)

    words = [("car", "automobile"), ("gem", "jewel"), ("journey", "voyage"), ("boy", "lad"),
             ("coast", "shore"), ("asylum", "madhouse"), ("magician", "wizard"), ("midday", "noon"),
             ("furnace", "stove"), ("food", "fruit"), ("bird", "cock"), ("bird", "crane"),
             ("tool", "implement"), ("brother", "monk"), ("lad", "brother"), ("crane", "implement"),
             ("journey", "car"), ("monk", "oracle"), ("cemetery", "woodland"), ("food", "rooster"),
             ("coast", "hill"), ("forest", "graveyard"), ("shore", "woodland"), ("monk", "slave"),
             ("coast", "forest"), ("lad", "wizard"), ("chord", "smile"), ("glass", "magician"),
             ("rooster", "voyage"), ("noon", "string")]
    similarities(words)


def pairwise(iterable: Iterable) -> zip:
    """
    Creates two iterators from an iterable, and sets the second one to be one item ahead.
    IMPORTANT SIDE EFFECT: Do not use the original iterable after passing into this function.
    :param iterable: The iterable to create two iterators from. Cannot be None.
    :return: The two iterators.
    """
    a: Iterator
    b: Iterator
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def colocations(work: StreamBackedCorpusView, n_to_print: int) -> None:
    """
    Finds colocations in a work. Prints the n top ones.
    :param work: The work to find colocations in. Cannot be None.
    :param n_to_print: The number of top colocations to print. Must be >= 0.
    :return: Nothing.
    """
    list_of_words: List[str] = to_lower_remove_punct(work)
    colocation_dict: Dict[FrozenSet[str], int] = get_colocations(list_of_words)
    sorted_colocations: List[Tuple[FrozenSet[str], int]] = sort_colocations(colocation_dict)
    print_top_n_colocations(sorted_colocations, n_to_print)


def to_lower_remove_punct(work: StreamBackedCorpusView) -> List[str]:
    """
    Removes punctuation and puts the work in lower case.
    :param work: The work to do the work on. Cannot be None.
    :return: The work without punctuation and in lower case.
    """
    return [word.lower() for word in work if word.isalpha()]


def get_colocations(list_of_words: List[str]) -> Dict[FrozenSet[str], int]:
    """
    Gets the actual colocations in the list of words.
    :param list_of_words: The list of words to find colocations in. Cannot be None.
    :return: The dictionary mapping colocations to the number of appearances.
    """
    colocation_dict: Dict = defaultdict(int)
    for word_one, word_two in pairwise(list_of_words):
        colocation_dict[frozenset({word_one, word_two})] += 1
    return colocation_dict


def sort_colocations(
        colocation_dict: Dict[FrozenSet[str], int]) -> List[Tuple[FrozenSet[str], int]]:
    """
    Sorts the colocations from most to least.
    :param colocation_dict: The colocations to sort.
    :return: The sorted colocation list.
    """
    return sorted(colocation_dict.items(), key=lambda kv: kv[1], reverse=True)


def print_top_n_colocations(sorted_colocations: List[Tuple[FrozenSet[str], int]], n: int) -> None:
    """
    Prints the top n colocations of a sorted list of colocations.
    :param sorted_colocations: The sorted list of colocations to print.
    :param n: The number of colocations to print.
    :return: Nothing.
    """
    print("Top {} colocations: ".format(n))
    for entry in sorted_colocations[0:n]:  # type: Tuple[FrozenSet[str], int]
        print(entry)


def similarities(words: List[Tuple[str, str]]) -> None:
    """
    Looks at the similarity scores for all the word pairs in the given list.
    :param words: The list of word pairs to consider. Cannot be None.
    :return: Nothing. List sorted by different scores printed.
    """
    wn_sims: List[float] = get_similarities(words, path_similarity)
    wn_words_by_sim: List[Tuple[str, str]] = sort_words_by_similarity(words, wn_sims)

    # The glove vectors were downloaded from authors website, then converted to word2vec format
    # Using Gensim's glove2word2vec
    glove_model: KeyedVectors = KeyedVectors.load_word2vec_format(
        "gensim_glove_vectors.txt", binary=False)
    glove_sims: List[float] = get_similarities(words, glove_model.similarity)
    glove_words_by_sim: List[Tuple[str, str]] = sort_words_by_similarity(words, glove_sims)

    print("Similarities:")
    print("Wordnet")
    print(wn_words_by_sim)
    print("Glove")
    print(glove_words_by_sim)


def get_similarities(
        words: List[Tuple[str, str]], sim_functon: Callable[[str, str], float]) -> List[float]:
    """
    Gets the similarity between word pairs given a list of word pairs.
    :param words: The list of word pairs to find the similarity of. Cannot be None.
    :param sim_functon: The function to use in order to get the similarity of each pair. Cannot be
                        None.
    :return: The similarities for each word pair.
    """
    return [sim_functon(word_one, word_two) for word_one, word_two in words]


def sort_words_by_similarity(
        words: List[Tuple[str, str]], similarity_scores: List[float]) -> List[Tuple[str, str]]:
    """
    Sorts a list of word pairs by its similarities.
    :param words: The words to sort. Cannot be None.
    :param similarity_scores: The scores to sort the words by. Cannot be None.
    :return: The list of word pairs sorted by similarity.
    """
    return [x for _, x in reversed(sorted(zip(similarity_scores, words)))]


def path_similarity(word_one: str, word_two: str) -> float:
    """
    Computes the path similarity between two strings.
    :param word_one: The first word. Cannot be None, must be in wordnet.
    :param word_two: The second word. Cannot be None, must be in wordnet.
    :return: The similarity.
    """
    synsets_one: List[wn.Synset] = wn.synsets(word_one)
    synsets_two: List[wn.Synset] = wn.synsets(word_two)
    greatest_similarity: float = 0

    for synset_one in synsets_one:  # type: wn.Synset
        for synset_two in synsets_two:  # type: wn.Synset
            score = synset_one.path_similarity(synset_two)
            if score is not None and score > greatest_similarity:
                greatest_similarity = score

    return greatest_similarity


if __name__ == "__main__":
    main()
