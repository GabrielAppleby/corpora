"""
Gabriel Appleby
Working With Corpora
Assignment 1
"""
from nltk.corpus import brown, gutenberg
from nltk.corpus.reader import ConcatenatedCorpusView, StreamBackedCorpusView
from nltk.text import Text


def main() -> None:
    """
    Goes through all of the homework questions.
    :return: Nothing.
    """
    question_five()
    question_six()
    question_twenty_seven()
    question_twenty_eight()


def question_five() -> None:
    """
    Answers question 5.
    Compare the lexical diversity scores for humor and romance fiction in brown corpus. Which genre
    is more lexically diverse?
    :return: None
    """
    humor_words: ConcatenatedCorpusView = brown.words(categories="humor")
    romance_words: ConcatenatedCorpusView = brown.words(categories="romance")

    humor_ld: float = lexical_diversity(humor_words)
    romance_ld: float = lexical_diversity(romance_words)

    print("Humor lexical diversity: {:.2}".format(humor_ld))
    print("Romance lexical diversity: {:.2}".format(romance_ld))
    print("Humor is more lexically diverse")


def question_six() -> None:
    """
    Answers question 6.
    Produce a dispersion plot of the four main protagonists in Sense and Sensibility: Elinor,
    Marianne, Edward, and Willoughby. What can you observe about the different roles played by the
    males and females in this novel? Can you identify the couples?
    :return: Nothing
    """
    sense_and_sensibility: Text = Text(gutenberg.words('austen-sense.txt'))
    sense_and_sensibility.dispersion_plot(["Elinor", "Marianne", "Edward", "Willoughby"])
    print("Women play a larger role in the novel than men")
    print("Elinor and Edward are a couple, and Marianne and Willoughby are a couple")


def question_twenty_seven() -> None:
    """
    Answers question 27.
    Define a function called vocab_size(text) that has a single parameter for the text, and which
    returns the vocabulary size of the text.
    :return: Nothing
    """

    def vocab_size(text: StreamBackedCorpusView) -> int:
        """
        Gets the vocab size of a text.
        :param text: The text to get the vocab size of.
        :return: The vocab size.
        """
        return len(set(text))

    emma_vs = vocab_size(gutenberg.words('austen-emma.txt'))
    print("Emma vocab size: {}".format(emma_vs))


def question_twenty_eight() -> None:
    """
    Answers question 28.
    Define a function percent(word, text) that calculates how often a given word occurs in a text,
    and expresses the result as a percentage.
    :return: Nothing.
    """

    def percent(word: str, text: StreamBackedCorpusView) -> float:
        """
        Calculates how often a given word occurs in a text.
        :param word: The world to calculate the percent occurrence for.
        :param text: The text to use when calculating how often the word should appear.
        :return: The percent occurrence of the word in the text.
        """
        return 100 * text.count(word) / len(text)

    persuasion_percent = percent("officer", gutenberg.words('austen-persuasion.txt'))
    print("Persuasion percent occurrence of the word officer: {}"
          .format(persuasion_percent))


def lexical_diversity(text: ConcatenatedCorpusView) -> float:
    """
    Calculates the lexical diversity of a text.
    :param text: The text to calculate the lexical diversity of.
    :return: The lexical diversity.
    """
    return len(set(text)) / len(text)


if __name__ == "__main__":
    main()
