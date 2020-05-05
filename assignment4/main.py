"""
Gabriel Appleby
Working With Corpora
Assignment 4
"""
import os
import pickle

from nltk.corpus import brown
from collections import defaultdict

from operator import lt, ge


def main():
    """
    Goes through all of the homework questions.
    :return: None.
    """
    all_brown_tagged_words, universal_brown_tagged, words_to_tags, tags_to_words = get_pickles()

    print("Question 18:")
    # Get unambiguous words
    unambiguous_words = extract_words_by_ambiguity(words_to_tags.keys(), words_to_tags, lt)
    # Percentage of unambiguous word types to all types.
    percentage_of_unambiguous_to_all_types = percentage_of_total(
        words_to_tags.keys(), unambiguous_words)
    print("Percentage of word types assigned same part-of-speech tag: {}"
          .format(percentage_of_unambiguous_to_all_types))

    # Get ambiguous words
    ambiguous_words = extract_words_by_ambiguity(words_to_tags.keys(), words_to_tags, ge)
    # Number of ambiguous word
    num_ambiguous_words = len(ambiguous_words)
    print("The number of words that appear with at least two tags: {}"
          .format(num_ambiguous_words))

    all_brown_words = [word for word, tag in all_brown_tagged_words]
    # Get ambiguous tokens
    ambiguous_tokens = extract_words_by_ambiguity(all_brown_words, words_to_tags, ge)
    # Percentage of ambiguous word types to all types.
    percentage_of_ambiguous_to_all_tokens = percentage_of_total(all_brown_words, ambiguous_tokens)
    print("Percentage of word tokens that involve these ambiguous words: {}"
          .format(percentage_of_ambiguous_to_all_tokens))
    print()

    print("Question 19:")
    print("It must remove the correct labels to create an input for the tagger.")
    print("For each tag it could compare the predicted value with the actual value, "
          "keeping track of the correct results." "Finally, just subtract divide by "
          "the total for the accuracy.")
    print()

    print("Question 20:")
    print("Sorted list of distinct words tagged as MD:")
    sorted_md_words = sorted(set(word.lower() for word in tags_to_words["MD"]))
    print(sorted_md_words)
    print("Words that can be plural nouns or third person singular verbs:")
    vbz_words = set(sorted(word.lower() for word in tags_to_words["VBZ"]))
    nns_words = set(sorted(word.lower() for word in tags_to_words["NNS"]))
    vbz_and_nns_words = vbz_words.intersection(nns_words)
    # Plural noun = NNS, Third person singular verb = VBZ
    print(vbz_and_nns_words)
    print("Words of form IN + DET + NN:")
    print("Actually words of ADP + DET + NOUN because "
          "universal tagger seemed to work better for this.")
    occurances = []
    current_occurance = []
    for word, tag in universal_brown_tagged:
        if tag == "ADP" and len(current_occurance) == 0:
            current_occurance.append(word)
        elif tag == "DET" and len(current_occurance) == 1:
            current_occurance.append(word)
        elif tag == "NOUN" and len(current_occurance) == 2:
            current_occurance.append(word)
            occurances.append(current_occurance)
            current_occurance = []
        else:
            current_occurance.clear()
    print(occurances)
    print("Ratio of masculine to feminine pronouns:")
    hes = 0
    shes = 0
    for word in all_brown_words:
        word = word.lower()
        if word == "he":
            hes += 1
        elif word == "she":
            shes += 1
    print(hes / shes)


def get_mappings(tagged_words):
    """
    Builds a mapping from words to POS tags.
    :param tagged_words: The tagged words.
    :return: The mapping.
    """
    words_to_tags = defaultdict(list)
    tags_to_words = defaultdict(list)
    for word, tag in tagged_words:
        if tag not in words_to_tags[word]:
            words_to_tags[word].append(tag)
        if word not in tags_to_words[tag]:
            tags_to_words[tag].append(word)
    return words_to_tags, tags_to_words


def extract_words_by_ambiguity(keys, mapping, op):
    """
    Extracts the given keys from the mapping based on whether or not they are ambiguous.
    The op controls whether we are looking for ambiguous, or unambiguous words.
    :param keys: The keys to look for.
    :param mapping: The mapping to look through.
    :param op: The op to use, lt for unambiguous, ge for ambiguous.
    :return: The extracted words.
    """
    extracted_words = []
    for key in keys:
        if op(len(mapping[key]), 2):
            extracted_words.append(key)
    return extracted_words


def percentage_of_total(keys, portion):
    """
    Gets the percentage of the keys the portion is.
    :param keys: The keys are the total.
    :param portion: The portion of the keys.
    :return: The percentage.
    """
    return len(portion) / len(keys)


def get_pickles():
    words_file_name = 'tagged_words.pkl'
    universal_words_file_name = 'universal_tagged_words.pkl'
    all_brown_tagged_words = []
    universal_all_brown_tagged_words = []
    words_to_tags, tags_to_words = {}, {}
    if os.path.exists(words_file_name):
        all_brown_tagged_words = pickle.load(open(words_file_name, "rb"))
    else:
        print("Assembling corpus.")
        for genre in brown.categories():
            all_brown_tagged_words.extend(brown.tagged_words(categories=genre))
        pickle.dump(all_brown_tagged_words, open(words_file_name, "wb"))

    if os.path.exists(universal_words_file_name):
        universal_all_brown_tagged_words = pickle.load(open(universal_words_file_name, "rb"))
    else:
        for genre in brown.categories():
            universal_all_brown_tagged_words.extend(
                brown.tagged_words(categories=genre, tagset='universal'))
        pickle.dump(universal_all_brown_tagged_words, open(universal_words_file_name, "wb"))

    words_to_tags_file_name = 'words_to_tags.pkl'
    tags_to_words_file_name = 'tags_to_words.pkl'
    if os.path.exists(words_to_tags_file_name) and os.path.exists(tags_to_words_file_name):
        words_to_tags = pickle.load(open(words_to_tags_file_name, "rb"))
        tags_to_words = pickle.load(open(tags_to_words_file_name, "rb"))
    else:
        print("Building mappings.")
        words_to_tags, tags_to_words = get_mappings(all_brown_tagged_words)
        pickle.dump(words_to_tags, open(words_to_tags_file_name, "wb"))
        pickle.dump(tags_to_words, open(tags_to_words_file_name, "wb"))

    return all_brown_tagged_words, universal_all_brown_tagged_words, words_to_tags, tags_to_words


if __name__ == "__main__":
    main()
