import collections
import math

from utils import command_split


def shannon_entropy(words):
    """Calculates the entropy of the dataset"""
    word_counts = collections.Counter(words)
    total_words = len(words)

    entropy = 0
    for _, count in word_counts.items():
        probability = count / total_words
        entropy -= probability * math.log2(probability)
    return entropy, entropy / len(word_counts)

def kullback_leibler_divergence_average(words, dataset_word_counts):
    """For an input sentance, calculates how much information does it contain compared to the whole dataset"""

    total_words_in_dataset = sum(x for x in dataset_word_counts.values())
    word_counts = collections.Counter(words)
    total_words = len(words)
    
    entropy = 0
    for word, count in word_counts.items():
        p_w = count / total_words
        q_w = dataset_word_counts[word] / total_words_in_dataset
        entropy += p_w * math.log2(p_w / q_w)
    
    return entropy / len(word_counts)


def kullback_leibler_divergence_corrigated(words, dataset_word_counts):
    """
    For an input sentance, calculates how much information does it contain compared to the whole dataset,
    Corrected with the position of each the word in the sentance and with other factors.
    """

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    def pos_corrigated_sigmoid(p):
        amp = 0.8
        intesiti = 10
        result = - amp * sigmoid(p * intesiti - intesiti / 2) + 1
        return result

    total_words_in_dataset = sum(x for x in dataset_word_counts.values())

    word_counts = collections.Counter(words)
    total_words = len(words)
    
    entropy = 0

    for word, count in word_counts.items():
        positions = [i for i, w in enumerate(words) if w == word]
        avg_position = sum(positions) / len(positions)
        pos_precentile = avg_position / (len(words) - 1) if len(words) > 2 else 0
        path_correction = 0.5 ** word.count('/')
        number_correction = 0.07 ** count if word.isnumeric() else 1
        correction = pos_corrigated_sigmoid(pos_precentile) * path_correction * number_correction
        p_w = count / total_words
        q_w = dataset_word_counts[word] / total_words_in_dataset
        entropy += correction * p_w * math.log2(p_w / q_w)
    
    return entropy / len(word_counts)


def create_distribution_function(scores):
    """
    From a list of scores, creates a distribution function.
    A distribution function in this case is a list of numbers between 0 and 1,
    and the sum of the values is 1.
    """
    min_score = min(scores)
    adjusted_scores = [score - min_score for score in scores]
    sum_scores = sum(adjusted_scores)
    return [score / sum_scores for score in adjusted_scores]