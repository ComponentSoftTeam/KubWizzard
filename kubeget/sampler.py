from itertools import accumulate
import collections
import math
import random

from matplotlib import pyplot as plt
from tqdm import tqdm
from dataset import Dataset

from dataset_statistics import create_distribution_function, kullback_leibler_divergence_corrigated, shannon_entropy
from utils import command_split

def random_with_distribution():
    while True:
        # rng = random.expovariate(1)
        rng = random.normalvariate(0.25, 0.45)
        # rng = random.uniform(0 + 1e-6, 1 - 1e-6)
        if rng > 0 and rng < 1:
            return rng

def create_scores(dataset: Dataset):
    dataset_with_context_split = [
        (entry, command_split(entry.get_context()))
        for entry in dataset
    ]

    dataset_word_counts = collections.Counter((word for _, words in dataset_with_context_split for word in words))
    return [
        (entry, kullback_leibler_divergence_corrigated(words, dataset_word_counts))
        for entry, words in tqdm(dataset_with_context_split, desc="Calculating entropy")
    ]

def sample(dataset, base_dataset, args):
    SAMPLES_NUM = args.sample

    # calculate the entropies for each entry
    dataset_with_scores = create_scores(dataset)

    # sort by entropy desc
    dataset_with_scores.sort(key=lambda x: x[1], reverse=True)
    sample_dataset = Dataset()

    with tqdm(total=SAMPLES_NUM, desc="Sampling") as pbar:
        while len(sample_dataset) < SAMPLES_NUM:
            mini_sample_size = (SAMPLES_NUM + 9) // 10
            samples = sorted([random_with_distribution() for _ in range(mini_sample_size)])
            
            # calculate the multipliers for each command

            # calculate the adaptive multiplier for each commadn based on the distribution of the base dataset
            histogram = {}
            for entry in dataset:
                histogram[entry.command_name] = histogram.get(entry.command_name, 0) + 1

            base_histogram = {}
            for entry in base_dataset:
                base_histogram[entry.command_name] = base_histogram.get(entry.command_name, 0) + 1

            adaptive_multiplier = {}
            multipliers = []
            for command_name, count in histogram.items():
                base_count = base_histogram.get(command_name, 0)
                mult = count / base_count if base_count else 1
                adaptive_multiplier[command_name] = mult
                multipliers.append(mult)

            # get the geometric mean of the multipliers
            geometric_mean = math.exp(math.fsum(math.log(m) for m in multipliers) / len(multipliers))
            # e^(1/n * sum(log(multipliers))) == e^(sum(log(multipliers^(1/n)))) == PI(multipliers^(1/n))
            
            for command_name in adaptive_multiplier:
                mult = adaptive_multiplier[command_name]
                dist = abs(mult - geometric_mean)
                if mult < geometric_mean:
                    adaptive_multiplier[command_name] = 0.05 * dist
                else:
                    adaptive_multiplier[command_name] = 10 / dist

            distribution_function = create_distribution_function([entropy * adaptive_multiplier[entry.command_name] for entry, entropy in dataset_with_scores])
            cumulative_distribution_function = list(accumulate(distribution_function))


            cdf_index = 0
            for sample in samples:
                while sample > cumulative_distribution_function[cdf_index]:
                    cdf_index += 1

                to_left = cumulative_distribution_function[cdf_index - 1]
                to_right = cumulative_distribution_function[cdf_index]

                index = cdf_index - 1 if abs(to_left - sample) < abs(to_right - sample) else cdf_index
                sample_entry = dataset_with_scores[index][0]
                
                if sample_dataset.add_entry(sample_entry):
                    pbar.update(1)

                if len(sample_dataset) >= SAMPLES_NUM:
                    break

    # plot the scores based on the cdf, the scores should be the x axis, and the cdf the y axis
    if args.plot:
        n = len(dataset)
        distribution_function = create_distribution_function([entropy for _, entropy in dataset_with_scores])
        _, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].set_title("Distribution function of the dataset")
        # axs[0, 0].plot([i / n for i in range(n)], distribution_function)
        # axs[0, 0].hist([entropy for _, entropy in dataset_with_scores], bins=100)
        axs[0, 0].hist(distribution_function, bins=100)
    
        samples = sorted([random_with_distribution() for _ in range(SAMPLES_NUM)])

        axs[0, 1].set_title("Sampling distribution")
        axs[0, 1].hist(samples, bins=100)
        axs[1, 1].set_title("Real samipling distribution")
        # This is defined as the per point product of the sampling distribution and the pdf
        axs[1, 1].hist(samples, bins=100, weights=[distribution_function[math.floor(s * n)] for s in samples])  

        sample_dataset_with_scores = create_scores(sample_dataset)
        sample_dataset_with_scores.sort(key=lambda x: x[1], reverse=True)
        sample_distribution_function = create_distribution_function([entropy for _, entropy in sample_dataset_with_scores])
        axs[1, 0].set_title("Distribution function of the sample dataset")
        axs[1, 0].hist(sample_distribution_function, bins=100)

    
    if args.verbose:
        
        histogram = {}
        for entry in sample_dataset:
            histogram[entry.command_name] = histogram.get(entry.command_name, 0) + 1

        base_histogram = {}
        for entry in base_dataset:
            base_histogram[entry.command_name] = base_histogram.get(entry.command_name, 0) + 1

        result = []
        for command_name, count in histogram.items():
            base_count = base_histogram.get(command_name, 0)
            result.append((command_name, count, count / base_count if base_count else 0))

        result.sort(key=lambda x: x[2], reverse=True)
        print('\n'.join(f'{command_name}: {count} ({multiplier:.2f}x)' for command_name, count, multiplier in result))

        
        sample_dataset_with_context_split = [
            (entry, command_split(entry.get_context()))
            for entry in sample_dataset
        ]
        
        words = [word for _, words in sample_dataset_with_context_split for word in words]

        _, avg_sample_entropy = shannon_entropy(sample_dataset)

        sample_relative_entropies = create_scores(sample_dataset)

        sample_avg_relative_entropy = sum((entropy for _, entropy in sample_relative_entropies)) / len(sample_relative_entropies)
        
        print("Top 10 of total:")
        print("\n"
              .join(f"{entropy}: {entry.command}"
                    for entry, entropy in sorted(dataset_with_scores, key=lambda x: x[1], reverse=True)[:10]
                )
            )
        
        print("\nTop 10 of sample:")
        print("\n"
              .join(f"{entropy}: {entry.command}" 
                    for entry, entropy in sorted(sample_relative_entropies, key=lambda x: x[1], reverse=True)[:10])
            )
    
        print("\n")
        verbose_dataset, _ = zip(*dataset_with_scores)
        for percentile in [1, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.1, 0.05, 0.01]:
            new_len = int(len(dataset) * percentile)
            _, avg_entropy = shannon_entropy(verbose_dataset[:new_len])

            # calculate the relative entropy
            verbose_dataset_with_context_split = [
                (entry, command_split(entry.get_context()))
                for entry in verbose_dataset[:new_len]
            ]

            words = [word for _, words in verbose_dataset_with_context_split for word in words]
            total_word_counts = collections.Counter(words)

            relative_entropies = [
                kullback_leibler_divergence_corrigated(
                    verbose_dataset_entry_words,
                    total_word_counts
                ) for _, verbose_dataset_entry_words in verbose_dataset_with_context_split
            ]
            
            avg_relative_entropy = sum(relative_entropies) / len(relative_entropies)
            # log the result
            print(f"Average shannon entropy of the top {percentile * 100}% of the dataset: {avg_entropy}")
            print(f"Average relative entropy of the top {percentile * 100}% of the dataset: {avg_relative_entropy}")
    
        print("\n") 
        print(f"Average shannon entropy of the sample dataset: {avg_sample_entropy}")
        print(f"Average relative entropy of the sample dataset: {sample_avg_relative_entropy}")
        print("\n")
    
    if args.plot:
        plt.tight_layout()
        plt.show()

    return sample_dataset