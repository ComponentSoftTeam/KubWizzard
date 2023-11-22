#!/usr/bin/env python3
import argparse
from benchmark import Benchmarks

from augment import generate_chain_of_thought, generate_instructions
from dataset import Dataset
from network import upload
from sampler import sample
from scrape import parse_kubectl
from config import HUGGINGFACE_DATASET_REPO
from ruleset import ruleset_expand

def main(args):
    base_dataset = dataset = parse_kubectl()
    if args.file:
        dataset = Dataset()
        dataset.load(args.file)
        
    if args.expand:
        dataset = ruleset_expand(base_dataset, args)
    
    if args.sample:
        dataset = sample(dataset, base_dataset, args)

    if args.questions:
        dataset = generate_instructions(dataset)

    if args.cot:
        dataset = generate_chain_of_thought(Dataset(dataset[:19_800]))

    if args.upload:
        upload(dataset)
    if args.dump:
        dataset.dump(args.dump)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubectl dataset generator.')
    
    parser.add_argument('-f', '--file', help='Specify a JSON file instead of generating the data.')

    parser.add_argument('-u', '--upload', action='store_true', help='Upload data')
    parser.add_argument('-d', '--dump', help='Upload data')

    parser.add_argument('-c', '--cot', action='store_true', help="Augment the dataset with a chain of thought column")
    parser.add_argument('-q', '--questions', action='store_true', help="Augment the dataset with additional questions")

    parser.add_argument('-e', '--expand', type=int, help="Set the number of lines to expand to with substitution rules")
    parser.add_argument('-s', '--sample', type=int, help="Set the number of lines to get after sampling, based on entropy")

    parser.add_argument('-p', '--plot', action='store_true', help="Plot the entropy distribution")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print verbose information")

    args = parser.parse_args()

    # upload or dump must be specified
    if not args.upload and not args.dump:
        parser.error("Either the upload or the dump argument must be specified.")

    args = parser.parse_args()
    main(args)
  
    bench = Benchmarks()
    for fn, data in bench.items():
        total = data['total']
        n = data['n']
        print(f'{fn}: Total: {total}, Called: {n} --- {1000*total/n:.6f}ms/it')
