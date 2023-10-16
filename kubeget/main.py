#!/usr/bin/env python3
import json
import argparse
import multiprocessing
from benchmark import Benchmarks, benchmark

from augment import gen_cot, gen_qi
from network import upload
from scrape import parse_kubectl
from config import HUGGINGFACE_DATASET_REPO
from ruleset import ruleset_expand
from utils import load_json

def main(args):
    
    if args.file:
        json_data = load_json(args.file)
        train_dataset, validate_dataset = json_data['train'], json_data['validate']
    else:
        dataset = parse_kubectl()
        

    if args.ruleset:
        dataset = ruleset_expand(dataset, args.size or len(dataset))
    if args.questions:
        dataset = gen_qi(dataset)

    if args.cot:
        dataset = gen_cot(dataset)

    if args.upload:
        upload(train_dataset, validate_dataset)
    elif args.dump:
        with open(args.dump, 'w') as file:
            json.dump({'train': train_dataset, 'validate': validate_dataset}, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kubectl dataset generator.')
    
    parser.add_argument('-f', '--file', help='Specify a JSON file instead of generating the data.')

    parser.add_argument('-u', '--upload', action='store_true', help='Upload data')
    parser.add_argument('-d', '--dump', help='Upload data')

    parser.add_argument('-r', '--ruleset', action='store_true', help="Augment the dataset with additional questions based on substitution rules")
    parser.add_argument('-c', '--cot', action='store_true', help="Augment the dataset with a chain of thought column")
    parser.add_argument('-q', '--questions', action='store_true', help="Augment the dataset with additional questions")
    
    parser.add_argument('-s', '--size', type=int, help="Set the number of lines in the final dataset")

    args = parser.parse_args()

    if args.size and not args.ruleset:
        parser.error("The size argument is only valid if the ruleset argument is specified.")

    args = parser.parse_args()
    main(args)
  
    bench = Benchmarks()
    for fn, data in bench.items():
        total = data['total']
        n = data['n']
        print(f'{fn}: Total: {total}, Called: {n} --- {1000*total/n:.6f}ms/it')
