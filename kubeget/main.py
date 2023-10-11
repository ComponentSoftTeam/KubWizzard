#!/usr/bin/env python3
import json
import argparse

from augment import gen_cot, gen_qi
from ops import flatten_and_split, upload
from scrape import parse_kubectl
from config import HUGGINGFACE_DATASET_REPO


def main(args):
    
    if args.file:
        with open(args.file, 'r') as file:
            json_data = json.load(file)

        train_dataset, validate_dataset = json_data['train'], json_data['validate']
    elif args.generate:
        dataset = parse_kubectl()
        dataset = gen_cot(dataset)
        dataset = gen_qi(dataset)

        (train_dataset, validate_dataset) = flatten_and_split(dataset)
    
    if args.upload:
        upload(train_dataset, validate_dataset)
    elif args.dump:
        with open(args.dump, 'w') as file:
            json.dump({'train': train_dataset, 'validate': validate_dataset}, file)

if __name__ == '__main__':
    def is_valid_json_file(arg):
        try:
            with open(arg, 'r') as file:
                json.load(file)
            return arg
        except (FileNotFoundError, json.JSONDecodeError):
            raise argparse.ArgumentTypeError(f"'{arg}' is not a valid JSON file")

    parser = argparse.ArgumentParser(description='Kubectl dataset generator.')
    group_in = parser.add_mutually_exclusive_group()
    group_in.add_argument('-f', '--file', type=is_valid_json_file, help='Specify a valid JSON file')
    group_in.add_argument('-g', '--generate', action='store_true', help='Generate data')
    group_out = parser.add_mutually_exclusive_group()
    group_out.add_argument('-u', '--upload', action='store_true', help='Upload data')
    group_out.add_argument('-d', '--dump', help='Upload data')

    args = parser.parse_args()

    main(args)