from tqdm import tqdm
import json
from datasets import DatasetDict, Dataset

def flatten_and_split(dataset):
    train_dataset = []
    validate_dataset = []

    with open('data.json', 'w') as f:
        json.dump(dataset, f)

    total = sum(len(x['examples']) for x in dataset)
    for data in dataset:
        for x in data['examples']:
            total += len(x['questions'])

    with tqdm(total=total, leave=True, desc="Generate dataset") as pbar:

      for data in dataset:
          command = data['command']
          description = data['description']
          syntax = data['syntax']
          examples = data['examples']
          flags = data['flags']

          for example in examples:
              example_description = example['description']
              example_code = example['code']
              example_cot = example['cot']
              example_questions = example['questions']

              for question in example_questions:
                  train_dataset.append({
                      'command': command,
                      'description': description,
                      'syntax': syntax,
                      'flags': flags,
                      'code': example_code,
                      'cot': example_cot,
                      'question': question
                  })
                  pbar.update(1)

      for data in dataset:
          command = data['command']
          description = data['description']
          syntax = data['syntax']
          examples = data['examples']
          flags = data['flags']

          for example in examples:
              example_description = example['description']
              example_code = example['code']
              example_cot = example['cot']
              example_questions = example['questions']

              validate_dataset.append({
                  'command': command,
                  'description': description,
                  'syntax': syntax,
                  'flags': flags,
                  'code': example_code,
                  'cot': example_cot,
                  'question': example_description
              })

              pbar.update(1)

    return (train_dataset, validate_dataset)
