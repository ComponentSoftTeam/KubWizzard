import json
from config import PROMPT_FILE
from dataset import Dataset
from gpt import gpt
from tqdm import tqdm
import random
from network import batch_request

from utils import load_json

K8S_COT = "You are a helpful developer that knows kubernetes and the kubectl cli. You aim to generate the chain of thought resoning for the provided commands, based on the natrural language instruction, and the documentation for the command."

def generate_chain_of_thought(dataset: Dataset):
    """ Generate the chain of thought reasoning for the dataset """
    
    dataset = dataset.copy()
    questions = load_json(PROMPT_FILE)['examples']

    # seed the random function to make the caching work
    random.seed(0)
    for entry in tqdm(dataset, leave=True, desc="Generate CoT"):
        command = entry.command
        objective = entry.objective
        question = entry.question

        docs = json.dumps(entry.dict())
        examples = random.sample(questions, 3)
        example_prompt = '\n\n'.join(f'General idea: {example["general"]}\Instruction: {example["instruction"]}\nCommand: {example["command"]}\nChain of Thought: {example["chain_of_thought"]}' for example in examples)

        prompt = (
            f'Generate the chain of thought process for the given command, based on the general idea and the instruction.\n\n'
            f'{example_prompt}\n\n'
            f'General idea: {objective}\n'
            f'Instruction: {question}\n'
            f'Command: {command}\n'
            f'You can find the documentation for the command here in JSON format: {docs}\n\n'
            f'Chain of Thought: '
        )

        chain_of_thought = gpt(K8S_COT, prompt)

        entry.chain_of_thought = chain_of_thought

        print("Prompt", '-'*10, '\n')
        print(prompt)
        print("Answer", '-'*10, '\n')

        print(chain_of_thought)
        print('-'*16, '\n')

        input("Press enter to continue")
        

    return dataset


K8S_INSTRUCTION = "You are helpful assistant that knows kubernetes and the kubectl cli. You aim to generate instructions for the provided commands, and the general idea."

def generate_instructions(dataset: Dataset):
    "Generate questions and instructions for the data"

    dataset = dataset.copy()

    questions = load_json(PROMPT_FILE)['examples']
    specific_questions = [question for question in questions if question['specific']]
    general_questions = [question for question in questions if not question['specific']]

    # seed the random function to make the caching work
    # generating the prompting separately to make caching more efficient
    random.seed(0)
    dataset_with_prompts = []
    for entry in tqdm(dataset, leave=True, desc="Generate prompts"):
        command = entry.command
        objective = entry.objective
        
        # using random.choice select 3 random specific questions
        is_specific = random.choice([True, False])
        if is_specific:
            examples = random.sample(specific_questions, 3)
            leader_prompt = f'For a given command and the general idea, provide in instruction for which the command is the answer, please include the parameters in your instruction'

        else:
            examples = random.sample(general_questions, 3)
            leader_prompt = f'For a given command, provide in instruction for which the command is the answer, please give a general instruction as your answer'


        example_prompt = '\n\n'.join(f'General idea: {example["general"]}\nCommand: {example["command"]}\nInstruction: {example["instruction"]}' for example in examples)
        
        prompt = (
            f'{leader_prompt}\n\n'
            f'{example_prompt}\n\n'
            f'General idea: {objective}\n'
            f'Command: {command}\n'
            f'Instruction: '
        )

        dataset_with_prompts.append((entry, prompt))

    def batch_gpt(data):
        entry, prompt = data
        entry.question = gpt(K8S_INSTRUCTION, prompt).strip()

    batch_request(batch_gpt, dataset_with_prompts, 12)

    return dataset
