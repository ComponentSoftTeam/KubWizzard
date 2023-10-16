from datasets import DatasetDict, Dataset
import concurrent.futures

from config import HUGGINGFACE_DATASET_REPO

def batch_request(fn, data, N):
    """Runs fn(row, i, data) for every row in data, keeps N processes running at a time (if available)"""

    with concurrent.futures.ThreadPoolExecutor(N) as executor:
        futures = [executor.submit(fn, row, i, data) for i, row in enumerate(data)]
        for future in concurrent.futures.as_completed(futures):
            future.result()



def upload(train_dataset, validate_dataset):
    """Uploads a train and validate split to huggingface"""
    
    ddict = DatasetDict({
        "train": Dataset.from_list(train_dataset),
        "validate": Dataset.from_list(validate_dataset),
    })
    ddict.push_to_hub(HUGGINGFACE_DATASET_REPO)
