import datasets
import concurrent.futures

from tqdm import tqdm

from config import HUGGINGFACE_DATASET_REPO
from dataset import Dataset

def batch_request(fn, data, N):
    """Runs fn(row, i, data) for every row in data, keeps N processes running at a time (if available)"""

    with tqdm(total=len(data), leave=True, desc=f"Batch requests to {fn.__name__}") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            for __ in executor.map(fn, data):
              pbar.update()

def upload(dataset: Dataset):
    """Uploads a train and validate split to huggingface"""
    
    hub_dataset = datasets.Dataset.from_list([ entry.dict() for entry in dataset])
    hub_dataset.push_to_hub(HUGGINGFACE_DATASET_REPO)
