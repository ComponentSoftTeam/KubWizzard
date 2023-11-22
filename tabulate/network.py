import concurrent.futures

from tqdm import tqdm

def batch_request(fn, data, N):
    """Runs fn(row, i, data) for every row in data, keeps N processes running at a time (if available)"""

    with tqdm(total=len(data), leave=True, desc=f"Batch requests to {fn.__name__}") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as executor:
            for _ in executor.map(fn, data):
              pbar.update()
