import hashlib
import pickle
import json
import os

import time
import random
import string

from config import CACHE_DIR

def get_unique_id(length=8):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
     
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)
    
def dump_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def cached(func):
    def wrapper(*args, **kwargs):
        SEP = '$|$'
        cache_token = (
            f'{func.__name__}{SEP}'
            f'{SEP.join(str(arg) for arg in args)}{SEP}'
            f'{SEP.join( str(key) + SEP * 2 + str(val) for key, val in kwargs.items())}'
        )

        hex_hash = hashlib.sha256(cache_token.encode()).hexdigest()
        cache_filename: str = os.path.join(CACHE_DIR, f"{hex_hash}")

        if os.path.exists(cache_filename):
            with open(cache_filename, "rb") as cache_file:
                return pickle.load(cache_file)
        
        result = func(*args, **kwargs)
        with open(cache_filename, "wb") as cache_file:
            pickle.dump(result, cache_file)
        
        return result
    return wrapper