import multiprocessing
from dotenv import load_dotenv
from huggingface_hub import login
import openai
import os

class Config:
    """Config sigleton"""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        
        load_dotenv()
        # login(os.getenv("HUGGINGFACE_API_KEY"))
        openai.api_key = os.getenv("OPENAI_API_KEY")
      
        self.HUGGINGFACE_DATASET_REPO: str = os.getenv("HUGGINGFACE_DATASET_REPO")
        self.CACHE_DIR: str = ".cache"
        self.RULESET: str = os.getenv("RULESET")
        self.PROMPT_FILE: str = os.getenv("PROMPT_FILE")

        if not os.path.exists(self.CACHE_DIR):
          os.makedirs(self.CACHE_DIR)


config = Config()

HUGGINGFACE_DATASET_REPO = config.HUGGINGFACE_DATASET_REPO 
CACHE_DIR = config.CACHE_DIR
RULESET = config.RULESET
PROMPT_FILE = config.PROMPT_FILE