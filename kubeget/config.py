from dotenv import load_dotenv
from huggingface_hub import login
import openai
import os

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self):
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # login(os.getenv("HUGGINGFACE_API_KEY"))
      
        self.HUGGINGFACE_DATASET_REPO: str = os.getenv("HUGGINGFACE_DATASET_REPO")
        self.CACHE_DIR: str = ".cache"
        self.base_url: str = "https://kubernetes.io/docs/reference/"
        self.ruleset: str = os.getenv("RULESET")

        if not os.path.exists(self.CACHE_DIR):
          os.makedirs(self.CACHE_DIR)


config = Config()

HUGGINGFACE_DATASET_REPO = config.HUGGINGFACE_DATASET_REPO 
CACHE_DIR = config.CACHE_DIR
base_url = config.base_url
RULESET = config.ruleset
