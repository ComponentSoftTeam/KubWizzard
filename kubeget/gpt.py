import hashlib
import openai
import time
import os

from config import CACHE_DIR

def gpt(system, prompt, model='gpt-3.5-turbo'):
    hash_object = hashlib.sha256(f'{system}$|${prompt}'.encode())
    hex_hash = hash_object.hexdigest()
    cache_filename: str = os.path.join(CACHE_DIR, f"{hex_hash}.txt")
    if os.path.exists(cache_filename):
        with open(cache_filename, "r", encoding="utf-8") as file:
            res: str = file.read()
        # print(f"Hit cache for gpt", file=sys.stderr)
    else:
        for attempt in range(100):  
          try:
            response = openai.ChatCompletion.create(
                model = model,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
          except openai.error.ServiceUnavailableError:
            print("Request failed, retry...")
            time.sleep(3)
          except openai.error.Timeout:
              print('Timeout exception, retry...')
              time.sleep(10)
          else:
            break
        else: 
          raise Exception('Failed to query openai chatcompletion')

        res = response.choices[0].message.content

        with open(cache_filename, "w", encoding="utf-8") as file:
                file.write(res)

        # print(f"Cache miss", file=sys.stderr)

    return res
