import openai
import time
import sys

from utils import cached

@cached
def gpt(system, prompt, model='gpt-3.5-turbo'):
    for _ in range(100):  
        try:
            response = openai.ChatCompletion.create(
                model = model,
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
        except openai.error.ServiceUnavailableError:
            print("Request failed, retry...", file=sys.stderr)
            time.sleep(3)
        except openai.error.Timeout:
            print('Timeout exception, retry...', file=sys.stderr)
            time.sleep(10)
        else:
            break
    else: 
        raise Exception('Failed to query openai chatcompletion')

    return response.choices[0].message.content