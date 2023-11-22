import openai
import time
import sys

from openai import error

from utils import cached

@cached
def gpt(system, prompt, model='gpt-3.5-turbo'):
    for _ in range(4):  
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
            time.sleep(10)
        except openai.error.Timeout:
            print('Timeout exception, retry...', file=sys.stderr)
            time.sleep(30)
        except error.RateLimitError:
            print('Hit rate limit, retry', file=sys.stderr)
            time.sleep(30)
        except error.TryAgain:
            print('Error retry...', file=sys.stderr)
            time.sleep(30)
        except Exception as ex:
            print(f"Unknow error: {str(ex)}")
            time.sleep(60)
        else:
            break
    else: 
        raise Exception('Failed to query openai chatcompletion')

    print(f"\n\nResponse:\n{response.choices[0].message.content}\n\n")
    return response.choices[0].message.content