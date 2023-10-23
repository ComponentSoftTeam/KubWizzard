import json
import math



with open('35k_questions.json', 'r') as f:
    data = json.load(f)

max_len = 128
ok = [x for x in data if len(x['question']) < max_len]
data = [x for x in data if len(x['question']) >= max_len]
print("ok: ", len(ok))
print("rem: ", len(data))


for x in data:
    p = x['question']
    d = p.index('. ') if '. ' in p else len(p)
    n = p.index('\n') if '\n' in p else len(p)
    l = min(d, n)
    x['question'] = p[:l]


data = ok + data

w = ['use the following command:', 'run the following command:']


b = [x for x in data if any(v in x['question'] for v in w)]
data = [x for x in data if not any(v in x['question'] for v in w)]
# print('\n\n'.join(x['question'] for x in b))

t = [x for x in data if x['question'].startswith('To')]
data = [x for x in data if not x['question'].startswith('To')]
# print('\n\n'.join(x['question'] for x in t))
# print(len(t))

i = [x for x in data if 'instruction' in x['question'].lower()]
data = [x for x in data if 'instruction' not in x['question'].lower()]
# print(len(i))
# print('\n\n'.join(x['question'] for x in i))

f = [x for x in data if 'following command' in x['question'].lower()]
data = [x for x in data if 'following command' not in x['question'].lower()]

# print(len(f))
# print('\n\n'.join(x['question'] for x in f))

print(f'Rem: {len(data)}')

with open('cleaned.json', 'w') as f:
    json.dump(data, f)

