import json
import math

import matplotlib.pyplot as plt

def cleanup_question():

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

def cleanup_cot():
    with open('cot.json', 'r') as f:
        data = json.load(f)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    # Create the first histogram on the first subplot
    

    lens = []
    for entry in data:
        cot = entry['chain_of_thought']
        lens.append(len(cot))

    # ax1.hist(lens, bins=100, edgecolor='black', alpha=0.7)

    # ax1.set_title("Before")
    # ax1.set_xlabel("Values")
    # ax1.set_ylabel("Frequency")
    # ax1.set_ylim(0, 10)
   
    # Display the plot

    lens = []
    remain = []
    for entry in data:
        cot = entry['chain_of_thought']

        lines = [l.strip() for l in cot.split('\n')]
        
        k = 0;
        while lines[k] == "":
            k+=1

        lines = lines[k:]
        
        if "" in lines:
            k = lines.index("")
            lines = lines[:k]

        cot = '\n'.join(lines)


        if len(cot) <= 2000:
            lens.append(len(cot))
            entry['chain_of_thought'] = cot
            remain.append(entry)

    data = remain
     # Create the second histogram on the second subplot
    # ax2.hist(lens, bins=100, edgecolor='black', alpha=0.7)
    # ax2.set_title("After")
    # ax2.set_xlabel("Values")
    # ax2.set_ylabel("Frequency")
    # ax2.set_ylim(0, 10)
    # # Adjust the layout to avoid overlapping labels
    # plt.tight_layout()

    # Display the plot
    # plt.show()


    longest = max(data, key=lambda x: len(x['chain_of_thought']))

    cnt = 0
    remain = []
    for entry in data:
        try:
            if any(not c.split()[1][0].isalpha() for c in entry['chain_of_thought'].split('\n')):
                # print(f"error with: {entry['chain_of_thought']}")
                # cnt += 1
                continue
        except:
            # print(f"big error with: {entry['chain_of_thought']}")
            # cnt += 1
            continue

        remain.append(entry)


    data = remain

    red_flags = [
        "by default",
        "default value",
        "do not use",
        "does not specify",
        "if necessary",
        "command doesn't specify",
        "default is",
        "defaults for",
        "the default",
        "default behavior",
        "other possible",
        "optional flags",
        "command does not",
        "other flags",
    ]

    lens = []
    remain = []
    for entry in data:
        cot = entry['chain_of_thought']
        # if len(cot.split('\n')) > 5:

        marks = []
        for l in red_flags:
            if l in cot.lower():
                marks.append(cot.lower().index(l))
                
        marks.sort()

        c = 0
        while c < len(marks) and cot[:marks[c]].count('\n') + 1 <= 3:
            c += 1

        marks = marks[c:]

        if marks:
            cut = marks[0]
            if len(marks) > 1 and cot[:marks[1]].count('\n') + 1 < 6:
                cut = marks[1]

            if len(marks) > 2 and cot[:marks[2]].count('\n') + 1 < 6:
                cut = marks[2]

            cot = '\n'.join(cot[:cut].split('\n')[:-1]).strip() 

        if not cot:
            print(f"\n\nThis entry is out: {entry['chain_of_thought']}\n")
            continue
        
        if len(cot) > 500:
            cot = '\n'.join(cot[:500].split('\n')[:-1]).strip() 


        entry['chain_of_thought'] = cot.strip()
        remain.append(entry)
        
        # lens.append(len(cot))

    dif = len(data) - len(remain)
    data = remain
    print(f"dif: {dif}")
    # print(f'cnt: {cnt}')
    # plt.hist(lens, bins=100)
    # plt.show()
    # print(f"cnt: {len(data)}")
    # print(json.dumps(longest))

    with open("cot_clean.json", "w") as f:
        json.dump(data, f)

cleanup_cot()