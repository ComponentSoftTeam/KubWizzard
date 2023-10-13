import json
import re
import random
from config import RULESET
from tqdm import tqdm

def color_text_with_mask(text, mask):
    colors = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m']
    reset_color = '\033[0m'

    colored_text = ""

    for char, color_index in zip(text, mask):
        if 0 <= color_index < len(colors):
            colored_text += f"{colors[color_index]}{char}{reset_color}"
        else:
            colored_text += char

    return colored_text

class Matcher:
    def __init__(self, pattern):
        self.pattern = pattern

    def match(self, text):
        raise NotImplementedError('Subclasses must implement the match method')

    def sub(self, text):
        raise NotImplementedError('Subclasses must implement the sub method')

    @classmethod
    def create(cls, pattern:str):
       pattern = pattern.strip()
       if '|' in pattern:
           return MultiMatcher(pattern)
       
       if (pattern.startswith('[') and not pattern.endswith(']')) or (not pattern.startswith('[') and pattern.endswith(']')):
            raise ValueError(f'Invalid pattern {pattern}')
       
       if pattern.startswith('['):
           return ListMatcher(pattern)
       elif pattern == '':
           return AnyMatcher('')
       else:
           return ExactMatcher(pattern)

# For ''
class AnyMatcher(Matcher):
    def __init__(self, pattern):
        super().__init__(pattern)

    def match(self, text):
        return (True, 0, self)
    
    def sub(self, text, new_val):
        return []
    
# For foo
class ExactMatcher(Matcher):
    def __init__(self, pattern):
        super().__init__(pattern)

    def match(self, text):
        res = self.pattern in text
        if not res:
            return False
        return (True, 3, self)
    
    def sub(self, text, new_val):
        start_positions = [match.start() for match in re.finditer(self.pattern, text)]

        return [(s, len(self.pattern), new_val) for s in start_positions]

# For foo|bar
class MultiMatcher(Matcher):
    def __init__(self, pattern):
        super().__init__(pattern)
        sub_patterns = [x.strip() for x in pattern.split('|')]
        self.matchers = [Matcher.create(p) for p in sub_patterns]
    
    def match(self, text):
        matches = [m.match(text) for m in self.matchers]
        matches = [m for m in matches if m]
        matches = [(b, p, matcher) for (b, p, matcher) in matches]
        if not matches:
            return False
        return max(matches, key=lambda x: x[1])

# For [foo, bar]
class ListMatcher(Matcher):
    def __init__(self, pattern):
        super().__init__(pattern)

        pattern = pattern[1:-1].strip()
        self.sub_patterns = [p.strip() for p in pattern.split(',')]

    def match(self, text):
        valid = [[-1]]
        for pattern in self.sub_patterns:
            start_indexes = [m.start() for m in re.finditer(pattern, text)]
            valid.append(start_indexes)

        latest = -1
        positions=[]
        # Note that this does not check for overlapping sequences
        for start_indexes in valid[1:]:
            for index in start_indexes:
                if index > latest:
                    latest = index
                    break
            else:
                return False
            positions.append(latest)

        return (True, 4, self)
    
    def sub(self, text, new_vals):
        valid = [[-1]]
        for pattern in self.sub_patterns:
            start_indexes = [m.start() for m in re.finditer(pattern, text)]
            valid.append(start_indexes)

        latest = -1
        positions=[]
        # Note that this does not check for overlapping sequences
        for start_indexes in valid[1:]:
            for index in start_indexes:
                if index > latest:
                    latest = index
                    break
            else:
                return False
            positions.append(latest)

        new_values = self.sub_patterns
        new_vals = [v.strip() for v in new_vals.strip()[1:-1].split(',')]
        return [(s, len(p), v) for s, p, v in zip(positions, self.sub_patterns, new_vals)]

class RuleSub:
    def __init__(self, text):
        self.text = list(text)
        self.mask = [0] * len(text)

    def show(self):
        return ''.join(self.text)
    
    def get_str(self):
        string = []
        for i in range(len(self.text)):
            if i == 0 or (self.mask[i] != 0 and self.mask[i - 1] == 0):
                string.append([])

            if self.mask[i] == 0:
                string[-1].append(self.text[i])
        
        return 'űáű'.join([''.join(s) for s in string])
    
    def highlight(self):
        return color_text_with_mask(''.join(self.text), self.mask)

class Rule:
    def __init__(self, rule:str, values):
        self.rule = rule

        rule = rule.strip()
        self.is_global = not rule.startswith('$')
        if not self.is_global:
            rule = rule[1:]

        self.namespace = None
        if not self.is_global:
            if '#' not in rule or len(rule.split('#')) != 2:
                raise ValueError(f'Invalid rule {self.rule}')

            self.namespace, rule = rule.split('#')
            self.namespace = self.namespace.strip()

        rule = rule.strip()
        if ':' not in rule or len(rule.split(':')) != 2:
            raise ValueError(f'Invalid rule {self.rule}')

        desc_rule, code_rule = rule.split(':')

        self.values = [(s["d-sub"], s["c-sub"]) for s in values]
        self.desc_matcher = Matcher.create(desc_rule)
        self.code_matcher = Matcher.create(code_rule)


    def match(self, desc, code, namespace=''):
        if namespace == '':
            if not self.is_global:
                return False
        else:
            if not self.is_global and namespace != self.namespace:
                return False
            
        prio = 0 if self.is_global else 16

        matches_desc = self.desc_matcher.match(desc.get_str())
        if not matches_desc: 
            return False
        matches_code = self.code_matcher.match(code.get_str())
        if not matches_code: 
            return False
        
        (_, prio_desc, matcher_desc) = matches_desc
        (_, prio_code, matcher_code) = matches_code

        prio += prio_desc + prio_code
        return (True, prio, matcher_desc, matcher_code)
    

class RuleSet:
    def __init__(self, ruleset_json):
        self.rules = [Rule(rule, sub) for (rule, sub) in ruleset_json.items()]

    def expand(self, desc, code, namespace=''):
        desc = RuleSub(desc)
        code = RuleSub(code)

        matching_rules = [(match[1], rule, match[2], match[3]) for rule, match in ((rule, rule.match(desc, code, namespace)) for rule in self.rules) if match]

        sub_order = 0
        # print(f'# From\nDescription: {desc.highlight()}\nCode: {code.highlight()}\n')

        while matching_rules:
            sub_order += 1
            _, rule, matcher_desc, matcher_code = max(matching_rules, key=lambda x: x[0])

            new_val_desc, new_val_code = random.choice(rule.values)

            for rulesub, text_str, matcher, new_val in [(desc, desc.get_str(), matcher_desc, new_val_desc), (code, code.get_str(), matcher_code, new_val_code)]:
                n = len(rulesub.text)
                shift_sep_mask_text = [0]*len(text_str)
                for i in range(len(shift_sep_mask_text)-2):
                    if text_str[i] == 'ű' and text_str[i + 1] == 'á' and text_str[i + 2] == 'ű':
                        shift_sep_mask_text[i] = -3

                for i in range(1, len(shift_sep_mask_text)):
                    shift_sep_mask_text[i] += shift_sep_mask_text[i - 1]
                    
                shift_mask_text = [0] * n
                i = 0
                while i < n:
                    while i < n and rulesub.mask[i] == 0:
                        i += 1
                    if i == n:
                        break

                    offset = 0
                    while i + offset < n and rulesub.mask[i + offset] != 0:
                        offset += 1
                    
                    shift_mask_text[i] = offset

                    i += offset

                for i in range(1, n):
                    shift_mask_text[i] += shift_mask_text[i - 1]
                

                shift_mask_text = [m for m, b in zip(shift_mask_text, rulesub.mask) if b == 0]
                sub_text = matcher.sub(text_str, new_val)
                sub_text.sort(key=lambda x: -x[0])

                for start, length, value in sub_text:
                    start += shift_sep_mask_text[start]
                    start += shift_mask_text[start]

                    rulesub.text = rulesub.text[:start] + list(value) + rulesub.text[start + length:]
                    middle = rulesub.mask[start:start+length]
                    if any(x != 0 for x in middle):
                        raise RuntimeError("writing restricted characters")
                    rulesub.mask = rulesub.mask[:start] + [sub_order] * len(value) + rulesub.mask[start + length:]
                    
            
            rules = [rule for (_, rule, _, _) in matching_rules]
            matching_rules = [(match[1], rule, match[2], match[3]) for rule, match in ((rule, rule.match(desc, code, namespace)) for rule in rules) if match]
        # print(f'# Expanded to \nDescription: {desc.highlight()}\nCode: {code.highlight()}\n\n')
        return (desc.show(), code.show())

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
    
def ruleset_expand(dataset, size=0):
    print(RULESET)
    with open(RULESET, "r") as f:
        ruleset_json = json.load(f)
    
    ruleset = RuleSet(ruleset_json)

    result_dataset = set()
    

    with tqdm(
        total=size,
        leave=True,
        desc="Generate Params ",
    ) as pbar:
        while len(result_dataset) < size:
            for data in dataset:
                command = data['command']
                description = data['description']
                syntax = data['syntax']
                flags = data['flags']
                examples = data['examples']

                for example in examples:
                    example_description = example['description']
                    example_code = example['code']

                    example_description, example_code = ruleset.expand(example_description, example_code, command.strip())
                    # input("")
                    entry = hashabledict({
                        'command': command,
                        'description': description,
                        'syntax': syntax,
                        'flags': flags,
                        'code': example_code,
                        'question': example_description
                    })

                    if entry not in result_dataset:
                        result_dataset.add(entry)
                        pbar.update(1)
        