from typing import List, Tuple
from tqdm import tqdm

import random
import json
import re
import array

from multiprocessing import Pool
import multiprocessing

from itertools import pairwise, count, takewhile, accumulate
from benchmark import benchmark

from config import RULESET
from dataset import Dataset, Entry

class Matcher:
    def __init__(self, pattern):
        self.pattern = pattern

    def match(self, text):
        raise NotImplementedError("Subclasses must implement the match method")

    def sub(self, text):
        raise NotImplementedError("Subclasses must implement the sub method")

    @classmethod
    def create(cls, pattern: str):
        pattern = pattern.strip()
        if "|" in pattern:
            return MultiMatcher(pattern)

        if pattern.startswith("[") != pattern.endswith("]"):
            raise ValueError(f"Invalid pattern {pattern}")

        if pattern.startswith("["):
            return ListMatcher(pattern)

        if pattern == "":
            return AnyMatcher("")

        return ExactMatcher(pattern)

class AnyMatcher(Matcher):
    """Matching rule(s): ''"""

    PRIO = 0

    def __init__(self, pattern):
        super().__init__(pattern)

    def match(self, text):
        return (AnyMatcher.PRIO, self)


    def sub(self, text, new_val):
        return []

class ExactMatcher(Matcher):
    """
    Matching rule(s): 'foo'
    Matches every accurance of the pattern
    """

    PRIO = 3

    def __init__(self, pattern):
        super().__init__(pattern)
        self.re_pattern = re.compile(re.escape(pattern))

    def match(self, text):
        return (ExactMatcher.PRIO, self) if self.pattern in text else None

    def sub(self, text, new_val):
        start_position = next((match.start() for match in self.re_pattern.finditer(text)))
        return [(s, len(self.pattern), new_val) for s in [start_position]]

class MultiMatcher(Matcher):
    """
    Matching rule(s): any rule that a subpattern has
    Matches the pattern with the highest priority out of the matching patterns
    """

    # Does not do substitution, only the subpatterns substitues, so it does not need a priority

    def __init__(self, pattern):
        super().__init__(pattern)
        sub_patterns = (x.strip() for x in pattern.split("|"))
        self.matchers = [Matcher.create(p) for p in sub_patterns]

    def match(self, text):
        match_results = (m.match(text) for m in self.matchers)
        matches = [m for m in match_results if m]
        return max(matches, key=lambda x: x[0]) if matches else None

class ListMatcher(Matcher):
    """
    Matching rule(s): 'foo' and 'bar' as a substring in this order
    Only matches the first accurance of the subpatterns where they match in order
    """

    PRIO = 4

    def __init__(self, pattern):
        super().__init__(pattern)

        pattern = pattern[1:-1].strip()
        self.re_sub_patterns = [(re.compile(re.escape(p.strip())), len(p.strip())) for p in pattern.split(",")]

    def match(self, text):
        valid = ((m.start() for m in re_pattern.finditer(text)) for (re_pattern, _) in self.re_sub_patterns)

        latest = -1
        for start_indexes in valid:
            latest = next((ind for ind in start_indexes if ind > latest), None)
            if not latest:
                return None

        return (ListMatcher.PRIO, self)

    def sub(self, text, new_vals):
        valid = ((m.start() for m in re_pattern.finditer(text)) for (re_pattern, _) in self.re_sub_patterns)

        latest = -1
        positions = []

        # TODO(Kristofy): this does not check for overlapping words in the sequence
        # So [apple, lemma] in 'appbananalemma lemma' would both match in applemma instead
        # of the expected matching with the two separate words
        for start_indexes in valid:
            latest = next((ind for ind in start_indexes if ind > latest), None)
            if not latest:
                raise RuntimeError("The pattern sould match when sub is called")
            positions.append(latest)

        new_vals = (v.strip() for v in new_vals.strip()[1:-1].split(","))
        return [(s, length, v) for s, (_, length), v in zip(positions, self.re_sub_patterns, new_vals)]

class RuleSub:
    mem = dict()
    SEP_STR = "űáű"
    SEP = array.array("i", [ord(c) for c in SEP_STR])
    SEP_LEN = len(SEP_STR)

    def __init__(self, text):
        self.text = array.array("u")
        self.text.fromunicode(text)
        self.mask = array.array("i", [0] * len(text))

    def show(self):
        return self.text.tounicode()

    def get_str(self):
        key = (self.text.tobytes(), self.mask.tobytes())

        if key not in RuleSub.mem:
            RuleSub.mem[key] = "".join(
                [
                    curr_char if curr == 0 else RuleSub.SEP_STR
                    for ((curr, next), curr_char) in zip(pairwise(self.mask), self.text)
                    if curr == 0 or next == 0
                ]
                + ([self.text[-1]] if self.mask[-1] == 0 else [])
            )

        return RuleSub.mem[key]

    def highlight(self):
        # Terminal escape sequences for colors
        RESET_COLOR = "\033[0m"
        COLORS = [
            RESET_COLOR,
            "\033[91m",
            "\033[92m",
            "\033[93m",
            "\033[94m",
            "\033[95m",
            "\033[96m",
            "\033[97m",
        ]

        n = len(COLORS)
        return "".join(
            [f"{COLORS[color_index % n]}{char}{RESET_COLOR}" for char, color_index in zip(self.text, self.mask)]
        )

class Rule:
    def __init__(self, rule: str, values):
        self.rule = rule

        rule = rule.strip()
        self.is_global = not rule.startswith("$")
        if not self.is_global:
            rule = rule[1:]

        self.namespace = None
        if not self.is_global:
            if "#" not in rule or len(rule.split("#")) != 2:
                raise ValueError(f"Invalid rule {self.rule}, no namespace separator, despite being a namespace rule")

            self.namespace, rule = rule.split("#")
            self.namespace = self.namespace.strip()

        rule = rule.strip().replace("\\:", RuleSub.SEP_STR)
        if ":" not in rule or len(rule.split(":")) != 2:
            if ":" not in rule:
                raise ValueError(f"Invalid rule {self.rule}, no separator")
            else:
                raise ValueError(f"Invalid rule {self.rule}, too many separators")
        desc_rule, code_rule = rule.split(":")
        
        desc_rule = desc_rule.strip().replace(RuleSub.SEP_STR, ":")
        code_rule = code_rule.strip().replace(RuleSub.SEP_STR, ":")

        self.values = [(s["d-sub"], s["c-sub"]) for s in values]
        self.desc_matcher = Matcher.create(desc_rule)
        self.code_matcher = Matcher.create(code_rule)

        if not isinstance(self.desc_matcher, AnyMatcher):
            if any(d_sub.strip() == "" for (d_sub, _) in self.values):
                raise ValueError(f"Invalid rule {self.rule}, empty description substitution")
            
        if not isinstance(self.code_matcher, AnyMatcher):
            if any(c_sub.strip() == "" for (_, c_sub) in self.values):
                raise ValueError(f"Invalid rule {self.rule}, empty code substitution")

        # desc and code matcher cannot have both an any mathcer at the same time
        desc_has_any = isinstance(self.desc_matcher, AnyMatcher) or (isinstance(self.desc_matcher, MultiMatcher) and any(isinstance(sub_matcher, AnyMatcher) for sub_matcher in self.desc_matcher.matchers) )
        code_has_any = isinstance(self.code_matcher, AnyMatcher) or (isinstance(self.code_matcher, MultiMatcher) and any(isinstance(sub_matcher, AnyMatcher) for sub_matcher in self.code_matcher.matchers) )
        if desc_has_any and code_has_any:
            raise ValueError(f"Invalid rule {self.rule}, both description and code matcher has any as a rule")

    def match(self, desc, code, namespace=None):
        if not self.is_global and namespace != self.namespace:
            return None

        prio = 0 if self.is_global else 16

        matches_desc = self.desc_matcher.match(desc.get_str())
        if not matches_desc:
            return None
        matches_code = self.code_matcher.match(code.get_str())
        if not matches_code:
            return None

        (prio_desc, matcher_desc) = matches_desc
        (prio_code, matcher_code) = matches_code

        prio += prio_desc + prio_code
        return (prio, matcher_desc, matcher_code)

class RuleSet:
    mem = dict()

    def __init__(self, ruleset_json):
        self.rules = [Rule(rule, sub) for (rule, sub) in ruleset_json.items()]

    def expand(self, desc, code, namespace=""):
        desc = RuleSub(desc)
        code = RuleSub(code)

        # The masks are still empty
        sub_key = (desc.text.tobytes(), code.text.tobytes())

        # Caching based on only the text, because at the beginning the mask is always empty
        if sub_key not in RuleSet.mem:
            RuleSet.mem[sub_key] = [
                (match[0], rule, match[1], match[2])
                for rule, match in zip(
                    self.rules,
                    map(lambda rule: rule.match(desc, code, namespace), self.rules),
                )
                if match
            ]

        matching_rules = RuleSet.mem[sub_key]

        for sub_order in takewhile(lambda _: matching_rules, count(start=1)):
            _, rule, matcher_desc, matcher_code = max(matching_rules, key=lambda x: x[0])
            new_val_desc, new_val_code = random.choice(rule.values)

            for text_obj, matcher, new_val in [
                (desc, matcher_desc, new_val_desc),
                (code, matcher_code, new_val_code),
            ]:
                text_obj_len = len(text_obj.text)

                text_str:str = text_obj.get_str()
                text_str_len:int = len(text_str)

                shift_sep_mask_text = array.array('i', [0] * text_str_len)
                for i in (match.start() for match in re.finditer(RuleSub.SEP_STR, text_str)):
                    shift_sep_mask_text[i] = -RuleSub.SEP_LEN 

                shift_sep_mask_text = array.array('i', accumulate(shift_sep_mask_text))

                shift_mask_text = [0] * text_obj_len
                i = 0
                while i < text_obj_len:
                    next_original = next((j for j, m in enumerate(text_obj.mask[i:], start=i) if m != 0), None)
                    if next_original == None: break;

                    i = next((j for j, m in enumerate(text_obj.mask[next_original:], start=next_original) if m == 0), None)
                    if i == None: break;
                    
                    shift_mask_text[i] = i - next_original

                shift_mask_text = accumulate(shift_mask_text)

                shift_mask_text = [m for m, b in zip(shift_mask_text, text_obj.mask) if b == 0]
                sub_text = matcher.sub(text_str, new_val)
                sub_text.reverse()

                for start, length, value in sub_text:
                    start += shift_sep_mask_text[start]
                    start += shift_mask_text[start]
                    
                    text_obj.text = text_obj.text[:start] + array.array("u", value) + text_obj.text[start + length :]
                    
                    text_obj.mask = (
                        text_obj.mask[:start]
                        + array.array("i", [sub_order] * len(value))
                        + text_obj.mask[start + length :]
                    )
            
            
            rules = [rule[1] for rule in matching_rules]
            matching_rules = [
                (match[0], rule, match[1], match[2])
                for rule, match in zip(
                    rules,
                    map(lambda rule: rule.match(desc, code, namespace), rules),
                )
                if match
            ]

        return (desc.show(), code.show())
    

def do_batch(dataset_ruleset: Tuple[List[Entry], RuleSet]):
    dataset, ruleset = dataset_ruleset
    expensions = [ruleset.expand(entry.objective, entry.command, entry.command_name) for entry in dataset]
    return [
        Entry(
            objective=objective,
            command=command,
            command_name=entry.command_name,
            description=entry.description,
            syntax=entry.syntax,
            flags=entry.flags
      ) for entry, (objective, command) in zip(dataset, expensions)
    ]

def ruleset_expand(base_dataset, args):
    with open(RULESET, "r") as f:
        ruleset_json = json.load(f)

    ruleset = RuleSet(ruleset_json)

    result_dataset = Dataset()
    size = args.expand
    total = 0
    with tqdm(
        total=size,
        leave=True,
        desc="Ruleset expand",
    ) as pbar:
        extended_dataset = base_dataset.to_list() * 4
        N = multiprocessing.cpu_count()
        with Pool(N) as pool:
            futures = pool.map_async(do_batch, [(extended_dataset, ruleset)] * N)
            while len(result_dataset) < size:
                results_list = futures.get()
                futures = pool.map_async(do_batch, [(extended_dataset, ruleset)] * N * 2)
                for results in results_list:
                    total += len(results)
                    remaining = max(size - len(result_dataset), 0)
                    inserted = result_dataset.add_entries(results)
                    pbar.update(min(inserted, remaining))
            futures.wait(0)
        # Sigle threaded:

        # while len(result_dataset) < size:
        #     results = do_batch((extended_dataset, ruleset))
        #     total += len(results)
        #     remaining = max(size - len(result_dataset), 0)
        #     inserted = result_dataset.add_entries(results)
        #     pbar.update(min(inserted, remaining))
   
    if args.verbose:
        print(f"Total: {total} tries")

    return result_dataset
