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


# Matching: ''
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

    def match(self, text):
        return (ExactMatcher.PRIO, self) if self.pattern in text else None

    def sub(self, text, new_val):
        start_positions = (match.start() for match in re.finditer(self.pattern, text))
        return [(s, len(self.pattern), new_val) for s in start_positions]


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
        # TODO(Kristofy): This does not check for overlapping words:
        # so 'banana' with a pattern ana will match twice, but
        # the two matching parts are overlapping in the middle 'a' string
        # it is a rare case, so it's low priority

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
        self.sub_patterns = [p.strip() for p in pattern.split(",")]

    def match(self, text):
        valid = ((m.start() for m in re.finditer(pattern, text)) for pattern in self.sub_patterns)

        latest = -1
        for start_indexes in valid:
            latest = next((ind for ind in start_indexes if ind > latest), None)
            if not latest:
                return None

        return (ListMatcher.PRIO, self)

    def sub(self, text, new_vals):
        valid = ((m.start() for m in re.finditer(pattern, text)) for pattern in self.sub_patterns)

        latest = -1
        positions = []

        # TODO(kristofy): this does not check for overlapping words in the sequence
        # So [apple, lemma] in 'appbananalemma lemma' would both match in applemma instead
        # of the expected matching with the two separate words
        for start_indexes in valid:
            latest = next((ind for ind in start_indexes if ind > latest), None)
            if not latest:
                raise RuntimeError("The pattern sould match when sub is called")
            positions.append(latest)

        new_vals = (v.strip() for v in new_vals.strip()[1:-1].split(","))
        return [(s, len(p), v) for s, p, v in zip(positions, self.sub_patterns, new_vals)]


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
                raise ValueError(f"Invalid rule {self.rule}")

            self.namespace, rule = rule.split("#")
            self.namespace = self.namespace.strip()

        rule = rule.strip()
        if ":" not in rule or len(rule.split(":")) != 2:
            raise ValueError(f"Invalid rule {self.rule}")

        desc_rule, code_rule = rule.split(":")

        self.values = [(s["d-sub"], s["c-sub"]) for s in values]
        self.desc_matcher = Matcher.create(desc_rule)
        self.code_matcher = Matcher.create(code_rule)

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

    # @benchmark
    def expand(self, desc, code, namespace=""):
        desc = RuleSub(desc)
        code = RuleSub(code)

        # The masks are still empty
        sub_key = (desc.text.tobytes(), code.text.tobytes())

        # Ceching based on only the text, because at the beginning the mask is always empty
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


def do_batch(dataset_ruleset):
    dataset, ruleset = dataset_ruleset

    expanded_dataset = [
        (
            ruleset.expand(example["description"], example["code"], data["command"].strip()),
            (
                data["command"],
                data["description"],
                data["syntax"],
                data["flags"],
            ),
        )
        for data in dataset
        for example in data["examples"]
    ]

    results = [
        {
            "example_description": expansion[0],
            "example_code": expansion[1],
            "command": command,
            "description": desc,
            "syntax": syntax,
            "flags": flags,
        }
        for expansion, (command, desc, syntax, flags) in expanded_dataset
    ]

    hashes = [hash((data[0], expansion[0], expansion[1])) for expansion, data in expanded_dataset]

    return results, hashes


def ruleset_expand(dataset, size):
    with open(RULESET, "r") as f:
        ruleset_json = json.load(f)

    ruleset = RuleSet(ruleset_json)

    result_dataset = list()
    result_hashes = set()

    total = 0
    with tqdm(
        total=size,
        leave=True,
        desc="Ruleset expand",
    ) as pbar:
        
        extended_dataset = dataset * 8

        N = multiprocessing.cpu_count()
        with Pool(N) as pool:
            futures = pool.map_async(do_batch, [(extended_dataset, ruleset)] * N)
            while len(result_dataset) < size:
                results_list = []

                @benchmark
                def waiting():
                    nonlocal results_list
                    results_list = futures.get()
                waiting()

                futures = pool.map_async(do_batch, [(extended_dataset, ruleset)] * N * 4)
                @benchmark
                def process_results():
                    nonlocal total
                    for results, hashes in results_list:
                        total += len(results)
                        for result, h in zip(results, hashes):
                            if len(result_dataset) < size and h not in result_hashes:
                                result_dataset.append(result)
                                result_hashes.add(h)
                                pbar.update(1)
                process_results()
            futures.wait(0)
        # Sigle threaded:

        # def do(data):
        #     data = dict(data)
        #     data["example_description"], data["example_code"] = ruleset.expand(
        #         data["example_description"],
        #         data["example_code"],
        #         data["command"].strip(),
        #     )
        #     data_hash = hash((data["command"], data["example_description"], data["example_code"]))
        #     return data, data_hash

        # while len(result_dataset) < size:
        #     for data in dataset:
        #         command = data["command"]
        #         description = data["description"]
        #         syntax = data["syntax"]
        #         flags = data["flags"]
        #         examples = data["examples"]

        #         for example in examples:
        #             example_description = example["description"]
        #             example_code = example["code"]

        #             entry, entry_hash = do(
        #                 {
        #                     "example_description": example_description,
        #                     "example_code": example_code,
        #                     "command": command,
        #                     "description": description,
        #                     "syntax": syntax,
        #                     "flags": flags,
        #                 }
        #             )

        #             if entry_hash not in result_hashes and len(result_dataset) < size:
        #                 result_hashes.add(entry_hash)
        #                 result_dataset.append(entry)
        #                 pbar.update(1)

    return result_dataset
