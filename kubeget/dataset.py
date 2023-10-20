import json
from typing import List, Optional, Set
from dataclasses import dataclass

@dataclass
class Entry:
    objective:str
    command_name:str
    command:str
    description:str
    syntax:str
    flags:str
    question: Optional[str] = None
    chain_of_thought: Optional[str] = None

    def dict(self):
        return {
            "objective": self.objective,
            "command_name": self.command_name,
            "command": self.command,
            "description": self.description,
            "syntax": self.syntax,
            "flags": self.flags,
            "question": self.question,
            "chain_of_thought": self.chain_of_thought
        }
    
    def __hash__(self):
        return hash((self.command_name, self.objective, self.command, self.question, self.chain_of_thought))

    
    def get_context(self):
        """ Queries the meaningful part of each command (not sematically) """

        command = self.command
        if command.startswith('kubectl '):
            command = command[len('kubectl '):]

        return command
       
class Dataset:
    def __init__(self, entries: List[Entry] = None, hashes: Set[int] = None):
        self.entries = entries or []
        self.hashes = hashes or set()

    def add_entry(self, entry: Entry):
        entry_hash = hash(entry)
        if entry_hash not in self.hashes:
            self.hashes.add(entry_hash)
            self.entries.append(entry)
            return True
        return False

    def add_entries(self, entries: List[Entry]):
        # reaturn the number of true
        return sum(self.add_entry(entry) for entry in entries)

    def load(self, file_path: str):
        with open(file_path, 'r') as file:
            data = json.load(file)
            self.entries = [Entry(**entry) for entry in data]
            self.hashes = set(hash(entry) for entry in self.entries)

    def dump(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump([entry.dict() for entry in self.entries], file)

    def copy(self):
        """ Copy the entries and the hashes with a deep copy """
        return Dataset(
            entries=[Entry(**entry.__dict__) for entry in self.entries],
            hashes=set(self.hashes)
        )

    def to_list(self):
        """ Returns a list of Entry objects, with a deep copy of the entries """
        return [Entry(**entry.__dict__) for entry in self.entries]

    def __iter__(self):
        return iter(self.entries)
    
    def __getitem__(self, index: int):
        return self.entries[index]
    
    def __len__(self):
        return len(self.entries)
    