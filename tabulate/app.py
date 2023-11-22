from dataclasses import dataclass
import json
import sys
from typing import Any, List, Tuple
from PyPDF2 import PdfReader
from tqdm import tqdm
from network import batch_request
from utils import cached
import re
import time

from transformers import AutoTokenizer
from openai import OpenAI
import os
from dotenv import load_dotenv
import pinecone

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")

yamls = []
scripts = []
lora_model_id = 'ComponentSoft/mistral-kubectl-instruct'
tokenizer = AutoTokenizer.from_pretrained(
    lora_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

SOFT_CAP = 512

@dataclass
class SubPart:
    title: str
    content: List[str]

    def dict(self):
        return {
            "title": self.title,
            "content": self.content,
        }

    def __str__(self):
        return f"SubPart: {self.title}\n{self.content}\n"

@dataclass
class Part:
    title: str
    intr: List[str]
    subparts: List[SubPart]

    def dict(self):
        if self.title == "More Information":
            return None
        return {
            "title": self.title,
            "intr": self.intr,
            "subparts": [s.dict() for s in self.subparts],
        }

    def __str__(self):
        if self.title == "More Information":
            return ""
        return f"Part: {self.title}\n{self.intr}\n"+"\n".join(str(s) for s in self.subparts) + "\n"

@dataclass
class Chapter:
    title: str
    intr: List[str]
    parts: List[Part]

    def dict(self):
        return {
            "title": self.title,
            "intr": self.intr,
            "parts": [p.dict() for p in self.parts],
        }
    
    def __str__(self):
        return f"Chapter: {self.title}\n{self.intr}\n" + "\n".join(str(p) for p in self.parts) + "\n"

@cached
def text_extractor(path):
    with open(path, 'rb') as f:
        pdf = PdfReader(f)
        text = ""
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text += page.extract_text() + '\n'
    return text

def dedupe_spaces(text):
    return re.sub(" +", " ", text.strip())

def strip_left_space(text):
    return '\n'.join(t.rstrip() for t in text.split('\n'))

def process_content(content):
    # Re wrap the lines where it is obvious that the new line is only there because of text wrapping
    content = content.strip()
    content = list(content+" "*3)

    active = True
    while active:
        active = False
        new_line_indices = [index for index, char in enumerate(content) if char == '\n'][::-1]
        for i in new_line_indices:
              l = content[i - 1].isalnum() or content[i - 1] in ".,?!):"
              r = content[i + 1].isalnum() or content[i + 1] in ".,?!(-"
              if l and r:
                  content[i] = ' '
                  active = True

    hypen_indices = [index for index, char in enumerate(content) if char == '-'][::-1]
    for i in hypen_indices:
        if content[i-1] == ' ' and (content[i+1] == '\n' or content[i+2] == '\n'):
            if content[i+2] == '\n':
              content.pop(i+2)

            if content[i+1] == '\n':
              content.pop(i+1)
                
            content[i] = ' '

    content = ''.join(content)

    # Remove chapter references
    content = re.sub(r'\([^\n(]*Chapter\s+\d+[^\n)]*\)', '', content)
    content = re.sub(r'(?:(?<=[.:;?!])|(?<=^))[^\n.:;?!,]*Chapter\s+\d+[^\n.,:;?!]*[,.:;?!](?:\s*"[^\n"]*")?', '', content)
    content = re.sub(r',[^\n.:;?!,]*Chapter\s+\d+[^\n.,:;?!]*(?:\s*"[^\n"]*")?', '', content)


    # Filter out left in page numbers

    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
        
    content = dedupe_spaces(content)
    content = strip_left_space(content)
    
    
    for i, yaml in enumerate(yamls):
        uid = f"_YAML${i}_"
        content = content.replace(f'\n{uid}', f' {uid}')

    for i, script in enumerate(scripts):
        uid = f"_CODE${i}_"
        content = content.replace(f"\n{uid}\n", f' {uid} ')

    chunks = [line for line in content.split('\n') if line.strip() != '']

    for i, yaml in enumerate(yamls):
        uid = f"_YAML${i}_"
        for j, chunk in enumerate(chunks):
            chunks[j] = chunk.replace(uid, f'\n```yaml\n{yaml}\n```\n')
            
    for i, script in enumerate(scripts):
        uid = f"_CODE${i}_"
        for j, chunk in enumerate(chunks):
            chunks[j] = chunk.replace(uid, f'\n```bash\n{script}\n```\n')

    for j, chunk in enumerate(chunks):
        chunks[j] = strip_left_space(chunk)

    # first
    parts = []
    chunks = [(len(tokenizer(chunk).input_ids), chunk) for chunk in chunks]

    running_sum = 0
    end_index = 0
    start_index = 0
    while end_index < len(chunks) and running_sum < SOFT_CAP:
        running_sum += chunks[end_index][0]
        end_index += 1

    parts.append('\n'.join(chunk for _, chunk in chunks[start_index:end_index]))

    STRIDE = 128
    while end_index < len(chunks):
        out = 0
        while out < STRIDE:
            out += chunks[start_index][0]
            start_index += 1

        running_sum -= out

        while end_index < len(chunks) and running_sum < SOFT_CAP:
            running_sum += chunks[end_index][0]
            end_index += 1

        parts.append('\n'.join(chunk for _, chunk in chunks[start_index:end_index]))

    return parts

def process_subpart(title, subpart):
    return SubPart(title, process_content(subpart))

def process_part(toc, part):
    title = toc["title"]
    if len(toc["parts"]) == 0:
        return Part(title, process_content(part), [])

    toc_re = re.compile(r"^\s*(" + '|'.join(t for t in toc["parts"]) + r")\s*$", flags=re.MULTILINE)
    found = [match.group(1) for match in toc_re.finditer(part)]
    
    n = len(found)
    if n != len(toc["parts"]):
        toc_re = re.compile(r"\b(" + '|'.join(t for t in toc["parts"]) + r")\s*$", flags=re.MULTILINE)
        found = [match.group(1) for match in toc_re.finditer(part)]
        n = len(found)
        

    if n != len(toc["parts"]) or found != [t for t in toc["parts"]]:
        print(f"({title})")
        print(f"ERROR splitting subchapters, {n} != {len(toc['parts'])}")
        print("found: ")
        print(toc_re.findall(part))
        print("re:")
        print(r"^\s*(" + '|'.join(t for t in toc["parts"]) + r")\s*$")
        with open(f"{title}.txt", "w") as f:
            f.write(part)
        sys.exit(1)
        return None
    
    intr, *subparts = toc_re.split(part)[::2]

    subparts = [process_subpart(t, p) for t, p in zip(toc["parts"], subparts)]
    
    return Part(title, process_content(intr), subparts)

def process_chapter(toc, chapter):
    chapter = chapter.strip()
    title, content = chapter.split('\n', 1)
    title = title.strip()
    content = content.strip()
    if (title != toc["title"]):
      print(f"ERROR detecting title: {title} - {toc['title']}")
      return None

    toc_re = re.compile(r"\b(" + '|'.join(t["title"] for t in toc["parts"]) + r")\s*$", flags=re.MULTILINE)
    found = [match.group(1) for match in toc_re.finditer(content)]
   
    n = len(found)
    if n != len(toc["parts"]) or found != [t["title"] for t in toc["parts"]]:
        print(f"({title})")
        print(f"ERROR splitting chapters, {n} != {len(toc['parts'])}")
        print("found: ")
        print(toc_re.findall(content))
        print("re:")
        print(r"\b(" + '|'.join(t["title"] for t in toc["parts"]) + r")\s*$")
        return None
    
    intr, *parts = toc_re.split(content)[::2]

    parts = [process_part(t, p.strip()) for t, p in zip(toc["parts"], parts)]
    
    return Chapter(title, process_content(intr), parts)

def cleanup(text):
    with open("original.txt", "w") as f:
        f.write(text)

    # delete page numbers
    text = re.sub(r"^.* \| \d+$\n", '', text,flags=re.MULTILINE)
    text = re.sub(r"^\d+ \| .*$\n", '', text,flags=re.MULTILINE)

    # replace text wrapping unicode - with, remove new line if there is one
    text = re.sub('\u2010\n', '', text)

    # replace text wrapping unicode - with -
    text = re.sub('\u2010', '-', text)

    # replace the unicode unordered list mark with '-'
    text = re.sub('\u2022', ' - ', text)

    # replace long -- to single -
    text = re.sub('\u2014', '-', text)

    # replace unicode aphostrophie with '
    text = re.sub('\u2019', '\'', text)
    
    # replace unicode start and end quote with "
    text = re.sub('\u201c', '"', text)
    text = re.sub('\u201d', '"', text)

    # replace unicode equivalence sign with ascii <=>
    text = re.sub('\u2194', '<=>', text)

    # replace unicode - signifing a numerical range with ascii -
    text = re.sub('\u2013', '<=>', text)

    # replace unicode cross or multiple sign with ascii *
    text = re.sub('\u00d7', '*', text)

    # replace unicode s with accent with s
    text = re.sub('\u0161', 's', text)

    # replace ... (ellipsis) with ... (3 dots)
    text = re.sub('\u2026', 's', text)

    # Delete Figures subtext
    
    text = re.sub(r'^Figure \d+-\d+\..*$', '', text, flags=re.MULTILINE)
    # code_re = re.compile(R"""
    #     ^(\s*)(?:(?:-?\s*[a-zA-Z./\-_ ]+\s*:)|-)\s*\.*\|\s*-?\s*(?:\n\s\1 .*$)+
    # """, re.MULTILINE | re.VERBOSE)
    # check for yaml blocks
    yaml_re = re.compile((
         R'''^(?:'''
          R'''Example.*$\n(?:^$\n)*'''
          R'''|\s*\$.*$\n'''
        R''')'''
        R'''^(?:[a-zA-Z0-9./\-_ ]+\s*:\s*[^|\n]*(?<![|\[])\s*$)'''
        R'''(?:\n^(?:'''
            R'''(?:\s*-?\s*[a-zA-Z0-9./\-_ ]+\s*:\s*[^|\n]*(?<![|\[])\s*)'''
            R'''|---'''
            R'''|\s*(?://\s*)?[.]{3,}s*'''
            R'''|\s+-\s+(?=$|(?:.*[^|\[]\s*$)).*'''
            R'''|\s+-\s+[^\s|]+\s*'''
            R'''|(\s+)(?:(?:-?\s*[a-zA-Z./\-_ ]+\s*:)|-)\s*\.*\|\s*-?\s*(?:\n\s\1 .*$)+'''
            R'''|\s*#.*'''
            R'''|\s*[a-zA-Z0-9./\-_ ]*\s*:\s*\[\s*(?:\n[^\]]*)+\].*'''
        R''')$)*'''
    ), re.MULTILINE )

    bash_re = re.compile(r'(?:^\s*\$[^\\\n]*(?:\\\s*$\n[^\\\n]*)*|^\s*(?:kubectl|docker)[^\\\n]*(?:\\\s*$\n\s+[^\\\n]*)+)', flags=re.MULTILINE)

    for i, match in enumerate(yaml_re.finditer(text)):
        m = match.group(0)

        head, rest = m.split('\n', 1)

        text = text.replace(m, f'{head.strip()}\n_YAML${i}_')
        yamls.append(strip_left_space(rest.strip()))

    for i, match in enumerate(bash_re.finditer(text)):
        m = match.group(0)

        text = text.replace(m, f'_CODE${i}_')
        scripts.append(strip_left_space(m))

    print(f"{len(yamls)} yaml blocks found")
    print(f"{len(scripts)} bash blocks found")

    # remove duplicate spaces
    text = dedupe_spaces(text)

    with open("data.txt", "w") as f:
        f.write(text)

    # split
    chapter_separator = re.compile(r'CHAPTER \d+$', flags=re.MULTILINE)
    chapters = chapter_separator.split(text)[1:]

    with open("toc.json", "r") as f:
        toc = json.load(f)["chapter_metadata"]


    print(f"there are: {len(chapters)} chapters")
    chapters = [process_chapter(t, chapter.strip()) for t, chapter in zip(toc, chapters)]

    return chapters

def apply(chapters: List[Chapter], fun):
    def function(texts: List[str]):
        for text in texts:
            fun(text)

    def apply_subpart(subparts: List[SubPart]):
        for subpart in subparts:
            function(subpart.content)
    
    def apply_part(parts: List[Part]):
        for part in parts:
            function(part.intr)
            apply_subpart(part.subparts)
            
        
    for chapter in chapters:
        function(chapter.intr)
        apply_part(chapter.parts)

def filter_out(chapters: List[Chapter], title):
    res = []
    for ch in chapters:
        ps = [p for p in ch.parts if p.title.strip() != title]
        res.append(Chapter(
            ch.title,
            ch.intr,
            ps
        ))      

    return res

@dataclass
class Data:
    text: str
    embedding: Any

def main():
    text = cleanup(text_extractor("./kubernetes.pdf")) 
    text = filter_out(text, "More Information")

    tokens = []
    apply(text, lambda x: tokens.append(Data(x, None)))
    
    with open("test.txt", "w") as f:
        f.write('\n'.join(str(t) for t in text))
    
    @cached
    def create_embedding(text: str):
        for _ in range(4):  
            try:
                res = openai.embeddings.create(input=text, model="text-embedding-ada-002")
            except Exception as ex:
                print(f"Unknow error: {str(ex)}")
                time.sleep(60)
            else:
                break
        else: 
            raise Exception('Failed to query openai chatcompletion')

        return res.data[0].embedding

    def batch_embedding(data: Data):
        data.embedding = create_embedding(data.text)

    batch_request(batch_embedding, tokens, 12)

    pinecone_name = "k8s"
    pinecone.delete_index(pinecone_name)
    print("Deleted index")
    pinecone.create_index(pinecone_name, dimension=len(tokens[0].embedding), metric="cosine")
    print("Created index")
    pinecone.describe_index(pinecone_name)
    index = pinecone.Index(pinecone_name)

    vectors=[{"id": str(i), "values": entry.embedding, "metadata": {"text": entry.text}} for i, entry in enumerate(tokens)]

    for i in tqdm(range(len(vectors)//10)):
        vecs = vectors[i*10: i*10 + 10]
        index.upsert(vectors=vecs)

    index.describe_index_stats()

if __name__ == '__main__':
    main()
    # text="What is a pod?"
    # res = openai.embeddings.create(input=text, model="text-embedding-ada-002").data[0].embedding
    # with open("test.txt", "w") as f:
    #     f.write(','.join(str(d) for d in res))
