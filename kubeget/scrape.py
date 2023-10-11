from bs4 import BeautifulSoup, NavigableString, PageElement, Tag
from tqdm import tqdm

import requests
import hashlib
import os

from config import CACHE_DIR

def fetch_and_cache_html(url: str) -> str:
    """Fetch HTML content from a URL and cache it."""
    # Check if the HTML is already cached
    hash_object = hashlib.md5(url.encode())
    hex_hash = hash_object.hexdigest()
    cache_filename: str = os.path.join(CACHE_DIR, f"{hex_hash}.html")
    if os.path.exists(cache_filename):
        with open(cache_filename, "r", encoding="utf-8") as file:
            html_content: str = file.read()
        # print(f"Hit cache for {url}", file=sys.stderr)
    else:
        # Fetch the HTML from the URL
        response = requests.get(url)
        if response.status_code == 200:
            html_content: str = response.text
            # Cache the HTML content
            with open(cache_filename, "w", encoding="utf-8") as file:
                file.write(html_content)

            # print(f"Cache miss {url}", file=sys.stderr)
        else:
            raise Exception(f"Failed to fetch HTML from {url}. Status code: {response.status_code}")

    return html_content

def scrape_page_kubectl(url: str) -> BeautifulSoup:
    """Scrape and process the HTML content of a page."""
    # You can add your scraping logic here
    html_content: str = fetch_and_cache_html(url)

    html_content = html_content.replace('<h2', '<h1').replace('</h2', '</h1')
    # Process the HTML content as needed
    soup: BeautifulSoup = BeautifulSoup(html_content, 'html.parser')

    return soup

def group_elements_by_h1(soup, bs):
    UNIQUE_MARKER = "d41d8cd98f00b204e9800998ecf8427e"
    def group_header(header_tag):
        div = bs.new_tag('div', attrs={"class": UNIQUE_MARKER})
        current_tag = header_tag
        current_tag.insert_before(div)
        
        prev_sibling = current_tag
        next_sibling = current_tag.find_next_sibling()
        div.append(prev_sibling.extract())

        while next_sibling and next_sibling.name and not next_sibling.name.startswith('h1'):
            prev_sibling = next_sibling
            next_sibling = next_sibling.find_next_sibling()
            div.append(prev_sibling.extract())

    headers = soup.select('h1')
    for header in headers:
        group_header(header)

    return soup

  
UNIQUE_MARKER='d41d8cd98f00b204e9800998ecf8427e'
def parse_kubectl():
    bs = scrape_page_kubectl('https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands')
    soup = bs.select_one('#page-content-wrapper')

    for prev in list(soup.select_one('hr').find_previous_siblings()):
        prev.extract()
    soup.select_one('hr').extract()

    for header_element in list(soup.select('h1[id^="-strong"][id$="-"]')):
        header_element.extract()

    soup = group_elements_by_h1(soup, bs)

    dataset = []
    blocks = soup.select(f'div.{UNIQUE_MARKER}')
    print('k8s scrape:')
    for block in tqdm(blocks):
        example_context = block.select('blockquote.example')
        example_code = block.select('blockquote.example+pre>code')
        examples = [{'description': desc.text.strip(), 'code': code.text.strip()} for desc, code in zip(example_context, example_code)]

        description_p = []
        description_next = block.select_one('h1 ~ p')
        while description_next and description_next.get('id', '') != 'usage':
            if description_next.name == 'p':
                description_p.append(description_next.text.strip())
            description_next = description_next.find_next_sibling()

        description = '\n'.join(description_p)
        syntax = block.select_one('#usage + p code').text[2:]
        flags_table = block.select_one('#flags + table tbody')
        if flags_table:
            flags = []
            for row in flags_table.select('tr'):
                flag_name, flag_short, flag_default, flag_usage = [r.text for r in row.select('td')]
                if flag_short:
                    flags.append(f'"flag": "--{flag_name}", "short": "-{flag_short}", "default": "{flag_default}", "usage": "{flag_usage}"')

            flags = '\n'.join(flags)
        else:
            flags = ""

        command = block.select_one('h1').text.strip()

        dataset.append({
            'command': command,
            'description': description,
            'syntax': syntax,
            'examples': examples,
            'flags': flags,
        })

    return dataset