"""
webscrap.py

Crawls the METU IE Summer Practice website and extracts content from
HTML pages, PDFs, and DOC/DOCX files. Outputs a simple JSON dataset
with one entry per page/document containing url, title, and raw content.

Chunking, deduplication, and vectorization are handled by vectorisation.py.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin, urlparse
from collections import deque
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
import io

# Optional imports for document parsing
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    from spire.doc import *
    from spire.doc.common import *
except ImportError:
    Document = None

# Set up logging for error handling and progress tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def perform_login(session):
    """
    Attempts to log in to the METU system using credentials from .env via Basic Authentication.
    If no credentials are provided, simply returns False.
    """
    load_dotenv()
    username = os.getenv("METU_USERNAME")
    password = os.getenv("METU_PASSWORD")
    
    if not username or not password or username == "your_username":
        logging.info("No valid METU_USERNAME and METU_PASSWORD found in .env. Proceeding anonymously.")
        return False
        
    logging.info(f"Setting up Basic Authentication for {username}...")
    session.auth = (username, password)
    
    # Test authentication against a protected page
    test_url = "https://sp-ie.metu.edu.tr/en/forms"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        test_response = session.get(test_url, headers=headers)
        if test_response.status_code == 200:
            logging.info("Login successful (Basic Authentication)!")
            return True
        elif test_response.status_code == 401:
            logging.error("Login failed! Incorrect username or password for Basic Auth.")
            return False
        else:
            logging.warning(f"Unexpected status code {test_response.status_code} during login check.")
            return False
            
    except Exception as e:
        logging.error(f"Login setup failed due to an error: {e}")
        return False

def fetch_html(url, session):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch HTML {url}: {e}")
        return None

def clean_raw_text(text):
    """
    Clean raw text from any source (HTML, PDF, DOC).
    Removes form template noise and normalizes whitespace.
    """
    if not text:
        return ""
    # Remove form template noise
    text = re.sub(r'\.{3,}', '', text)           # "......." patterns
    text = re.sub(r'_{3,}', '', text)             # "________" patterns
    # Normalize whitespace
    text = re.sub(r'[\r\t\f\v ]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

def _clean_document_title(filename):
    """Convert document filenames like 'sp_application_form_ie400_eng_0.pdf' to readable titles."""
    if not any(filename.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
        return filename
    
    name = filename.rsplit('.', 1)[0]           # remove extension
    name = re.sub(r'_\d+$', '', name)           # remove version suffix (_0, _1)
    name = name.replace('_', ' ').replace('-', ' ')
    name = name.title()
    
    # Fix known abbreviations to uppercase
    name = re.sub(r'\bIe\s*(\d)', r'IE \1', name)   # "Ie 300" → "IE 300"
    name = re.sub(r'\bIe(\d)', r'IE \1', name)       # "Ie300" → "IE 300"
    name = re.sub(r'\bSp\b', 'SP', name)
    name = re.sub(r'\bSgk\b', 'SGK', name)
    name = re.sub(r'\bOhs\b', 'OHS', name)
    name = re.sub(r'\bEng\b', '(EN)', name)
    name = re.sub(r'\bTr\b', '(TR)', name)
    
    return name

def fetch_pdf_text(url, session):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if not pypdf:
            logging.warning(f"pypdf not installed. Cannot read {url}")
            return ""
            
        pdf_file = io.BytesIO(response.content)
        reader = pypdf.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return clean_raw_text(text)
    except Exception as e:
        logging.error(f"Failed to read PDF {url}: {e}")
        return ""

def fetch_docx_text(url, session):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if not docx:
            logging.warning(f"python-docx not installed. Cannot read {url}")
            return ""
            
        docx_file = io.BytesIO(response.content)
        doc = docx.Document(docx_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return clean_raw_text(text)
    except Exception as e:
        logging.error(f"Failed to read DOCX {url}: {e}")
        return ""

def fetch_doc_text(url, session):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        if Document is None:
            logging.warning(f"Spire.Doc not installed. Cannot read {url}")
            return ""
            
        # Spire.Doc requires reading from a file object path on disk.
        temp_path = "temp_download.doc"
        with open(temp_path, "wb") as f:
            f.write(response.content)
            
        doc = Document()
        doc.LoadFromFile(temp_path)
        text = doc.GetText()
        doc.Close()
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        text = text.replace("Evaluation Warning: The document was created with Spire.Doc for Python.", "")
        return clean_raw_text(text)
    except Exception as e:
        logging.error(f"Failed to read DOC {url}: {e}")
        return ""

def clean_data(soup):
    """Extract and clean text content from an HTML page."""
    if not soup:
        return ""
    
    # Removed 'aside' to prevent data loss in important sidebars
    tags_to_remove = ['nav', 'footer', 'header', 'script', 'style', 'noscript', 'meta', 'link']
    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            element.decompose()
            
    noise_classes = ['links', 'share-links', 'social', 'language-switcher', 'date-display-single', 'language', 'social-media']
    for cls in noise_classes:
        for element in soup.find_all(class_=cls):
            element.decompose()
            
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('div', id='content') or soup.find('div', class_='region-content')
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        text = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)
    
    return clean_raw_text(text)

def get_all_internal_links(base_url, session):
    visited = set()
    queue = deque([base_url])
    all_links = set([base_url])
    base_domain = urlparse(base_url).netloc
    base_path = urlparse(base_url).path

    while queue:
        current_url = queue.popleft()
        if current_url in visited:
            continue
            
        visited.add(current_url)
        logging.info(f"Discovering links on: {current_url}")
        
        soup = fetch_html(current_url, session)
        if not soup:
            continue
            
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                continue
                
            full_url = urljoin(current_url, href)
            full_url = full_url.split('#')[0]
            
            parsed_url = urlparse(full_url)
            
            if parsed_url.netloc == base_domain and parsed_url.path.startswith(base_path):
                # We skip obviously non-text extensions, but keep pdf and doc/docx
                if not full_url.lower().endswith(('.xls', '.xlsx', '.zip', '.rar', '.jpg', '.png', '.jpeg')):
                    if full_url not in all_links:
                        all_links.add(full_url)
                        
                        # Only append HTML pages to the queue, prevent fetching HTML tags from a PDF!
                        if not full_url.lower().endswith(('.pdf', '.doc', '.docx')):
                            queue.append(full_url)
                        
    return list(all_links)

def main():
    base_url = "https://sp-ie.metu.edu.tr/en"
    session = requests.Session()
    
    # Add retry strategy for robustness against transient network errors
    retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 0. Attempt Login using environment variables
    perform_login(session)
    
    logging.info(f"Starting crawl to find all pages under: {base_url}")
    
    # 1. Discover all internal pages
    urls_to_scrape = get_all_internal_links(base_url, session)
    logging.info(f"Found {len(urls_to_scrape)} unique pages/documents to scrape.")
    
    final_dataset = []
    
    # 2. Iterate over ALL found pages and documents and extract content
    for url in urls_to_scrape:
        logging.info(f"Scraping content from: {url}")
        
        page_title = url.split('/')[-1] if not url.endswith('/') else url.split('/')[-2]
        cleaned_text = ""
        
        # For document files, convert filename to a human-readable title
        if any(url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx']):
            page_title = _clean_document_title(page_title)
        
        if url.lower().endswith('.pdf'):
            cleaned_text = fetch_pdf_text(url, session)
        elif url.lower().endswith('.docx'):
            cleaned_text = fetch_docx_text(url, session)
        elif url.lower().endswith('.doc'):
            cleaned_text = fetch_doc_text(url, session)
        else:
            soup = fetch_html(url, session)
            if soup:
                title_tag = soup.title
                page_title = title_tag.string.strip() if title_tag and title_tag.string else page_title
                cleaned_text = clean_data(soup)
        
        if cleaned_text:
            final_dataset.append({
                "url": url,
                "title": page_title,
                "content": cleaned_text
            })
        else:
            logging.warning(f"No content extracted from {url}")

    # 2.5. Deduplicate entries (by normalized URL and by exact content)
    # URL normalization strips version suffixes (_0, _1) from document filenames
    # so ie300-manual_0.pdf and ie300-manual_1.pdf are caught as duplicates,
    # while IE 300 vs IE 400 variants (different base names) are preserved.
    seen_urls = set()
    seen_content = set()
    deduped = []
    for entry in final_dataset:
        url_key = entry['url'].rstrip('/')
        # Normalize versioned document URLs: strip _0, _1 etc. before extension
        url_key = re.sub(r'_\d+\.(pdf|doc|docx)$', r'.\1', url_key)
        content_key = hash(entry['content'].strip())
        if url_key in seen_urls or content_key in seen_content:
            logging.info(f"Removed duplicate: {entry['title']}")
            continue
        seen_urls.add(url_key)
        seen_content.add(content_key)
        deduped.append(entry)
    
    removed = len(final_dataset) - len(deduped)
    if removed > 0:
        logging.info(f"Deduplication removed {removed} entries.")
    final_dataset = deduped

    # 3. Save to JSON file locally
    output_filename = "metu_ie_chatbot_dataset.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully scraped {len(final_dataset)} pages/documents to {output_filename}")
    except IOError as e:
        logging.error(f"Failed to save JSON file: {e}")

if __name__ == "__main__":
    main()