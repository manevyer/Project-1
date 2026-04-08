import requests
from bs4 import BeautifulSoup
import json
import logging
from urllib.parse import urljoin, urlparse
from collections import deque

# Set up logging for error handling and progress tracking
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def fetch_html(url):
    """
    Connects to the given URL using a proper User-Agent header to avoid being blocked.
    Returns the parsed BeautifulSoup object or None if an error occurs.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None

def clean_data(soup):
    """
    Cleans the HTML by removing non-content elements like nav, footer, header, scripts, style, etc.
    Returns the cleaned plain text.
    """
    if not soup:
        return ""
    
    # List of tags to completely remove to clean up the noise
    tags_to_remove = ['nav', 'footer', 'header', 'script', 'style', 'aside', 'noscript', 'meta', 'link']
    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            element.decompose()
            
    # Remove common noise classes (social media, share links, language switchers, dates)
    noise_classes = ['links', 'share-links', 'social', 'language-switcher', 'date-display-single', 'language', 'social-media']
    for cls in noise_classes:
        for element in soup.find_all(class_=cls):
            element.decompose()
            
    # Attempt to find the main content area
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.find('div', id='content') or soup.find('div', class_='region-content')
    
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        text = soup.body.get_text(separator='\n', strip=True) if soup.body else soup.get_text(separator='\n', strip=True)
        
    return text

def chunk_text(text):
    """
    Splits the cleaned text into meaningful chunks without discarding small pieces of information.
    We split by double newlines and merge smaller chunks so we don't lose data.
    """
    raw_chunks = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for raw in raw_chunks:
        raw = raw.strip()
        if not raw:
            continue
        # If the chunk is very small (like a heading or list item), append it to the current chunk
        # Also group up to ~500 characters to make meaningful chatbot responses
        if len(current_chunk) + len(raw) < 500:
            if current_chunk:
                current_chunk += "\n\n" + raw
            else:
                current_chunk = raw
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = raw
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def format_to_json(chunks, source_url, page_title):
    """
    A dummy function mapping extracted chunks to the target JSON schema.
    """
    dataset = []
    
    for i, chunk in enumerate(chunks):
        topic = f"{page_title} - Section {i+1}" if page_title else f"Information from {source_url} - Section {i+1}"
        
        # Simple heuristic to categorize topic based on keyword presence
        lower_chunk = chunk.lower()
        if 'ie 300' in lower_chunk or 'ie300' in lower_chunk:
            topic = "IE 300 Internship Requirements"
        elif 'ie 400' in lower_chunk or 'ie400' in lower_chunk:
            topic = "IE 400 Internship Requirements"
        elif 'document' in lower_chunk or 'form' in lower_chunk:
            topic = "Required Documents and Forms"
        elif 'deadline' in lower_chunk or 'date' in lower_chunk:
            topic = "Deadlines and Important Dates"
        elif 'step' in lower_chunk or 'apply' in lower_chunk or 'application' in lower_chunk:
            topic = "Application Steps and Procedures"
            
        entry = {
            "topic": topic,
            "user_query_variations": [
                f"What is the information regarding {topic.lower()}?",
                f"Can you provide details about {topic.lower()}?",
                f"Tell me about {topic.lower()} from the IE summer practice site."
            ],
            "chatbot_response": chunk
        }
        dataset.append(entry)
        
    return dataset

def get_all_internal_links(base_url):
    """
    Crawls the website starting from base_url to find all internal links.
    Ensures all pages under the given base_url are discovered.
    """
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
        
        soup = fetch_html(current_url)
        if not soup:
            continue
            
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Ignore anchor links, mailto, etc.
            if href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                continue
                
            full_url = urljoin(current_url, href)
            # Remove fragments
            full_url = full_url.split('#')[0]
            
            parsed_url = urlparse(full_url)
            
            # Check if it belongs to the same domain and starts with the base path '/en'
            if parsed_url.netloc == base_domain and parsed_url.path.startswith(base_path):
                # Ensure it's not a file download (like .pdf, .doc) for HTML processing
                if not full_url.lower().endswith(('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar')):
                    if full_url not in all_links:
                        all_links.add(full_url)
                        queue.append(full_url)
                        
    return list(all_links)

def main():
    base_url = "https://sp-ie.metu.edu.tr/en"
    logging.info(f"Starting crawl to find all pages under: {base_url}")
    
    # 1. Discover all internal pages
    urls_to_scrape = get_all_internal_links(base_url)
    logging.info(f"Found {len(urls_to_scrape)} unique pages to scrape.")
    
    final_dataset = []
    
    # 2. Iterate over ALL found pages and extract data
    for url in urls_to_scrape:
        logging.info(f"Scraping content from: {url}")
        
        soup = fetch_html(url)
        if not soup:
            continue
            
        page_title = soup.title.string.strip() if soup.title else ""
        
        cleaned_text = clean_data(soup)
        
        if cleaned_text:
            chunks = chunk_text(cleaned_text)
            json_data = format_to_json(chunks, url, page_title)
            final_dataset.extend(json_data)
        else:
            logging.warning(f"No content extracted from {url}")

    # 3. Save to JSON file locally
    output_filename = "metu_ie_chatbot_dataset.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully scraped all information and saved {len(final_dataset)} entries to {output_filename}")
    except IOError as e:
        logging.error(f"Failed to save JSON file: {e}")

if __name__ == "__main__":
    main()