import pandas as pd
import requests
import re
import logging
from thefuzz import fuzz
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load prose_data.csv
try:
    df = pd.read_csv("data/data.csv")
except FileNotFoundError:
    logger.error("data/prose_data_corrected.csv not found. Please run validate_gutenberg_ids.py first.")
    exit(1)

# Function to fetch text and extract title
def get_gutenberg_title(gid, expected_title):
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.text) > 100 and "Project Gutenberg" in response.text:
                # Extract title from header
                title_match = re.search(r'^Title:\s*(.+)$', response.text, re.MULTILINE)
                if title_match:
                    return True, title_match.group(1).strip(), url, len(response.text)
                else:
                    logger.warning(f"No title found in header for ID {gid} at {url}")
                    return True, None, url, len(response.text)
            else:
                logger.warning(f"Failed to fetch {url}: Status code {response.status_code or 'empty response'}")
        except requests.RequestException as e:
            logger.warning(f"Failed to fetch {url}: {e}")
        time.sleep(1)  # Avoid rate-limiting
    return False, None, None, 0

# Function to compare titles using fuzzy matching
def titles_match(expected, actual):
    if actual is None:
        return False, 0
    # Normalize titles: remove punctuation, lowercase, strip subtitles
    def normalize_title(title):
        title = re.sub(r'[^\w\s]', '', title.lower())
        title = title.split(' or ')[0].split(' a ')[0].split(' being ')[0]  # Remove subtitles
        return title.strip()
    
    expected_norm = normalize_title(expected)
    actual_norm = normalize_title(actual)
    score = fuzz.ratio(expected_norm, actual_norm)
    return score >= 80, score  # 80% similarity threshold

# Check for potential non-prose texts
non_prose_keywords = ['poems', 'poetry', 'play', 'tragedy', 'comedy', 'sonnets', 'epithalamion']

def flag_non_prose(title):
    title_lower = title.lower()
    return any(keyword in title_lower for keyword in non_prose_keywords)

# Verify titles
results = []
non_prose_warnings = []
short_text_warnings = []

for _, row in df.iterrows():
    gid = row['GutenbergID']
    title = row['Title']
    author = row['Author']
    movement = row['Movement']
    
    logger.info(f"Checking {title} (ID {gid})...")
    is_valid, actual_title, url, text_length = get_gutenberg_title(gid, title)
    
    if is_valid:
        if actual_title:
            matches, similarity = titles_match(title, actual_title)
            status = 'Match' if matches else 'Mismatch'
            results.append({
                'GutenbergID': gid,
                'Expected_Title': title,
                'Actual_Title': actual_title,
                'Author': author,
                'Movement': movement,
                'Status': status,
                'Similarity': similarity,
                'URL': url,
                'Text_Length': text_length
            })
            logger.info(f"{'✅' if matches else '❌'} {status} for ID {gid}: Expected '{title}', Got '{actual_title}' (Similarity: {similarity}%)")
        else:
            results.append({
                'GutenbergID': gid,
                'Expected_Title': title,
                'Actual_Title': 'Unknown',
                'Author': author,
                'Movement': movement,
                'Status': 'No Title Found',
                'Similarity': 0,
                'URL': url,
                'Text_Length': text_length
            })
            logger.error(f"❌ No title found for ID {gid} at {url}")
    else:
        results.append({
            'GutenbergID': gid,
            'Expected_Title': title,
            'Actual_Title': 'N/A',
            'Author': author,
            'Movement': movement,
            'Status': 'Invalid ID',
            'Similarity': 0,
            'URL': None,
            'Text_Length': 0
        })
        logger.error(f"❌ Invalid ID {gid} for {title}")
    
    # Flag potential non-prose texts
    if flag_non_prose(title):
        non_prose_warnings.append((gid, title, author, movement))
        logger.warning(f"⚠️ {title} (ID {gid}) may not be prose (possible play/poetry).")
    
    # Flag short texts
    if is_valid and text_length < 1000:  # Arbitrary threshold for very short texts
        short_text_warnings.append((gid, title, author, movement, text_length))
        logger.warning(f"⚠️ {title} (ID {gid}) is very short ({text_length} characters). May be skipped by text_features.py.")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('title_verification_results.csv', index=False)

non_prose_df = pd.DataFrame(non_prose_warnings, columns=['GutenbergID', 'Title', 'Author', 'Movement'])
non_prose_df.to_csv('non_prose_warnings.csv', index=False)

short_text_df = pd.DataFrame(short_text_warnings, columns=['GutenbergID', 'Title', 'Author', 'Movement', 'Text_Length'])
short_text_df.to_csv('short_text_warnings.csv', index=False)

logger.info(f"\nSummary:")
logger.info(f"Total IDs checked: {len(df)}")
logger.info(f"Matches: {len(results_df[results_df['Status'] == 'Match'])}")
logger.info(f"Mismatches: {len(results_df[results_df['Status'] == 'Mismatch'])}")
logger.info(f"No Title Found: {len(results_df[results_df['Status'] == 'No Title Found'])}")
logger.info(f"Invalid IDs: {len(results_df[results_df['Status'] == 'Invalid ID'])}")
logger.info(f"Potential non-prose texts: {len(non_prose_warnings)}")
logger.info(f"Short texts: {len(short_text_warnings)}")
logger.info(f"Results saved to 'title_verification_results.csv'")
logger.info(f"Non-prose warnings saved to 'non_prose_warnings.csv'")
logger.info(f"Short text warnings saved to 'short_text_warnings.csv'")