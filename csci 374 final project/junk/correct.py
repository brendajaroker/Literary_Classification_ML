import pandas as pd
import requests
import logging
from thefuzz import fuzz
from urllib.parse import quote
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load input files
try:
    prose_df = pd.read_csv("data/wrong_ids.csv")
except FileNotFoundError:
    logger.error("data/prose_data.csv not found. Please ensure the file exists.")
    exit(1)

try:
    verification_df = pd.read_csv("title_verification_results.csv")
except FileNotFoundError:
    logger.error("title_verification_results.csv not found. Please run verify_gutenberg_titles.py first.")
    exit(1)

# Function to search Gutendex for ID
def search_gutendex_id(title, author):
    query = f"{quote(title)}"
    if author and author != "Anonymous":
        query += f" {quote(author)}"
    search_url = f"https://gutendex.com/books/?search={query}"
    try:
        response = requests.get(search_url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"Gutendex search failed for {title} by {author}: Status code {response.status_code}")
            return None, None
        
        data = response.json()
        results = data.get("results", [])
        if not results:
            logger.warning(f"No Gutendex results for {title} by {author}")
            return None, None
        
        best_id, best_title, best_score = None, None, 0
        for result in results:
            found_title = result.get("title", "")
            found_authors = [a.get("name", "") for a in result.get("authors", [])]
            score = fuzz.ratio(title.lower(), found_title.lower())
            if author and author != "Anonymous" and any(fuzz.ratio(author.lower(), fa.lower()) > 70 for fa in found_authors):
                if score > best_score and score >= 70:
                    best_id = result.get("id")
                    best_title = found_title
                    best_score = score
            elif author == "Anonymous" and score > best_score and score >= 70:
                best_id = result.get("id")
                best_title = found_title
                best_score = score
        
        if best_id:
            return str(best_id), best_title
        return None, None
    except requests.RequestException as e:
        logger.warning(f"Gutendex search failed for {title} by {author}: {e}")
        return None, None

# Function to validate Gutenberg ID
def validate_gutenberg_id(gid, title):
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.text) > 100 and "Project Gutenberg" in response.text:
                return True, url
            logger.warning(f"Validation failed for {title} (ID {gid}) at {url}")
        except requests.RequestException as e:
            logger.warning(f"Validation failed for {title} (ID {gid}) at {url}: {e}")
    return False, None

# Process entries
corrected_rows = []

for idx, row in prose_df.iterrows():
    title = row['Title']
    author = row['Author']
    movement = row['Movement']
    original_id = row['GutenbergID']
    is_prose = row['IsProse']
    
    logger.info(f"Processing {title} by {author} (ID {original_id})...")
    
    # Check verification results
    ver_row = verification_df[verification_df['Expected_Title'] == title]
    if not ver_row.empty:
        ver_row = ver_row.iloc[0]
        status = ver_row['Status']
        actual_title = ver_row['Actual_Title']
        similarity = ver_row['Similarity']
        
        if status == 'Match':
            corrected_rows.append({
                'Title': title,
                'Author': author,
                'Movement': movement,
                'GutenbergID': original_id,
                'IsProse': is_prose
            })
            logger.info(f"✅ Kept ID {original_id} for {title} (Match, Similarity: {similarity}%)")
        else:
            # Handle Mismatch, No Title Found, Invalid ID
            logger.info(f"⚠️ {status} for {title}: Searching for correct ID...")
            new_id, found_title = search_gutendex_id(title, author)
            if new_id:
                is_valid, url = validate_gutenberg_id(new_id, title)
                if is_valid:
                    corrected_rows.append({
                        'Title': title,
                        'Author': author,
                        'Movement': movement,
                        'GutenbergID': new_id,
                        'IsProse': is_prose
                    })
                    logger.info(f"✅ Corrected ID to {new_id} for {title} (Found: {found_title})")
                else:
                    logger.error(f"❌ Invalid new ID {new_id} for {title}")
            else:
                logger.error(f"❌ No ID found for {title} by {author}")
    else:
        # Not in verification results; search for ID
        logger.info(f"⚠️ {title} not found in verification results. Searching for ID...")
        new_id, found_title = search_gutendex_id(title, author)
        if new_id:
            is_valid, url = validate_gutenberg_id(new_id, title)
            if is_valid:
                corrected_rows.append({
                    'Title': title,
                    'Author': author,
                    'Movement': movement,
                    'GutenbergID': new_id,
                    'IsProse': is_prose
                })
                logger.info(f"✅ Corrected ID to {new_id} for {title} (Found: {found_title})")
            else:
                logger.error(f"❌ Invalid new ID {new_id} for {title}")
        else:
            logger.error(f"❌ No ID found for {title} by {author}")
    
    time.sleep(2)  # Avoid rate-limiting

# Save corrected dataset
corrected_df = pd.DataFrame(corrected_rows, columns=['Title', 'Author', 'Movement', 'GutenbergID', 'IsProse'])
corrected_df.to_csv('data/correct_wrong.csv', index=False)

logger.info(f"\nSummary:")
logger.info(f"Total entries processed: {len(prose_df)}")
logger.info(f"Corrected entries: {len(corrected_rows)}")
logger.info(f"Corrected dataset saved to 'data/prose_data_final.csv'")