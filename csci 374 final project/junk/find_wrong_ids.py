import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Load input files
try:
    prose_df = pd.read_csv("prose_data.csv")
except FileNotFoundError:
    logger.error("data/prose_data.csv not found. Please ensure the file exists.")
    exit(1)

try:
    final_df = pd.read_csv("prose_data_final.csv")
except FileNotFoundError:
    logger.error("data/prose_data_final.csv not found. Copying all entries from prose_data.csv to wrong_ids.csv.")
    prose_df.to_csv("wrong_ids.csv", index=False)
    logger.info("All entries from prose_data.csv saved to wrong_ids.csv")
    exit(0)

# Identify missing titles
missing_entries = []
final_titles = set(final_df['Title'].str.lower())

for idx, row in prose_df.iterrows():
    title = row['Title']
    if title.lower() not in final_titles:
        missing_entries.append({
            'Title': title,
            'Author': row['Author'],
            'Movement': row['Movement'],
            'GutenbergID': row['GutenbergID'],
            'IsProse': row['IsProse']
        })
        logger.info(f"Title missing from prose_data_final.csv: {title} (ID: {row['GutenbergID']})")

# Save missing entries to wrong_ids.csv
if missing_entries:
    missing_df = pd.DataFrame(missing_entries, columns=['Title', 'Author', 'Movement', 'GutenbergID', 'IsProse'])
    missing_df.to_csv("wrong_ids.csv", index=False)
    logger.info(f"Saved {len(missing_entries)} missing entries to wrong_ids.csv")
else:
    logger.info("No missing entries found. All titles from prose_data.csv are in prose_data_final.csv.")

# Summary
logger.info(f"\nSummary:")
logger.info(f"Total entries in prose_data.csv: {len(prose_df)}")
logger.info(f"Total entries in prose_data_final.csv: {len(final_df)}")
logger.info(f"Missing entries saved to wrong_ids.csv: {len(missing_entries)}")