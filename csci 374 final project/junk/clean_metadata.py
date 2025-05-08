import pandas as pd
import re

# Load dataset
df = pd.read_csv("metadata.csv")

# Function to assign literary movement based on pub_year
def get_movement(pub_year):
    try:
        year = int(pub_year)
    except:
        return "Unknown"
    
    if year <= 1660:
        return "Renaissance"
    elif 1780 <= year <= 1830:
        return "Romanticism"
    elif 1837 <= year <= 1901:
        return "Victorian"
    elif 1860 <= year <= 1910:
        return "Realism"
    elif 1900 <= year <= 1945:
        return "Modernism"
    elif 1945 <= year <= 1980:
        return "Postmodernism" #there are none on gutenberg
    elif year > 1980:
        return "Contemporary"
    else:
        return "Unknown"

# Function to extract Gutenberg ID from URL
def extract_gutenberg_id(url):
    if pd.isna(url) or not isinstance(url, str):
        return ""
    match = re.search(r'/epub/(\d+)/', url)
    if match:
        return match.group(1)
    return ""

# Filter and rename necessary columns
df_filtered = df[["title", "author", "pub_year", "pg_eng_url"]].copy()
df_filtered.columns = ["Title", "Author", "PubYear", "GutenbergURL"]

# Assign movement and extract ID
df_filtered["Movement"] = df_filtered["PubYear"].apply(get_movement)
df_filtered["GutenbergID"] = df_filtered["GutenbergURL"].apply(extract_gutenberg_id)

# Drop rows with empty Gutenberg ID
df_filtered = df_filtered[df_filtered["GutenbergID"] != ""]

# Drop duplicates based on Gutenberg ID
df_filtered = df_filtered.drop_duplicates(subset="GutenbergID")

# Final selection of columns
df_final = df_filtered[["Title", "Author", "Movement", "GutenbergID"]]

# Save to CSV
df_final.to_csv("edited_metadata.csv", index=False)
print("Saved to edited_metadata.csv")