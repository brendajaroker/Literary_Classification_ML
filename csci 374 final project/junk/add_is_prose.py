import pandas as pd
import requests
import time

# Load your dataset
df = pd.read_csv("edited_metadata.csv")

# Define keywords that signal the book is likely NOT prose
non_prose_terms = ['Poetry', 'Drama', 'Sermons', 'Letters', 'Hymns', 'Songs', 'Meditations']

def is_prose_gutendex(gutenberg_id):
    try:
        url = f"https://gutendex.com/books/{gutenberg_id}"
        response = requests.get(url)
        if response.status_code != 200:
            return None  # API call failed
        
        data = response.json()
        subjects = data.get("subjects", [])
        bookshelves = data.get("bookshelves", [])

        # Check if any known non-prose indicators are present
        combined_tags = subjects + bookshelves
        for tag in combined_tags:
            if any(term.lower() in tag.lower() for term in non_prose_terms):
                return False

        return True  # Otherwise assume prose
    except Exception as e:
        print(f"Error processing ID {gutenberg_id}: {e}")
        return None

# Apply with a slight delay to avoid API throttling
results = []
for idx, row in df.iterrows():
    gid = row["GutenbergID"]
    result = is_prose_gutendex(gid)
    results.append(result)
    time.sleep(0.2)  # To be nice to the API

df["IsProse"] = results

# Save to new CSV
df.to_csv("edited_metadata_with_isprose.csv", index=False)
print("Done! Saved as 'edited_metadata_with_isprose.csv'")
