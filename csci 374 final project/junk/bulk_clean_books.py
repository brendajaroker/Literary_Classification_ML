import os
import pandas as pd
import requests, re
from pathlib import Path

df = pd.read_csv("movement_seed_with_ids.csv")
df = df[df["GutenbergID"].astype(str).str.isdigit()]

os.makedirs("clean_texts", exist_ok=True)

def clean_text(gid, max_words=10000):
    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                text = r.text
                start = re.search(r"\*\*\* START OF.*?\*\*\*", text)
                end = re.search(r"\*\*\* END OF.*?\*\*\*", text)
                body = text[start.end():end.start()] if start and end else text
                return " ".join(body.split()[:max_words])
        except:
            continue
    return None

for _, row in df.iterrows():
    gid = row["GutenbergID"]
    text = clean_text(gid)
    if text:
        fname = f"{row['Movement']}_{row['Author'].replace(' ', '_')}_{gid}.txt"
        with open(os.path.join("clean_texts", fname), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {fname}")
