import requests
import os
import json
import time
import random
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.resolve()
KEYS_FILENAME = "secrets.json"
DATASET_ROOT = "images"
# google limtis image search to 10 results; max 100 queries/day
GOOGLE_SEARCH_LIMIT = 10

def read_keys():
    """ Reads Google API key and Custom Search Engine ID """
    keys_path = SCRIPT_PATH / KEYS_FILENAME

    with open(keys_path, "r") as f:
        secrets = json.load(f)

    API_KEY = secrets["api_key"]
    CSE_ID = secrets["cse_id"]
    
    return API_KEY, CSE_ID

# since Google Images might block IP for web scraping with automated tools,
# we resorted to API-based approach for image retrieval.
def google_image_search(
    class_name,
    query_variants,
    api_key,
    cse_id,
    root_dir=DATASET_ROOT,
    images_per_variant=10,
):
    """ API-based image scraping from Google Images """
    save_dir = SCRIPT_PATH / root_dir / class_name
    os.makedirs(save_dir, exist_ok=True)
    total_downloaded = 0

    for variant in query_variants:
        downloaded = 0
        for start in range(1, images_per_variant + 1, GOOGLE_SEARCH_LIMIT):
            params = {
                "q": variant,
                "key": api_key,
                "cx": cse_id,
                "searchType": "image",
                "num": GOOGLE_SEARCH_LIMIT,
                "start": start,
            }

            try:
                response = requests.get("https://customsearch.googleapis.com/customsearch/v1", params=params)
                
                # debug
                # print(response.url)
                # print(response.text)

                data = response.json()
                items = data.get("items", [])
            except Exception as e:
                print(f"API error: {e}")
                continue

            for item in items:
                img_url = item.get("link")
                if not img_url:
                    continue
                try:
                    img_data = requests.get(img_url, timeout=5).content
                    unique_id = random.randint(0, 100000)
                    filename = save_dir / f"{variant.replace(' ', '_')}_{unique_id}.jpg"
                    with open(filename, 'wb') as f:
                        f.write(img_data)
                    downloaded += 1
                    total_downloaded += 1
                    print(f"[{class_name}] Downloaded: {img_url}")
                except Exception as e:
                    print(f"[{class_name}] Error downloading {img_url}: {e}")

            time.sleep(1)
    print(f"[{class_name}] Total downloaded: {total_downloaded}")

classes = {
    # "robot": ["robot face", "android head", "cyberpunk robot", "humanoid robot"],
    "person": ["human face", "man portrait", "professional profile picture", "person profile photo", "football player"]
}

if __name__=="__main__":
    api_key, cse_id = read_keys()
    
    for class_name, variants in classes.items():
        google_image_search(class_name, variants, api_key, cse_id, root_dir="images", images_per_variant=30)
